"""
Download RFEv2 daily rainfall and regrid onto the EXACT IFS / constants grid.

Source : ftp://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/rfe2/geotiff/
Target : the 384x352 grid of elev.nc / lsm.nc / the IFS forecasts
         (lat -13.65..24.65, lon 19.15..54.25, 0.1 deg)
Output : E:/CGAN/RFE/YYYY/YYYYMMDD.nc  (variable: precipitation, units: mm/day)

Why regrid instead of clip:
  The raw africa_rfe tif covers all of Africa. We interpolate it onto the
  *same* lat/lon as elev.nc so that forecast, truth and constants register
  cell-for-cell. The output coords are copied verbatim from elev.nc, so there
  is no possibility of a rounding/offset mismatch.

Notes:
  * Atomic write: each file is written to a .tmp, validated, then os.replace'd.
    An interrupted run can never leave a truncated stub at the final path.
  * Skip rule: a date is skipped only if an existing file is already 384x352.
    Old 349x297 files are therefore re-fetched and overwritten automatically.
  * Default range is 2018-2021 (the training years). Widen START/END if needed.
"""

import ftplib, zipfile, os, io, time
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import netCDF4 as nc_lib
import xarray as xr
import rasterio
from scipy.interpolate import RegularGridInterpolator

# ---- target grid: copied verbatim from elev.nc so everything matches ----
ELEV_PATH = r"E:\CGAN\SEWAA-forecasts\cGAN_data\elev.nc"
_elev       = xr.open_dataset(ELEV_PATH)
TARGET_LATS = _elev["lat"].values.astype("float64")   # ascending: -13.65 .. 24.65
TARGET_LONS = _elev["lon"].values.astype("float64")   # ascending:  19.15 .. 54.25
_elev.close()
TARGET_H, TARGET_W = len(TARGET_LATS), len(TARGET_LONS)   # 384, 352
# precompute the sampling mesh once
_LON_GRID, _LAT_GRID = np.meshgrid(TARGET_LONS, TARGET_LATS)   # both (384, 352)

FTP_HOST  = "ftp.cpc.ncep.noaa.gov"
FTP_DIR   = "/fews/fewsdata/africa/rfe2/geotiff/"
OUT_DIR   = r"E:\CGAN\RFE"
WORKERS   = 1                      # single-threaded to avoid segfault
MIN_BYTES = 1000
START     = date(2018, 1, 1)       # training years; widen if you need more
END       = date(2021, 12, 31)


def tif_to_nc(tif_bytes, out_path, date_str):
    # --- read the full africa tif ---
    with rasterio.MemoryFile(tif_bytes) as memfile:
        with memfile.open() as src:
            data = src.read(1).astype("float64")
            tr = src.transform
            nodata = src.nodata
            H, W = data.shape
            src_lons = tr.c + (np.arange(W) + 0.5) * tr.a      # ascending (a>0)
            src_lats = tr.f + (np.arange(H) + 0.5) * tr.e      # descending (e<0)

    # clean nodata / negatives -> 0 (rainfall, ocean treated as dry)
    if nodata is not None:
        data = np.where(data == nodata, 0.0, data)
    data = np.where(np.isfinite(data), data, 0.0)
    data = np.maximum(data, 0.0)

    # RegularGridInterpolator needs ascending coords
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        data = data[::-1, :]

    # --- regrid onto the exact elev.nc grid ---
    interp = RegularGridInterpolator((src_lats, src_lons), data,
                                     method="linear",
                                     bounds_error=False,
                                     fill_value=0.0)
    y = interp((_LAT_GRID, _LON_GRID))          # (384, 352), ascending lat
    y = np.maximum(y, 0.0).astype("float32")

    # --- atomic write ---
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    try:
        ds = nc_lib.Dataset(tmp_path, "w", format="NETCDF4")
        ds.createDimension("lat", TARGET_H)
        ds.createDimension("lon", TARGET_W)
        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lon_var = ds.createVariable("lon", "f4", ("lon",))
        prcp    = ds.createVariable("precipitation", "f4", ("lat", "lon"),
                                    fill_value=-9999.0, zlib=True, complevel=4)
        lat_var.units = "degrees_north"; lat_var.long_name = "latitude"
        lon_var.units = "degrees_east";  lon_var.long_name = "longitude"
        prcp.units = "mm/day"; prcp.long_name = "daily precipitation"
        lat_var[:] = TARGET_LATS
        lon_var[:] = TARGET_LONS
        prcp[:] = y
        ds.source = "RFEv2"; ds.date = date_str
        ds.region = "ICPAC_IFSgrid"; ds.regrid = "linear -> elev.nc grid"
        ds.close()

        # validate before promoting
        with nc_lib.Dataset(tmp_path, "r") as chk:
            if chk["precipitation"].shape != (TARGET_H, TARGET_W):
                raise ValueError("written shape mismatch")
        if os.path.getsize(tmp_path) < MIN_BYTES:
            raise ValueError(f"written file too small ({os.path.getsize(tmp_path)} bytes)")

        os.replace(tmp_path, out_path)          # atomic on same filesystem
    except Exception:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass
        raise


def _is_on_target_grid(path):
    """Skip only if the existing file is already on the 384x352 target grid."""
    if not os.path.exists(path) or os.path.getsize(path) < MIN_BYTES:
        return False
    try:
        with nc_lib.Dataset(path, "r") as ds:
            return ds["precipitation"].shape == (TARGET_H, TARGET_W)
    except Exception:
        return False


def download_one(date_str, available):
    year = date_str[:4]
    out_path = os.path.join(OUT_DIR, year, f"{date_str}.nc")
    if _is_on_target_grid(out_path):
        return date_str, "SKIP"
    # clear any stub / old-grid file / stale tmp so we write clean
    for p in (out_path, out_path + ".tmp"):
        if os.path.exists(p):
            try: os.remove(p)
            except OSError: pass

    filename = f"africa_rfe.{date_str}.tif.zip"
    if filename not in available:
        return date_str, "MISS"
    for attempt in range(3):
        try:
            ftp = ftplib.FTP(FTP_HOST, timeout=60)
            ftp.login(); ftp.cwd(FTP_DIR)
            buf = io.BytesIO()
            ftp.retrbinary(f"RETR {filename}", buf.write)
            ftp.quit(); buf.seek(0)
            with zipfile.ZipFile(buf) as zf:
                tif_bytes = zf.read([n for n in zf.namelist() if n.endswith(".tif")][0])
            tif_to_nc(tif_bytes, out_path, date_str)
            return date_str, "OK"
        except Exception as e:
            if attempt < 2: time.sleep(10)
            else: return date_str, f"ERR: {e}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Target grid: {TARGET_H}x{TARGET_W}  "
          f"lat {TARGET_LATS.min():.2f}..{TARGET_LATS.max():.2f}  "
          f"lon {TARGET_LONS.min():.2f}..{TARGET_LONS.max():.2f}", flush=True)
    print("Fetching FTP file list...", flush=True)
    available = None
    for attempt in range(10):
        try:
            ftp = ftplib.FTP(FTP_HOST, timeout=90); ftp.login(); ftp.cwd(FTP_DIR)
            available = set(ftp.nlst()); ftp.quit()
            break
        except Exception as e:
            print(f"  FTP list attempt {attempt+1} failed: {e} — retrying in 30s...", flush=True)
            time.sleep(30)
    if available is None:
        print("Could not fetch FTP file list after 10 attempts. Aborting."); return

    dates = []; d = START
    while d <= END:
        dates.append(d.strftime("%Y%m%d")); d += timedelta(days=1)
    print(f"Total dates: {len(dates)}, workers: {WORKERS}", flush=True)

    downloaded = skipped = missing = errors = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(download_one, d, available): d for d in dates}
        for f in as_completed(futures):
            ds, status = f.result()
            if status == "OK":   print(f"  [OK]   {ds}", flush=True); downloaded += 1
            elif status == "SKIP": skipped += 1
            elif status == "MISS": print(f"  [MISS] {ds}", flush=True); missing += 1
            else: print(f"  [{status}] {ds}", flush=True); errors += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {missing} missing, {errors} errors.")


if __name__ == "__main__":
    main()
