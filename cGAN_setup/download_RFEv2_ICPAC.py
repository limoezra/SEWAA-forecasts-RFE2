"""
Download RFEv2 daily rainfall for the ICPAC region (2001-01-01 to present)
Source: ftp://ftp.cpc.ncep.noaa.gov/fews/fewsdata/africa/rfe2/geotiff/
Output: E:/CGAN/RFE/YYYY/YYYYMMDD.nc  (variable: precipitation, units: mm/day)

PATCHED:
  * tif_to_nc now writes to a .tmp file, validates it, then atomically
    renames to the final path. An interrupted write can no longer leave a
    truncated 48-byte stub at YYYYMMDD.nc.
  * download_one skips a date only if an existing file is a plausible size
    (>= MIN_BYTES). Stubs / partials are deleted and re-fetched instead of
    being silently SKIPped forever.
"""

import ftplib, zipfile, os, io, time
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import netCDF4 as nc_lib
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import unary_union

SHP        = r"E:\CGAN\Shapefiles\ICPAC_REGIONAL\ICPAC_ADM0.shp"
_gdf       = gpd.read_file(SHP)
ICPAC_GEOM = [mapping(unary_union(_gdf.geometry))]

FTP_HOST  = "ftp.cpc.ncep.noaa.gov"
FTP_DIR   = "/fews/fewsdata/africa/rfe2/geotiff/"
OUT_DIR   = r"E:\CGAN\RFE"
WORKERS   = 16
START     = date(2001, 1, 1)
END       = date.today()
MIN_BYTES = 1000   # a real daily file is tens of KB; anything smaller is a stub


def tif_to_nc(tif_bytes, out_path, date_str):
    with rasterio.MemoryFile(tif_bytes) as memfile:
        with memfile.open() as src:
            out_image, out_transform = mask(src, ICPAC_GEOM, crop=True, nodata=-9999)
            data = out_image[0].astype("float32")
            height, width = data.shape
            # compute lat/lon arrays from transform
            lons = np.array([out_transform.c + (i + 0.5) * out_transform.a for i in range(width)])
            lats = np.array([out_transform.f + (j + 0.5) * out_transform.e for j in range(height)])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    try:
        # --- write to a temp file, never directly to the final path ---
        ds = nc_lib.Dataset(tmp_path, "w", format="NETCDF4")
        ds.createDimension("lat", height)
        ds.createDimension("lon", width)
        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lon_var = ds.createVariable("lon", "f4", ("lon",))
        prcp    = ds.createVariable("precipitation", "f4", ("lat", "lon"),
                                    fill_value=-9999.0, zlib=True, complevel=4)
        lat_var.units = "degrees_north"; lat_var.long_name = "latitude"
        lon_var.units = "degrees_east";  lon_var.long_name = "longitude"
        prcp.units = "mm/day"; prcp.long_name = "daily precipitation"
        prcp.missing_value = np.float32(-9999.0)
        lat_var[:] = lats
        lon_var[:] = lons
        prcp[:] = np.where(data == -9999, -9999.0, data)
        ds.source = "RFEv2"
        ds.date   = date_str
        ds.region = "ICPAC"
        ds.close()

        # --- validate the temp file before promoting it ---
        with nc_lib.Dataset(tmp_path, "r") as check:
            if check["precipitation"].shape != (height, width):
                raise ValueError("written precipitation shape mismatch")
        if os.path.getsize(tmp_path) < MIN_BYTES:
            raise ValueError(f"written file too small ({os.path.getsize(tmp_path)} bytes)")

        # --- atomic promote: os.replace is atomic on the same filesystem ---
        os.replace(tmp_path, out_path)
    except Exception:
        # never leave a partial temp file lying around
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def download_one(date_str, available):
    year = date_str[:4]
    out_path = os.path.join(OUT_DIR, year, f"{date_str}.nc")

    # skip ONLY if an existing file looks complete; otherwise clear and re-fetch
    if os.path.exists(out_path) and os.path.getsize(out_path) >= MIN_BYTES:
        return date_str, "SKIP"
    for p in (out_path, out_path + ".tmp"):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

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
