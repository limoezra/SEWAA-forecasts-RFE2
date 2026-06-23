def load_truth_and_mask(date,
                        time_idx,
                        log_precip=False):
    '''
    Returns a single (truth, mask) item of RFE2 daily rainfall.

    RFE2 is now pre-regridded (by the downloader) onto the exact IFS / constants
    384x352 grid, so NO interpolation is needed here -- we read it directly.
    This keeps truth, forecast and constants registered cell-for-cell.

    Parameters:
        date: forecast start date (YYYYMMDD string)
        time_idx: unused for daily data, kept for API compatibility
        log_precip: whether to apply log10(1+x) transformation
    '''
    fcst_date = datetime.datetime.strptime(date, "%Y%m%d")
    truth_dt = fcst_date + datetime.timedelta(hours=int(LEAD_IDX) * HOURS)
    datestr = truth_dt.strftime('%Y%m%d')
    data_path = os.path.join(TRUTH_PATH, str(truth_dt.year), f"{datestr}.nc")

    ds = xr.open_dataset(data_path)
    y = ds["precipitation"].values.squeeze()   # already (384, 352) on the IFS grid
    ds.close()

    y = np.where(np.isfinite(y), y, 0.0)        # ocean / missing -> 0
    y = np.maximum(y, 0.0)                       # clip tiny negatives from processing
    y = y / 24.0                                 # mm/day -> mm/hr to match IFS tp units

    mask = np.full(y.shape, False, dtype=bool)   # all valid; boundary handled at eval

    if log_precip:
        return np.log10(1 + y), mask
    else:
        return y, mask
