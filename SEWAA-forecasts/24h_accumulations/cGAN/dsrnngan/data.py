""" File for handling data loading and saving. """
import os
import datetime
import pickle

import numpy as np
import netCDF4 as nc
import xarray as xr

import read_config


data_paths = read_config.get_data_paths()
TRUTH_PATH = data_paths["GENERAL"]["TRUTH_PATH"]
FCST_PATH = data_paths["GENERAL"]["FORECAST_PATH"]
CONSTANTS_PATH = data_paths["GENERAL"]["CONSTANTS_PATH"]
NORMALISATION_PATH = data_paths["GENERAL"]["NORMALISATION_PATH"]

# Which lead time (Start of 24h accumulation period) are we using?
LEAD_IDX = data_paths["GENERAL"]["LEAD_IDX"]

# IMERG 13-field originals (commented for RFE2 adaptation):
# all_fcst_fields = ['cp', 'mcc', 'sp', 'ssr', 't2m', 'tciw', 'tclw', 'tcrw', 'tcw', 'tcwv', 'tp', 'u700', 'v700']
# accumulated_fields = ['cp', 'ssr', 'tp']
# nonnegative_fields = ['cp', 'mcc', 'sp', 'ssr', 't2m', 'tciw', 'tclw', 'tcrw', 'tcw', 'tcwv', 'tp']
# HOURS = 6  # 6-hr IMERG data

all_fcst_fields = ['tp', 't2m', 'tcwv', 'sp']
accumulated_fields = ['tp']
nonnegative_fields = ['tp', 'tcwv']

HOURS = 24  # daily RFE2 data


# utility function; generator to iterate over a range of dates
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(days=n)


def denormalise(x):
    """
    Undo log-transform of rainfall.  Also cap at 100 (feel free to adjust according to application!)
    """
    return np.minimum(10**x - 1.0, 100.0)


def logprec(y, log_precip=False):
    if log_precip:
        return np.log10(1.0+y)
    else:
        return y


def get_dates(year,
              start_hour,
              end_hour):
    '''
    Returns list of valid forecast start dates for which RFE2 truth data
    exists. For each IFS forecast date D, truth is the RFE2 file for D+1.
    Dates are returned as a list of YYYYMMDD strings.

    Parameters:
        year (int): forecasts starting in this year
        start_hour (int): Lead time of first forecast desired
        end_hour (int): Lead time of last forecast desired
    '''
    assert year in (2018, 2019, 2020, 2021)
    assert start_hour >= 0
    assert end_hour <= 168
    assert start_hour % HOURS == 0
    assert end_hour % HOURS == 0
    assert end_hour >= start_hour

    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year + 1, 1, 1)
    valid_dates = []

    for curdate in daterange(start_date, end_date):
        # truth file is the RFE2 accumulation for the day after the forecast date
        truth_date = curdate + datetime.timedelta(days=1)
        truth_path = os.path.join(TRUTH_PATH, str(truth_date.year),
                                  truth_date.strftime('%Y%m%d') + '.nc')
        if os.path.exists(truth_path):
            valid_dates.append(curdate.strftime('%Y%m%d'))

    return valid_dates


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


def load_hires_constants(batch_size=1):
    oro_path = os.path.join(CONSTANTS_PATH, "elev.nc")
    df = xr.load_dataset(oro_path)
    # Orography in m.  Divide by 10,000 to give O(1) normalisation
    z = df["elevation"].values
    z /= 10000.0
    df.close()

    lsm_path = os.path.join(CONSTANTS_PATH, "lsm.nc")
    df = xr.load_dataset(lsm_path)
    # LSM is already 0:1
    lsm = df["lsm"].values
    df.close()

    temp = np.stack([z, lsm], axis=-1)  # shape H x W x 2
    return np.repeat(temp[np.newaxis, ...], batch_size, axis=0)  # shape batch_size x H x W x 2


def load_fcst_truth_batch(dates_batch,
                          time_idx_batch,
                          fcst_fields=all_fcst_fields,
                          log_precip=False,
                          norm=False):
    '''
    Returns a batch of (forecast, truth, mask) data, although usually the batch size is 1
    Parameters:
        dates_batch (iterable of strings): Dates of forecasts
        time_idx_batch (iterable of ints): Corresponding 'valid_time' array indices
        fcst_fields (list of strings): The fields to be used
        log_precip (bool): Whether to apply log10(1+x) transform to precip-related forecast fields, and truth
        norm (bool): Whether to apply normalisation to forecast fields to make O(1)
    '''
    batch_x = []  # forecast
    batch_y = []  # truth
    batch_mask = []  # mask

    for time_idx, date in zip(time_idx_batch, dates_batch):
        batch_x.append(load_fcst_stack(fcst_fields, date, time_idx, log_precip=log_precip, norm=norm))
        truth, mask = load_truth_and_mask(date, time_idx, log_precip=log_precip)
        batch_y.append(truth)
        batch_mask.append(mask)

    return np.array(batch_x), np.array(batch_y), np.array(batch_mask)


def load_fcst(field,
              date,
              time_idx,
              log_precip=False,
              norm=False):
    '''
    Returns forecast field data for the given date and time interval.

    Four channels are returned for each field:
        - instantaneous fields: mean and stdev at the start of the interval, mean and stdev at the end of the interval
        - accumulated field: mean and stdev of increment over the interval, and the last two channels are all 0
    '''

    # print(f"Loading forecast {field} on {date}")

    # First index in the lead time
    lead_idx = LEAD_IDX

    yearstr = date[:4]
    year = int(yearstr)
    ds_path = os.path.join(FCST_PATH, yearstr, f"{field}.nc")

    # open using netCDF
    nc_file = nc.Dataset(ds_path, mode="r")
    all_data_mean = nc_file[f"{field}_mean"]
    all_data_sd = nc_file[f"{field}_sd"]
    # data is stored as [day of year, valid time index, lat, lon]

    # calculate first index (i.e., day of year, with Jan 1 = 0)
    fcst_date = datetime.datetime.strptime(date, "%Y%m%d").date()
    fcst_idx = fcst_date.toordinal() - datetime.date(year, 1, 1).toordinal()

    if field in accumulated_fields:
        # return mean, sd, 0, 0.  zero fields are so that each field returns a 4 x ny x nx array.
        # accumulated fields have been pre-processed s.t. data[:, j, :, :] has accumulation between times j and j+1
        data1 = np.mean(all_data_mean[fcst_idx, lead_idx:lead_idx+4, :, :], axis=0)            # Mean of the accumulations
        data2 = np.sqrt(np.mean(all_data_sd[fcst_idx, lead_idx:lead_idx+4, :, :]**2, axis=0))  # RMS of the standard deviations
        data = np.stack([data1, data2], axis=-1)
    else:
        # return mean and std computed using the trapezium rule
        temp_data_mean = all_data_mean[fcst_idx, lead_idx:lead_idx+5, :, :]
        temp_data_var = all_data_sd[fcst_idx, lead_idx:lead_idx+5, :, :]**2  # Convert to variances
        data1 = (temp_data_mean[0, :, :]/2 + np.sum(temp_data_mean[1:4,:,:], axis=0) + temp_data_mean[4,:,:]/2)/4
        data2 = (temp_data_var[0, :, :]/2 + np.sum(temp_data_var[1:4,:,:], axis=0) + temp_data_var[4,:,:]/2)/4
        data = np.stack([data1, np.sqrt(data2)], axis=-1)

    nc_file.close()

    if field in nonnegative_fields:
        data = np.maximum(data, 0.0)  # eliminate any data weirdness/regridding issues

    if field in ["tp", "cp"]:
        # precip is measured in metres, so multiply to get mm
        data *= 1000
        data /= HOURS  # convert to mm/hr
    elif field in accumulated_fields:
        # for all other accumulated fields [just ssr for us]
        data /= (HOURS*3600)  # convert from a 6-hr difference to a per-second rate

    if field in ["tp", "cp"] and log_precip:
        return logprec(data, log_precip)
    elif norm:
        # apply transformation to make fields O(1), based on historical
        # forecast data from one of the training years
        if fcst_norm is None:
            raise RuntimeError("Forecast normalisation dictionary has not been loaded")
        if field in ["mcc"]:
            # already 0-1
            return data
        elif field in ["sp", "t2m"]:
            # these are bounded well away from zero, so subtract mean from ens mean (but NOT from ens sd!)
            data[:, :, 0] -= fcst_norm[field]["mean"]
            return data/fcst_norm[field]["std"]
        elif field in nonnegative_fields:
            return data/fcst_norm[field]["max"]
        else:
            # winds
            return data/max(-fcst_norm[field]["min"], fcst_norm[field]["max"])
    else:
        return data


def load_fcst_stack(fields,
                    date,
                    time_idx,
                    log_precip=False,
                    norm=False):
    '''
    Returns forecast fields, for the given date and time interval.
    Each field returned by load_fcst has two channels (see load_fcst for details),
    then these are concatentated to form an array of H x W x 4*len(fields)
    '''
    field_arrays = []
    for f in fields:
        field_arrays.append(load_fcst(f, date, time_idx, log_precip=log_precip, norm=norm))
    return np.concatenate(field_arrays, axis=-1)


def get_fcst_stats_slow(field, year=2018):
    '''
    Calculates and returns min, max, mean, std per field,
    which can be used to generate normalisation parameters.

    These are done via the data loading routines, which is
    slightly inefficient.
    '''
    dates = get_dates(year, start_hour=0, end_hour=24)

    mi = 0.0
    mx = 0.0
    dsum = 0.0
    dsqrsum = 0.0
    nsamples = 0
    for datestr in dates:
        for time_idx in range(28):
            data = load_fcst(field, datestr, time_idx)[:, :, 0]
            mi = min(mi, data.min())
            mx = max(mx, data.max())
            dsum += np.mean(data)
            dsqrsum += np.mean(np.square(data))
            nsamples += 1
    mn = dsum / nsamples
    sd = (dsqrsum/nsamples - mn**2)**0.5
    return mi, mx, mn, sd


def get_fcst_stats_fast(field, year=2018):
    '''
    Calculates and returns min, max, mean, std per field,
    which can be used to generate normalisation parameters.

    These are done directly from the forecast netcdf file,
    which is somewhat faster, as long as it fits into memory.
    '''
    ds_path = os.path.join(FCST_PATH, str(year), f"{field}.nc")
    nc_file = nc.Dataset(ds_path, mode="r")

    if field in accumulated_fields:
        data = nc_file[f"{field}_mean"][:, :-1, :, :]  # last time_idx is full of zeros
    else:
        data = nc_file[f"{field}_mean"][:, :, :, :]

    nc_file.close()

    if field in ["tp", "cp"]:
        # precip is measured in metres, so multiply to get mm
        data *= 1000
        data /= HOURS  # convert to mm/hr
        data = np.maximum(data, 0.0)  # shouldn't be necessary, but just in case
    elif field in accumulated_fields:
        # for all other accumulated fields [just ssr for us]
        data /= (HOURS*3600)  # convert from a 6-hr difference to a per-second rate

    mi = data.min()
    mx = data.max()
    mn = np.mean(data, dtype=np.float64)
    sd = np.std(data, dtype=np.float64)
    return mi, mx, mn, sd


def gen_fcst_norm(year=2018):
    '''
    One-off function, used to generate normalisation constants, which
    are used to normalise the various input fields for training/inference.
    '''

    stats_dic = {}
    fcstnorm_path = os.path.join(NORMALISATION_PATH, f"FCSTNorm{year}.pkl")

    # make sure we can actually write there, before doing computation!!!
    with open(fcstnorm_path, 'wb') as f:
        pickle.dump(stats_dic, f)

    for field in all_fcst_fields:
        print(field)
        mi, mx, mn, sd = get_fcst_stats_fast(field, year)
        stats_dic[field] = {}
        stats_dic[field]['min'] = mi
        stats_dic[field]['max'] = mx
        stats_dic[field]['mean'] = mn
        stats_dic[field]['std'] = sd

    with open(fcstnorm_path, 'wb') as f:
        pickle.dump(stats_dic, f)


def load_fcst_norm(year=2018):
    print("In load_fcst_norm")
    fcstnorm_path = os.path.join(NORMALISATION_PATH, f"FCSTNorm{year}.pkl")
    print(f"fcstnorm_path = {fcstnorm_path}")
    with open(fcstnorm_path, 'rb') as f:
        return pickle.load(f)


try:
    print("Loading forecast normalisations")
    fcst_norm = load_fcst_norm(2018)
except:  # noqa
    fcst_norm = None
    print("******************************************")
    print("*** FORECAST NORMALISATIONS NOT LOADED ***")
    print("******************************************")
