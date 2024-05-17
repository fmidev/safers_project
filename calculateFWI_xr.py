"""
Calculate FWI
"""
import datetime as dt
import calendar
import logging

import xarray as xr
import numpy as np

from FWI_functions_xr import vFFMCcalc, vDCcalc, vDMCcalc
from FWI_functions_xr import vISIcalc, vBUIcalc, vFWIcalc
from FWI_functions_xr import interpolate_grid
from FWI_functions_xr import modified_ordinal_day
from FWI_functions_xr import add_daylength

from safers_data_tables import safers_attrs
from safers_data_tables import int16_minvalue, int16_maxvalue

import safers_utils as utils

from safers_grib import opengrib

# import warnings
# warnings.filterwarnings('ignore')
# np.seterr(invalid='ignore', over='ignore', divide='ignore')

# era5climdata = '/data/safers/data/FWI_FFMC_DMC_DC_ERA5_1980_2019_mean.nc'
# clim_mean = '/data/safers/data/FWI_FFMC_DMC_DC_ERA5_1980_2019_mean.nc'
# clim_std = '/data/safers/data/FWI_FFMC_DMC_DC_ERA5_1980_2019_std.nc'


def calculate_FWI_ENS_1(file, fctype, inumber, mapdata, chunks=None,
                        ncdir='/data/tmp/safers_nc',
                        ffmc0=85.0, dc0=15.0, dmc0=6.0,
                        level='WARNING', log='stderr'):
    """Process one realization in grib file.
    Intended to be used in multiprocessing.
    """
    # These are needed in multiprocessing
    # logger = logging.getLogger()
    # logger.setLevel(level)
    if log == 'stderr':
        logging.basicConfig(level=getattr(logging, level.upper(), None))
    else:
        logging.basicConfig(level=getattr(logging, level.upper(), None),
                            filename=log)
    logging.debug('opening %d', inumber)
    np.seterr(invalid='ignore', over='ignore', divide='ignore')
    #
    variables = ['t2m', 'd2m', 'u10', 'v10', 'tp']
    # need to have minstep > 0
    ds = opengrib(file, type=fctype, variable=variables, minstep=3,
                  chunks=chunks, ntime=0, redim=True, convert=False,
                  number=inumber).load()
    logging.debug('opened %d', inumber)
    if (ds is None) or (not np.all(np.isin(['t2m'], list(ds.data_vars)))):
        logging.warning('Member %d does not exist', inumber)
        return None
    ds = utils.safers_convert(ds)
    ds = utils.addrh(ds, units='C', limit=True)
    ds = utils.addtp24(ds, drop=False)
    # some problem here!!!
    # ds = utils.uvtows(ds, drop=True, dwi=False))
    # ds['ws10'] = ds['ws10'] * 3.6  #  # m/s to km/h
    ds['ws'] = np.sqrt(np.square(ds['u10']) + np.square(ds['v10'])) * 3.6
    ds = ds.drop_vars(['u10', 'v10', 'd2m'])
    ds = utils.rotated_ll_to_regular(ds, realization='time', fctype=fctype,
                                     lon='longitude', lat='latitude')
    ds = utils.addprevtp24(ds, ncdir, variable='tp24')
    ds = dstonoon(ds, mapdata)
    #   ds = ds.where(mapdata.lsm != 0)
    fwi = calculate_FWI_ds(ds, ffmc0, dc0, dmc0).load()
    fwi = fwi.where(mapdata.lsm != 0)
    logging.info('done realization %d', inumber)
    return fwi


def calculate_FWI_1(ds, ffmc0, dc0, dmc0, dtype=np.float32):
    """FWI calculation for one time."""
    fwids = xr.Dataset(coords=ds.coords)
    fwids.attrs = ds.attrs

    ffmc = vFFMCcalc(ds, ffmc0)
    dc = vDCcalc(ds, dc0)
    dmc = vDMCcalc(ds, dmc0)
    isi = vISIcalc(ds, ffmc)
    bui = vBUIcalc(dmc, dc)
    fwi = vFWIcalc(isi, bui)
    #
    fwids['fwi'] = fwi.astype(dtype)
    fwids['isi'] = isi.astype(dtype)
    fwids['bui'] = bui.astype(dtype)
    fwids['dc'] = dc.astype(dtype)
    fwids['dmc'] = dmc.astype(dtype)
    fwids['ffmc'] = ffmc.astype(dtype)

    # attributes
    fwids = safers_attrs(fwids)
    # fwids = fwids.assign_coords({'time': ds['time']})
    # fwids['time'] = ds['time']

    fwids = fwids.drop_vars(['surface', 'depth', 'entireAtmosphere',
                             'quantile', 'time_coord'],
                            errors='ignore')

    return fwids


def calculate_FWI_ds(ds, ffmc0, dc0, dmc0):
    """FWI calculations over all times in Dataset"""
    fwis = []
    ffmc0['time'] = ds.isel(time=[0])['time']
    dc0['time'] = ds.isel(time=[0])['time']
    dmc0['time'] = ds.isel(time=[0])['time']

    for i in range(len(ds.time)):
        dsi = ds.isel(time=[i])
        if i > 0:
            ffmc0 = fwi['ffmc'].assign_coords({'time': dsi['time']}).copy()
            dc0 = fwi['dc'].assign_coords({'time': dsi['time']}).copy()
            dmc0 = fwi['dmc'].assign_coords({'time': dsi['time']}).copy()
        fwi = calculate_FWI_1(dsi, ffmc0, dc0, dmc0)
        fwis.append(fwi)
    fwids = xr.concat(fwis, dim='time')
    fwids['time'] = ds['time']
    # print(fwids.time.attrs)
    return fwids


def isdst(t):
    """Is time within European daylight saving time.
    European Summer time (EST) last sunday in March until last sunday in October
    """
    EST_start = dt.datetime(int(t.year), 3,
                            max(calendar.monthcalendar(t.year, 3)[-1][calendar.SUNDAY],
                                calendar.monthcalendar(t.year, 3)[-2][calendar.SUNDAY]))
    EST_end = dt.datetime(int(t.year), 10,
                          max(calendar.monthcalendar(t.year, 10)[-1][calendar.SUNDAY],
                              calendar.monthcalendar(t.year, 10)[-2][calendar.SUNDAY]))

    return ((dt.datetime(int(t.year), int(t.month), int(t.day)) > EST_start) and
            (dt.datetime(int(t.year), int(t.month), int(t.day)) < EST_end))


def noon_calc(ds, time, mapdata, dst=True):
    """
    calculate local noon for each location
    Negative/westerly offset
    takes the time zones into account
    """
    t = time.dt
    if dst:
        utc_offset = mapdata.offset_summer
    else:
        utc_offset = mapdata.offset_winter
    # local noon at 12:00 LT
    timearray = np.datetime64(dt.datetime(int(t.year), int(t.month),
                                          int(t.day),
                                          12, 0, 0))
    localnoon = timearray - utc_offset
    # localnoon_nearest = dst.time.sel(time=localnoon.flatten(),
    #                                 method='nearest').values.reshape(localnoon.shape)
    # mapdata = mapdata.assign(localnoon=(('latitude', 'longitude'), localnoon)) #,
    #                 closest2localnoon=(('latitude', 'longitude'), localnoon_nearest))

    datasets = []
    for ti in np.unique(localnoon):
        ds1 = ds.interp(time=ti, assume_sorted=True)
        ds1['time'] = time
        ds1 = ds1.where(localnoon == ti)
        datasets.append(ds1)
    ds3 = xr.merge(datasets)
    ds3 = ds3.assign_coords({'time': np.array(time)})
    return ds3


def dstonoon(ds, mapdata, times=None):
    """Interpolate to local noon."""
    dsall = []
    if times is None:
        times = ds['time'][ds['time'].dt.hour == 12]
        # times = ds.time.where(ds.time.dt.time == dt.time(12),
        #            drop=True).values.astype('datetime64[h]')

    for t in times:
        dsi = noon_calc(ds, t, mapdata, dst=True)
        dsall.append(dsi)
    dsall = xr.concat(dsall, dim='time')
    dsall['time'].attrs = ds['time'].attrs
    dsall = add_daylength(dsall, ds.time.dt.month.values[0])  # month HERE!!
    return dsall


# not ready
def init_effis(ds, effis):
    """Initialize FWI calculations using EFFIS"""
    ds_pd = xr.open_dataset(effis, engine='netcdf4')
    ds_pd = interpolate_grid(ds_pd, ds)
    previous_day = ((ds.forecast_reference_time.values.
                     astype('datetime64[s]') - np.timedelta64(1, 'D')).
                    tolist().strftime('%Y-%m-%dT12:00:00.000000000'))
    # select which forecast from previous days origin time to use
    ds_pd = ds_pd.sel(time=previous_day)
    FFMC0 = ds_pd['ffmc']
    DC0 = ds_pd['dc']
    DMC0 = ds_pd['dmc']
    logging.info('Use EFFIS values from %s as inital values, time: %s',
                 effis, previous_day)
    return FFMC0, DC0, DMC0


def init_era5(ds, era5climdata='/data/safers/data/FWI_FFMC_DMC_DC_ERA5_1980_2019_mean.nc'):
    """Initialize using era5 climatology."""
    era5 = xr.open_dataset(era5climdata)
    time0 = ds['forecast_reference_time'].copy()
    iday = time0.dt.dayofyear.values - 1
    if not time0.dt.is_leap_year and time0.dt.month > 3:
        iday = iday + 1
    ds_pd = (era5.isel(ordinal_day=iday).
             reindex_like(ds.isel(time=0), method='nearest'))

    ds_pd = ds_pd.expand_dims({'time': ds['time']})

    FFMC0 = ds_pd['ffmc']
    DC0 = ds_pd['dc']
    DMC0 = ds_pd['dmc']
    logging.info('Use ERA5 day=%d values as inital values', iday)
    return FFMC0, DC0, DMC0


def init_prev(ds, datadir='/data/tmp/safers_fwi', fctype='ENS', fail='era5'):
    """Initialize using previous FWI."""

    # noon previous day
    time0 = ds['forecast_reference_time'].copy()
    time0 = time0 - np.timedelta64(12 + time0.dt.hour.values, 'h')

    if ds['forecast_reference_time'].dt.hour.values != 0:
        ot = 24 + time0.dt.hour.values
    else:
        ot = 24

    prevfile = utils.prevfcfile(ds=ds, ncdir=datadir, prefix='Fwi_',
                                fctype=fctype, olderthan=ot)
    if prevfile is None:
        logging.info('Previous FWI data not foud')
        if fail == 'era5':
            return init_era5(ds)
        else:
            return init_default()

    dsprev = xr.open_dataset(prevfile)
    # THIS CAN FAIL!!
    try:
        dst_pd = dsprev.sel(time=[time0.values])
    except Exception:
        logging.info('Previous FWI time %s not foud in %s',
                     time0.values, prevfile)
        if fail == 'era5':
            return init_era5(ds)
        else:
            return init_default()

    # reindex needed for old 32 bit data coordinates
    FFMC0 = dst_pd['ffmc'].reindex_like(ds.isel(time=0), method='nearest')
    DC0 = dst_pd['dc'].reindex_like(ds.isel(time=0), method='nearest')
    DMC0 = dst_pd['dmc'].reindex_like(ds.isel(time=0), method='nearest')

    logging.info('Previous FWI data in %s, time: %s', prevfile, time0.values)
    return FFMC0, DC0, DMC0


def init_default():
    """Default initialization."""
    FFMC0 = 85.0
    DC0 = 15.0
    DMC0 = 6.0
    return FFMC0, DC0, DMC0


# does not work
def init_fwi(method='default', time0=None):
    """FWI initialization."""
    if method == 'default':
        FFMC0, DC0, DMC0 = init_default()
    elif method == 'era5':
        FFMC0, DC0, DMC0 = init_era5(time0)  # does not work yet
    return FFMC0, DC0, DMC0


def add_anomalies(dst,
                  clim_mean='/data/safers/data/FWI_FFMC_DMC_DC_ERA5_1980_2019_mean.nc',
                  clim_std='/data/safers/data/FWI_FFMC_DMC_DC_ERA5_1980_2019_std.nc'):
    """
    dst: FWI dataset
    clim: ERA5 climatology distribution with years as one dimension
    calculates the anomaly by subtracting climatology (interp to grid) from
    fwi values
    standardized anomaly : [observation - mean(time series)] /
                             standard_deviation(time_series)
    """
    ds = dst.assign_coords(mod_ordinal_day=modified_ordinal_day(dst.time))
    ds = ds.swap_dims({'time': 'mod_ordinal_day'})

    mean = xr.open_dataset(clim_mean).sel(ordinal_day=ds.mod_ordinal_day)
    std = xr.open_dataset(clim_std).sel(ordinal_day=ds.mod_ordinal_day)

    # dst_anomalies = ds - interpolate_grid(clim_mean,ds)
    dst_anomalies = ((ds - interpolate_grid(mean, ds)) /
                     interpolate_grid(std, ds)).astype('float32')

    # rename anomaly variables and add attributes
    datakeys = list(dst_anomalies.data_vars)
    dst_anomalies = dst_anomalies.rename_vars(
        dict(zip(datakeys, [var + '_anomaly' for var in datakeys])))
    dst_anomalies = dst_anomalies.swap_dims({'mod_ordinal_day': 'time'})
    dst_final = (xr.merge([dst, dst_anomalies]).
                 drop(['mod_ordinal_day', 'ordinal_day']))

    for v, var_long_name in zip(list(dst_anomalies.data_vars),
                                [dst[v].long_name + ' standardized anomaly' for v in datakeys]):
        dst_final[v].attrs['long_name'] = var_long_name
        dst_final[v].attrs['units'] = 'Numeric'
        dst_final[v].attrs['FMI_ens_note'] = (
            'This is the anomaly calculated using 40-year Era5 mean ',
            'value and standard deviation'
            )

    # dst_final = dst_final.where(dst_final > int16_minvalue, int16_minvalue)
    # dst_final = dst_final.where(dst_final < int16_maxvalue, int16_maxvalue)
    dst_final = utils.replace_minmax(dst_final, int16_minvalue, int16_maxvalue)

    return dst_final


def add_danger_classes(ds, var='fwi'):
    """
    5 or 7 classes are needed
    # fire_danger_classes = [1,2,3,4,5,6,7]
    # fire_danger_classes_labels = ['very low','low','moderate','high','very high','extreme','very extreme']

    EFFIS thresholds (https://effis.jrc.ec.europa.eu/about-effis/technical-background/fire-danger-forecast)
    """

    classes = {'fwi': [5.2, 11.2, 21.3, 38, 50, 70],
               'ffmc': [82.7, 86.1, 89.2, 93],
               'dmc': [15.7, 27.9, 53.1, 140.7],
               'dc': [256.1, 334.1, 450.6, 749.4],
               'isi': [3.2, 5, 7.5, 13.4],
               'bui': [24.2, 40.7, 73.3, 178.1]}

    ds[f'{var}_danger_class'] = xr.where((ds[var] < classes[var][0]), 1,
                                         ds[var], keep_attrs=True)
    ds[f'{var}_danger_class'] = xr.where((ds[var] >= classes[var][0]) &
                                         (ds[var] < classes[var][1]),
                                         2, ds[f'{var}_danger_class'])
    ds[f'{var}_danger_class'] = xr.where((ds[var] >= classes[var][1]) &
                                         (ds[var] < classes[var][2]), 3,
                                         ds[f'{var}_danger_class'])
    ds[f'{var}_danger_class'] = xr.where((ds[var] >= classes[var][2]) &
                                         (ds[var] < classes[var][3]), 4,
                                         ds[f'{var}_danger_class'])

    if len(classes[var]) > 4:
        ds[f'{var}_danger_class'] = xr.where((ds[var] >= classes[var][3]) &
                                             (ds[var] < classes[var][4]), 5,
                                             ds[f'{var}_danger_class'])
        ds[f'{var}_danger_class'] = xr.where((ds[var] >= classes[var][4]) &
                                             (ds[var] < classes[var][5]), 6,
                                             ds[f'{var}_danger_class'])
        ds[f'{var}_danger_class'] = xr.where((ds[var] >= classes[var][5]), 7,
                                             ds[f'{var}_danger_class'])
    else:
        ds[f'{var}_danger_class'] = xr.where((ds[var] >= classes[var][3]), 5,
                                             ds[f'{var}_danger_class'])
    return ds
