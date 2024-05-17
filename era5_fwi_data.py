#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# process era5 data from voima
# collect 2t, 2d, 10u, 10v, and calculate tp24 from tp

import sys
import numpy as np
import xarray as xr
import dask

args = sys.argv
if len(args) < 2:
    print('usage:', args[0], 'grib file')
    sys.exit(1)

gribfile = args[1]

if len(args) > 2:
    ncfile = args[2]
else:
    ncfile = '/data/tmp/era5/test2.nc'


def calctp24(ds, month=None):
    """tp24 from tp."""
    tp = (ds['tp'].stack(z=['time', 'step']).
          swap_dims({'z': 'valid_time'}).
          cumsum(dim='valid_time').
          drop_vars(['z']).
          rename({'valid_time': 'time'}).
          transpose('time', 'latitude', 'longitude')
          )

    time1 = tp.time[tp.time >= tp.time[0] + np.timedelta64(24, 'h')]
    time2 = time1 - np.timedelta64(24, 'h')
    tp24 = xr.full_like(tp, fill_value=np.nan)
    tp24.loc[dict(time=time1)] = (tp.sel(time=time1).values -
                                  tp.sel(time=time2).values)

    tp24 = tp24 * 1000  # to mm
    # tp24 = tp24.transpose('time', 'latitude', 'longitude')
    tp24.attrs['long_name'] = 'total precipitation in the last 24 hours'
    tp24.attrs['units'] = 'mm'

    if month is None:
        month = tp.time.dt.month[48].values

    tp24 = tp24.where(tp24.time.dt.month == month, drop=True)

    return tp24


def calctp(ds, month=None):
    """tp from grib."""
    tp = (ds['tp'].stack(z=['time', 'step']).
          swap_dims({'z': 'valid_time'}).
          cumsum(dim='valid_time').
          drop_vars(['z']).
          rename({'valid_time': 'time'}).
          transpose('time', 'latitude', 'longitude')
          )

    if month is None:
        month = tp.time.dt.month[48].values
    # drop values outside month
    tp = tp.where(tp.time.dt.month == month, drop=True)

    return tp


def crop(ds, bb=[-25.0 + 360, 40.0, 25.5, 72.0]):
    """Crop dataset."""
    ds = ds.where(((ds.longitude >= bb[0]) |
                   (ds.longitude <= bb[1])) &
                  (ds.latitude >= bb[2]) &
                  (ds.latitude <= bb[3]), drop=True)
    ds.coords['longitude'] = (ds.longitude + 180) % 360 - 180
    ds = ds.sortby(ds.longitude)
    return ds


def readgrib(f, var, noindex=False, ntime=1):
    """Read one variable from file."""
    gribargs = {'filter_by_keys': {'shortName': var}}
    if noindex:
        gribargs.update({'indexpath': ''})
    ds = xr.open_dataset(f,
                         backend_kwargs=gribargs,
                         engine='cfgrib',
                         chunks={'time': ntime})
    return ds


# variables = ['t2m', 'd2m', 'tp', 'u10', 'v10']
# variables = ['t2m', 'd2m', 'u10', 'v10']
variables2 = ['2t', '2d', '10u', '10v']

ds = []
for v in variables2:
    ds.append(readgrib(gribfile, v))
ds = xr.merge(ds, compat='override')
ds = crop(ds)

ds2 = readgrib(gribfile, 'tp')
ds2 = crop(ds2)
ds2.load()
tp24 = calctp24(ds2)

ds['tp24'] = tp24
ds['tp'] = calctp(ds2)

ds = ds.drop_vars(['surface', 'realization', 'number',
                   'forecast_reference_time', 'step', 'valid_time'],
                  errors='ignore')

i16encoding = {'dtype': 'int16',
               'scale_factor': 0.01,
               'add_offset': 0.0,
               '_FillValue': -9999}
data_vars = list(ds.data_vars)
data_vars.remove('tp')
encoding = {key: i16encoding for key in data_vars}
#            if key.partition('_')[0] in int16vars}


ds.to_netcdf(ncfile, encoding=encoding)

# print(ds)
