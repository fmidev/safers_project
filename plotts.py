#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot time series at one point data in netcdf file

example:
./plotts.py /data/safers/data/FWI_FFMC_DMC_DC_ERA5_1980_2019_mean.nc --var fwi --lon 25 --lat 63 --out /tmp/o.png

"""


import sys
# import logging
import argparse

# import numpy as np
import xarray as xr
# import matplotlib.pyplot as plt
# from pandas import to_datetime, to_timedelta
from safers_plots import plotts, plot_fwi_ts, plot_fwi_anomaly

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=None, type=str)
parser.add_argument("--variable", "-v", default=None, type=str)
parser.add_argument("ncfile", metavar="ncfile", type=str, nargs=1)
parser.add_argument("--lon", type=float)
parser.add_argument("--lat", type=float)
parser.add_argument("--fwi", action='store_true')
parser.add_argument("--fwianomaly", action='store_true')
opts = parser.parse_args(sys.argv[1:])

variable = opts.variable
lon = opts.lon
lat = opts.lat
nc = opts.ncfile[0]
gfile = opts.out

ds = xr.open_dataset(nc)

if opts.fwi:
    plot_fwi_ts(ds, lon=lon, lat=lat, file=gfile)
    sys.exit()

if opts.fwianomaly:
    plot_fwi_anomaly(ds, lon=lon, lat=lat, file=gfile)
    sys.exit()

# rt = da.coords.get("forecast_reference_time")
# rt = '' if rt is None else to_datetime(rt.values)
# lt = da.coords.get("leadtime")
# lt = '' if lt is None else to_timedelta(lt.values)

# title = f'From {rt} to {lt}'
plotts(ds, variable=variable, lon=lon, lat=lat, file=gfile)
