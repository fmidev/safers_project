#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# plot data in netcdf file

# arguments ncfile variable time [gfile]

import sys
import logging
import argparse

import xarray as xr
from pandas import to_datetime, to_timedelta
from safers_plots import plotmap

parser = argparse.ArgumentParser()
parser.add_argument("--out", default=None, type=str)
# parser.add_argument("--variable", type=str, required=True)
parser.add_argument("--variable", "-v", default=None, type=str)
parser.add_argument("ncfile", metavar="ncfile", type=str, nargs=1)
parser.add_argument("--vmin", default=None, type=float)
parser.add_argument("--vmax", default=None, type=float)
parser.add_argument("--time", default=0, type=int)
parser.add_argument("--notime", action='store_true')
parser.add_argument("--wind", action='store_true')
parser.add_argument("--cmap", "-c", default=None, type=str)
opts = parser.parse_args(sys.argv[1:])

variable = opts.variable
time = opts.time
vmin = opts.vmin
vmax = opts.vmax
cmap = opts.cmap
nc = opts.ncfile[0]
gfile = opts.out

ds = xr.open_dataset(nc)

if variable is None:
    variable = list(ds.data_vars)[0]

if opts.notime:
    da = ds[variable]
    tt = variable
else:
    da = ds[variable].isel(time=time)
    tt = to_datetime(ds.time.isel(time=time).values)

if opts.wind:
    u = ds['u10'].isel(time=time)
    v = ds['v10'].isel(time=time)
else:
    u = None
    v = None

rt = da.coords.get("forecast_reference_time")
lt = da.coords.get("leadtime")

rt = '' if rt is None else to_datetime(rt.values)
if lt is None:
    lt = ''
else:
    lt = to_timedelta(lt.values)
    if rt != '':
        lt = lt + rt

title = f'From {rt} to {lt}'
if rt == '' and lt == '':
    title = f'{tt}'
plotmap(da, title=title, file=gfile, cmap=cmap, vmin=vmin, vmax=vmax,
        u=u, v=v)
