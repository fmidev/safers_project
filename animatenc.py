#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# animate data in netcdf file
#
# ./animatenc.py ncfile.nc --var t2m --out anim.gif


import sys
import logging
import argparse

import xarray as xr
from pandas import to_datetime, to_timedelta
from safers_plots import animate

parser = argparse.ArgumentParser()
parser.add_argument("--out", "-o", default=None, type=str)
parser.add_argument("--variable", "-v", default=None, type=str)
parser.add_argument("ncfile", metavar="ncfile", type=str, nargs=1)
parser.add_argument("--vmin", default=None, type=float)
parser.add_argument("--vmax", default=None, type=float)
parser.add_argument("--cmap", "-c", default=None, type=str)
parser.add_argument("--int", default=300, type=float)
parser.add_argument("--dpi", default=100, type=float)
parser.add_argument("--label", default=None, type=str)
opts = parser.parse_args(sys.argv[1:])

variable = opts.variable
vmin = opts.vmin
vmax = opts.vmax
cmap = opts.cmap
nc = opts.ncfile[0]
gfile = opts.out

ds = xr.open_dataset(nc)

if variable is None:
    variable = list(ds.data_vars)[0]
da = ds[variable]

rt = da.coords.get("forecast_reference_time")
rt = '' if rt is None else to_datetime(rt.values)
lt = da.coords.get("leadtime")
lt = '' if lt is None else to_timedelta(lt.values)

title = f'From {rt} to {lt}'
anim = animate(da, cmap=cmap, vmin=vmin, vmax=vmax,
               interval=opts.int, dpi=opts.dpi,
               show=False, label=opts.label)
if gfile is None:
    gfile = 'anim.gif'
anim.save(gfile)
print(f'Saved {gfile}')
