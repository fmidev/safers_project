#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Open nc file and save it with zlib compression
# uses savenc from safers_utils.py

# marko.laine@fmi.fi

import sys
import os
import argparse
import xarray as xr
from safers_utils import savenc


parser = argparse.ArgumentParser()
#parser.add_argument("--zlib", action='store_true')
parser.add_argument("--int16", action='store_true',
                    help='use int16 instead of zlib')

parser.add_argument('ncfile', metavar='ncfile', type=str, nargs=1,
                    help='ncfile to read')

parser.add_argument('zfile', metavar='zfile', type=str, nargs='?',
                    help='output file')

opts = parser.parse_args(sys.argv[1:])

args = sys.argv

ncfile = opts.ncfile[0]
zfile = opts.zfile
if zfile is None:
    if opts.int16:
        zfile = os.path.splitext(ncfile)[0] + '_int16.nc'
    else:
        zfile = os.path.splitext(ncfile)[0] + '_zip.nc'

ds = xr.open_dataset(ncfile)
print(f'Opened {ncfile}')
    
# discrete int16 encoding
if opts.int16:
    savenc(ds, zfile, zlib=False, discrete=True)
    print('int')
else:
    savenc(ds, zfile, zlib=True, discrete=False)
    print('zlib')
print(f'Saved {zfile}')
