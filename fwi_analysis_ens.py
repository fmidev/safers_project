#!/usr/bin/env python3

"""
Extract first time from ENS FWI nc file and save it
Save only ENS mean
"""

import sys
import os
import logging
import argparse

import xarray as xr

from safers_utils import savenc

parser = argparse.ArgumentParser()
parser.add_argument("--log", default='stderr', type=str)
parser.add_argument("--level", default='INFO', type=str)
parser.add_argument("--out", "-o", default=None, type=str)
parser.add_argument("--infile", "-i", default=None, type=str)
opts = parser.parse_args(sys.argv[1:])

# %% logging
if opts.log == 'stderr':
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None))
else:
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None),
                        filename=opts.log)

# Capture warnings to log file (e.g. RuntimeWarning from xarray)
logging.captureWarnings(True)

data_vars = ['fwi', 'isi', 'bui', 'dc', 'dmc', 'ffmc']

ds = xr.open_dataset(opts.infile).isel(time=[0])[data_vars]
logging.debug('opened %s', opts.infile)

savenc(ds, file=opts.out, discrete=True)
# ds.to_netcdf(opts.out)

logging.info('Generated analysis file in %s', opts.out)
