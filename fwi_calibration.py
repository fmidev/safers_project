#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FWI calibration

Read FMI ens files for last 30 days and train a calibration model.
Save calibrated fwi forecasts.

Still under construction.

marko.laine@fmi.fi
"""

import sys
import os
import glob
import logging
import argparse
import datetime
import warnings

import numpy as np
import xarray as xr

#import safers_utils as utils
#from upload_EC import upload_nc

import calib

# %% command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--log", default='stderr', type=str)
parser.add_argument("--level", default='INFO', type=str)
parser.add_argument("--outdir", default='/data/tmp/safers_fwi', type=str)
parser.add_argument("--fwidir", default='/data/tmp/safers_fwi', type=str)
parser.add_argument("--metadir", default='/data/tmp/metadata', type=str)
parser.add_argument("--datadir", default='/data/safers/data', type=str)
parser.add_argument("--region", default=None, type=str)
parser.add_argument("--fwiin", type=str, required=True)
parser.add_argument("--fwiout", type=str, required=True)
parser.add_argument("--modelout", default=None, type=str)
parser.add_argument("--variable", default=None, type=str)
parser.add_argument("--ndays", default=30, type=int)
parser.add_argument("--ensdir", default='/data/safers/fwi_ens', type=str)
parser.add_argument("--upload", action='store_true')
opts = parser.parse_args(sys.argv[1:])

# %% logging
if opts.log == 'stderr':
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None))
else:
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None),
                        filename=opts.log)


# ignore FutureWarning from rpy2
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Capture warnings to log file (e.g. RuntimeWarning from xarray)
logging.captureWarnings(True)

# warnings.filterwarnings('ignore')
# np.seterr(all="ignore")
np.seterr(invalid='ignore', over='ignore', divide='ignore')

logging.info('Starting fwi calibration')

fwidir = opts.fwidir
ensdir = opts.ensdir
fwiin = opts.fwiin
fwiout = opts.fwiout
ndays = opts.ndays

# read ens files for last 30 days and train the calibration model

today = datetime.datetime.now()

files = sorted(glob.glob(os.path.join(ensdir, 'Fwi_*_ENS.nc')))
files = calib.valid_files(files, today, delta=-ndays)
if len(files) < ndays//2:
    logging.error('Too few files for calibration, %d', len(files))
    sys.exit(1)
logging.info('read %d files', len(files))

mask = calib.getmask(opts.region)

# which variable to calibrate
# only works for one now?
variables = ['fwi', 'dc']
#variables = ['dc']
variables = ['fwi']

# only 12 leadtimes in output
leadtimes = np.arange(1, 15)

# fc, obs = calib.load_tigge(files, load=True)

# calibrate the forecasts given in --fwiin argument
fwi = xr.open_dataset(fwiin)
fwi_calib = fwi.copy()


for variable in variables:
    model = calib.train_many(files, mask=mask,  # fc=fc, obs=obs,
                             variable=variable,
                             leadtimes=leadtimes,
                             meanvar=variable + '_mean',
                             sdvar=variable + '_std')

    if opts.modelout is not None:
        logging.info('saved model to %s', opts.modelout)
        logging.warning('not implemented yet')

    logging.info('training done for %s', variable)

    fwi_calib = fwi_calib.rename({variable: variable + '_mean'})
    fwi_calib = calib.calibrate_many(fwi_calib, model, mask=mask,
                                     leadtimes=leadtimes,
                                     meanvar=variable + '_mean',
                                     sdvar=variable + '_std')
    fwi_calib = fwi_calib.rename({variable + '_mean_cal': variable + '_cal',
                                  variable + '_mean': variable})

    logging.info('calibration done for %s', variable)

# save calibrated
# remove all other variables for now
data_vars = list(fwi_calib.data_vars)
data_vars = [s for s in data_vars if s.rfind('_cal') > 0]

fwi_calib = fwi_calib[data_vars]


# fwi.to_netcdf(ofile)
#utils.savenc(fwi_calib, file=fwiout, zlib=False, discrete=True)
fwi_calib.to_netcdf(fwiout)
logging.info('saved to %s', fwiout)
logging.info('End of fwi calibration')
