#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process HRES grib file, save it as netcdf and optionally send the file to datalake.

Example usage:
./process_fc_hres.py --grib /data/tmp/fc.grib --upload

marko.laine@fmi.fi
"""


import sys
import os
import logging
import argparse
# from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import safers_utils as utils
from safers_grib import opengrib
from safers_s3 import parsefile_HRES
from upload_EC import upload_nc
from safers_data_tables import safers_attrs

parser = argparse.ArgumentParser()
parser.add_argument("--log", default='stderr', type=str)
parser.add_argument("--level", default='INFO', type=str)
parser.add_argument("--outdir", default='/data/tmp/safers_nc/', type=str)
parser.add_argument("--metadir", default='/data/tmp/metadata/', type=str)
parser.add_argument("--upload", action='store_true')
parser.add_argument("--zlib", action='store_true')
parser.add_argument("--discrete", action='store_true')
parser.add_argument("--grib", type=str)
opts = parser.parse_args(sys.argv[1:])

if opts.log == 'stderr':
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None))
else:
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None),
                        filename=opts.log)

# Capture warnings to log file (e.g. RuntimeWarning from xarray)
logging.captureWarnings(True)

fctype = 'HRES'

variables = ['t2m', 'r', 'd2m', 'u10', 'v10', 'tp', 'cape']
minstep = 1

# tmpdir = opts.tmpdir
outdir = opts.outdir
metadir = opts.metadir
thisdir = os.path.dirname(os.path.realpath(__file__))

# generate output directory
os.makedirs(outdir, exist_ok=True)
os.makedirs(metadir, exist_ok=True)

metadatatemplate = os.path.join(thisdir, 'metadata_template_ECMWF.json')

gribfile = opts.grib
f1, f2, origintime = parsefile_HRES(gribfile)
ncfile = os.path.join(outdir, os.path.splitext(f2)[0] + '_HRES' + '.nc')

upload_to_datalake = opts.upload

logging.info('Origin time: %s', origintime)

ds = opengrib(gribfile, type='HRES', variable=variables, minstep=minstep)
bb = np.r_[ds.lon.min().values, ds.lon.max().values, ds.lat.max().values, ds.lat.min().values]
# some edits, lon back to longitude, time as coordinate, instead of leadtime (which is timedelta)
ds = ds.swap_dims({'leadtime': 'time'}).rename({'lon': 'longitude', 'lat': 'latitude'})
ds = ds.drop_vars(['surface', 'realization'], errors='ignore')

# rename r to r2
if np.all(np.isin('r', list(ds.data_vars))):
    ds = ds.rename({'r': 'r2'})
    ds['r2'].attrs['long_name'] = '2 metre relative humidity'

# if r2 is missing, generate it
if not np.all(np.isin('r2', list(ds.data_vars))):
    ds = utils.addrh(ds, units='K')
logging.debug('Added r2')

dataid = utils.safers_data_id(fctype)
fctime = pd.to_datetime(ds.time.values[0])
fctime2 = pd.to_datetime(ds.time.values[-1])
ncname = utils.datafilename(origintime, fctime,
                            varid=dataid, fctime2=fctime2,
                            fctype=fctype, variable='several',
                            filetype='nc')

# convert u10 and v10 to wind speed and direction
#ds = utils.uvtows(ds)
# do other conversions
ds = utils.safers_convert(ds)
# ds = utils.convertssr(ds)  # no ssr in HRES
# Calculate wind speed and direction
ds = utils.uvtows(ds, dwi=True, drop=False)

# calculate tp24
logging.debug('Calculate tp24')
ds = utils.addtp24(ds, drop=False)
ds = utils.addprevtp24(ds, opts.outdir, variable='tp24')

logging.debug('Calculate tp24 done')

# apply new attributes
ds = safers_attrs(ds)

ds = ds.drop_vars(['surface', 'depth', 'realization',
                   'entireAtmosphere', 'quantile'],
                  errors='ignore')

#ds.to_netcdf(ncfile)
utils.savenc(ds, ncfile, zlib=opts.zlib, discrete=opts.discrete)

logging.info('Wrote %s', ncfile)

variables = list(ds.data_vars.keys())

# dataid = utils.safers_data_id(fctype)
metadatafile = os.path.join(metadir, f'metadata_{fctype}_{origintime.strftime("%Y%m%d%H%M")}.json')

fctimes = ds.time.values

utils.make_metadata_nc(origintime, variables, fctimes,
                       fctype=fctype,
                       template=metadatatemplate,
                       bb=bb,
                       name=ncname,
                       file=metadatafile)
logging.info('Wrote %s', metadatafile)

if upload_to_datalake:
    logging.info('Upload to SAFRES datalake')
    upload_nc(metadatafile, ncfile)
    logging.info('Upload to SAFRES datalake DONE')

logging.info('Done all for %s %s', fctype, origintime)
