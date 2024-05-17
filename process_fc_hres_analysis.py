#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process HRES grib file, save it as netcdf and optionally send the file to datalake.

This processes the analysis part of "extra" HRES grib file

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

from safers_data_tables import safers_hres_domain, safers_hres_res
# from safers_data_tables import safers_data_id


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

variables = ['t2m', 'd2m', 'u10', 'v10', 'msl']

# use grib names here
maxvars = ['mn2t6', 'mx2t6', '10fg6', 'mn2t3', 'mx2t3', '10fg3']
avgvars = ['litota1', 'litota3', 'litota6']


minstep = 0

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
ncfile = os.path.join(outdir, os.path.splitext(f2)[0] + '_HRES_analysis' + '.nc')

upload_to_datalake = opts.upload

logging.info('Origin time: %s', origintime)

ds = opengrib(gribfile, type='HRES', variable=variables, # minstep=minstep,
              maxvars=maxvars, avgvars=avgvars, analysis=True)

# some edits, lon back to longitude, time as coordinate, instead of leadtime (which is timedelta)
#ds = ds.swap_dims({'leadtime': 'time'}).rename({'lon': 'longitude', 'lat': 'latitude'})
ds = ds.rename({'lon': 'longitude', 'lat': 'latitude'})
#ds = ds.drop_vars(['surface', 'realization'], errors='ignore')


if not np.all(np.isin('r2', list(ds.data_vars))):
    ds = utils.addrh(ds, units='K')

dataid = utils.safers_data_id(fctype)
fctime = pd.to_datetime(ds.time.values)

i = 1
while pd.isnull(fctime):
    i = i + 1
    fctime = pd.to_datetime(ds.time.values[i])


fctime2 = pd.to_datetime(ds.time.values)
i = -1
while pd.isnull(fctime2):
    i = i - 1
    fctime2 = pd.to_datetime(ds.time.values[i])


ncname = utils.datafilename(origintime, fctime,
                            varid=dataid, fctime2=fctime2,
                            fctype=fctype, variable='analysis',
                            filetype='nc')

# convert u10 and v10 to wind speed and direction
# ds = utils.uvtows(ds)
# do other conversions
ds = utils.safers_convert(ds)
# ds = utils.convertssr(ds)  # no ssr in HRES

# calculate tp24
# logging.debug('Calculate tp24')
# ds = utils.addtp24(ds, drop=False)
# ds = utils.addprevtp24(ds, opts.outdir, variable='tp24')

# logging.debug('Calculate tp24 done')


ds = ds.drop_vars(['surface', 'depth', 'realization',
                   'entireAtmosphere', 'quantile'],
                  errors='ignore')


ds.load()

time = ds.get('time')
forecast_reference_time = ds.get('forecast_reference_time')

ds = ds.expand_dims('time')
# print(ds)

# rotate

ds = utils.rotated_ll_to_regular(ds, realization='time',
                                 llmm=safers_hres_domain,
                                 res=safers_hres_res,
                                 lon='longitude', lat='latitude')


#ds = utils.rotated_ll_to_regular3(ds,
#                                  llmm=safers_hres_domain,
#                                  res=safers_hres_res,
#                                  lon='longitude', lat='latitude')

#ds = ds.assign_coords({'time': time,
#                       'forecast_reference_time': forecast_reference_time})


# apply new attributes
ds = safers_attrs(ds)

#ds.to_netcdf(ncfile)
utils.savenc(ds, ncfile, zlib=opts.zlib, discrete=opts.discrete)
logging.info('Wrote %s', ncfile)

variables = list(ds.data_vars.keys())

# dataid = utils.safers_data_id(fctype)
metadatafile = os.path.join(metadir, f'metadata_{fctype}_analysis_{origintime.strftime("%Y%m%d%H%M")}.json')

# fctimes = ds.time.values
fctimes = np.atleast_1d(np.array(time))

bb = np.r_[ds.longitude.min().values,
           ds.longitude.max().values,
           ds.latitude.max().values,
           ds.latitude.min().values]

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
