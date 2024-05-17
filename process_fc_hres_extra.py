#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process HRES grib file, save it as netcdf and optionally send the file to datalake.

This processes the "extra" HRES grib files (not analysis, stepRange=0).
The data set will have litota, mn2, mx2.

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

variables = ['litota1', 'mn2t3', 'mx2t3', 'mn2t6', 'mx2t6',
             'litota3', 'litota6']

variables = ['litota1', 'mn2t3', 'mx2t3']


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
ncfile = os.path.join(outdir, os.path.splitext(f2)[0] + '_HRES_extra' + '.nc')

upload_to_datalake = opts.upload

logging.info(f'Origin time: {origintime}')

ds = opengrib(gribfile, type='HRES', variable=variables, minstep=minstep,
    maxvars=maxvars, avgvars=avgvars)
# some edits, lon back to longitude, time as coordinate, instead of leadtime (which is timedelta)
ds = ds.swap_dims({'leadtime': 'time'}).rename({'lon': 'longitude', 'lat': 'latitude'})
ds = ds.drop_vars(['surface', 'realization'], errors='ignore')

# rename r to r2
if np.all(np.isin('r', list(ds.data_vars))):
    ds = ds.rename({'r': 'r2'})
    ds['r2'].attrs['long_name'] = '2 metre relative humidity'


dataid = utils.safers_data_id(fctype)
fctime = pd.to_datetime(ds.time.values[0])

i = 1
while pd.isnull(fctime):
    i = i + 1
    fctime = pd.to_datetime(ds.time.values[i])


fctime2 = pd.to_datetime(ds.time.values[-1])
i = -1
while pd.isnull(fctime2):
    i = i - 1
    fctime2 = pd.to_datetime(ds.time.values[i])


ncname = utils.datafilename(origintime, fctime,
                            varid=dataid, fctime2=fctime2,
                            fctype=fctype, variable='extra',
                            filetype='nc')

# convert u10 and v10 to wind speed and direction
#ds = utils.uvtows(ds)
# do other conversions
ds = utils.safers_convert(ds)
# ds = utils.convertssr(ds)  # no ssr in HRES

# calculate tp24
#logging.debug('Calculate tp24')
#ds = utils.addtp24(ds, drop=False)
#ds = utils.addprevtp24(ds, opts.outdir, variable='tp24')
#logging.debug('Calculate tp24 done')


ds = ds.drop_vars(['surface', 'depth', 'realization',
                   'entireAtmosphere', 'quantile'],
                  errors='ignore')

# rotate

ds.load()
#print(ds['mn2'])

ds = utils.rotated_ll_to_regular(ds, realization='time',
                                 llmm=safers_hres_domain, res=safers_hres_res,
                                 lon='longitude', lat='latitude')


# print(ds)

# this is not needed as we already have data in every hour??
datavars = list(ds.data_vars)
cvars = [
     ['litotax', 'litota1', 'litota3'],
     ['litota', 'litotax', 'litota6'],
     ['mn2', 'mn2t3', 'mn2t6'],
     ['mx2', 'mx2t3', 'mx2t6'],
     ['fg10', 'fg310', 'p10fg6'],
]
for ivar in range(len(cvars)):
    datavars = list(ds.data_vars)
    v = cvars[ivar]
    if (v[1] in datavars) and (v[2] in datavars):
        ds[v[0]] = ds[v[1]].combine_first(ds[v[2]])

dvars = [cvars[i][j] + k
         for k in ['', '_p10', '_p90']
         for j in [1, 2]
         for i in range(len(cvars))]
# ds = ds.drop(dvars, errors='ignore')
# ds = ds.drop(['litotax'], errors='ignore')

# some renaming here
ds = ds.rename({'litota1': 'litota', 'mn2t3': 'mn2', 'mx2t3': 'mx2'})

# print(ds)
# print(ds['mn2'])

# apply new attributes
ds = safers_attrs(ds)

# ds.to_netcdf(ncfile)
utils.savenc(ds, ncfile, zlib=opts.zlib, discrete=opts.discrete)
logging.info(f'Wrote {ncfile}')

variables = list(ds.data_vars.keys())

# dataid = utils.safers_data_id(fctype)
metadatafile = os.path.join(metadir, f'metadata_{fctype}_extra_{origintime.strftime("%Y%m%d%H%M")}.json')

fctimes = ds.time.values
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
logging.info(f'Wrote {metadatafile}')

if upload_to_datalake:
    logging.info('Upload to SAFRES datalake')
    upload_nc(metadatafile, ncfile)
    logging.info('Upload to SAFRES datalake DONE')

logging.info(f'Done all for {fctype} {origintime}')
