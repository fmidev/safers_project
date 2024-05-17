#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process ENS or EXT grib file, save it as netcdf and optionally send the file to datalake.

usage:
./process_fc_ens.py --fctype ENS --grib /data/tmp/fc.grib --upload


marko.laine@fmi.fi
"""

import sys
import os
import logging
import argparse
# from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import dask
import pandas as pd

import safers_utils as utils
import safers_grib
from safers_s3 import parsefile_ENS

from safers_data_tables import safers_ext_domain, safers_ext_res
from safers_data_tables import safers_ens_domain, safers_ens_res
from safers_data_tables import safers_data_id
from safers_data_tables import safers_attrs

from upload_EC import upload_nc

parser = argparse.ArgumentParser()
parser.add_argument("--log", default='stderr', type=str)
parser.add_argument("--level", default='INFO', type=str)
parser.add_argument("--outdir", default='/data/tmp/safers_nc/', type=str)
parser.add_argument("--metadir", default='/data/tmp/metadata/', type=str)
parser.add_argument("--fctype", default='ENS', type=str)
parser.add_argument("--dropuv", action='store_true')
parser.add_argument("--upload", action='store_true')
parser.add_argument("--zlib", action='store_true')
parser.add_argument("--discrete", action='store_true')
parser.add_argument("--grib", type=str)
parser.add_argument("--variables", default='all', type=str)
parser.add_argument("--minstep", default=3, type=int)
opts = parser.parse_args(sys.argv[1:])

if opts.log == 'stderr':
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None))
else:
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None),
                        filename=opts.log)

# Capture warnings to log file (e.g. RuntimeWarning from xarray)
logging.captureWarnings(True)

dask.config.set(**{'array.slicing.split_large_chunks': True})

fctype = opts.fctype
variable_type = opts.variables

if fctype == 'ENS':
    safers_res = safers_ens_res
    safers_domain = safers_ens_domain

    if variable_type == 'all':
        variables = ['t2m', 'd2m', 'u10', 'v10', 'tp',
                 'mn2t3', 'mx2t3', 'mn2t6', 'mx2t6',
                 'fg310', 'p10fg6', 'msl',
                 'ssr', 'swvl1', 'swvl2',
                 'cape', 'litota3', 'litota6'
                 ]
    elif variable_type == 'basic':
        variables = ['t2m', 'd2m', 'u10', 'v10', 'tp']
    elif variable_type == 'extra':
        # do not put 'mn2t3' (etc) first
        variables = [
            'msl', 'ssr', 'swvl1', 'swvl2',
            # 'cape', 'litota3', 'litota6',
            'mn2t3', 'mx2t3', 'mn2t6', 'mx2t6',
            'fg310', 'p10fg6',
        ]
    elif variable_type == 'lightning':
        # need to have some variable with all time steps here
        variables = [
            'cape', 'litota3', 'litota6',
        ]
    else:
        logging.error('Unknown type %s %s', fctype, variable_type)
        sys.exit(1)

elif fctype == 'EXT':
    safers_res = safers_ext_res
    safers_domain = safers_ext_domain

    if variable_type == 'all':
        variables = ['t2m', 'd2m', 'u10', 'v10', 'tp',
                     'mn2t6', 'mx2t6', 'p10fg6', 'msl',
                     'ssr', 'swvl1', 'swvl2',
                     'cape', 'litota6',
                     ]
    elif variable_type == 'basic':
        variables = [
            't2m', 'd2m', 'u10', 'v10', 'tp',
        ]
    elif variable_type == 'extra':
        variables = [
            'msl', 'ssr', 'swvl1', 'swvl2',
            # 'cape', 'litota6',
            'mn2t6', 'mx2t6', 'p10fg6',
        ]
    elif variable_type == 'lightning':
        variables = [
            'cape', 'litota6',
        ]
    else:
        logging.error('Unknown type %s %s', fctype, variable_type)
        sys.exit(1)

else:
    logging.error('Unknown fctype %s', fctype)
    sys.exit(1)

dataid = safers_data_id(fctype, variable_type)

# rename these variables before merging (not used)
# But used to rename at the end
rename_vars = {'litota1': 'litota',
               'litota3': 'litota',
               'litota6': 'litota',
               'p10fg6': 'fg10',
               'fg310': 'fg10',
               'mn2t3': 'mn2',
               'mn2t6': 'mn2',
               'mx2t3': 'mx2',
               'mx2t6': 'mx2',
               }

# rename = {}

# use grib names here
maxvars = ['mn2t6', 'mx2t6', '10fg6', 'mn2t3', 'mx2t3', '10fg3']
avgvars = ['litota3', 'litota6']

# test
# variables = ['t2m', 'tp', 'ssr']
# variables = ['t2m']

# funs = ['mean', 'std']
funs = ['mean', 'p10', 'p90']

outdir = opts.outdir
metadir = opts.metadir
thisdir = os.path.dirname(os.path.realpath(__file__))

# generate output directory
os.makedirs(outdir, exist_ok=True)
os.makedirs(metadir, exist_ok=True)


metadatatemplate = os.path.join(thisdir, 'metadata_template_ECMWF.json')

gribfile = opts.grib
f1, f2, origintime = parsefile_ENS(gribfile)

minstep = opts.minstep

# FIXME, see later (ncfile and ncname will be different)
# extra = '_extra' if variable_type == 'extra' else ''
extra = '' if variable_type == 'all' or variable_type == 'basic' else f'_{variable_type}'
ncfile = os.path.join(outdir, os.path.splitext(f2)[0] + '_' + fctype + extra + '.nc')

upload_to_datalake = opts.upload

logging.info('Origin time: %s', origintime)


logging.debug('Opening grib')
ds = safers_grib.opengrib(gribfile, type=fctype, variable=variables, minstep=minstep,
                          maxvars=maxvars, avgvars=avgvars,
                          chunks={'step': 4})

logging.debug('Opening grib done')

# some edits, lon back to longitude, time as coordinate, instead of leadtime (which is timedelta)
ds = ds.swap_dims({'leadtime': 'time'}).rename({'lon': 'longitude', 'lat': 'latitude'})

# should be done for the ensemble (which is still in K)
# calculate relative humidity
if not np.all(np.isin('r2', list(ds.data_vars))):
    ds = utils.addrh(ds, units='K')
logging.debug('Added r2')

# Add wind speed for every member
# Wind direction is added later after mean calculation
if variable_type in ('basic', 'all'):
    logging.debug('Calculate wind speed')
    ds = utils.uvtows(ds, dwi=False, drop=False)
    logging.debug('Calculate wind speed done')

logging.debug('Calculate tp24')
ds = utils.addtp24(ds, drop=False)
logging.debug('Calculate tp24 done')

logging.debug('Starting combining')
ds2 = []
for fun in funs:
    rename = not fun == 'mean'
    ds2 += [utils.ensmean(ds, fun=fun, rename=rename)]

ds = xr.merge(ds2, compat='override')
# clean up
del ds2

ds = ds.drop_vars(['surface', 'depth', 'realization',
                   'entireAtmosphere', 'quantile'],
                  errors='ignore')

logging.debug('done')
logging.debug('load into memory and do computations')
ds.load()
logging.debug('load into memory done')

logging.debug('start rotating')
# grid conversion from rotated to regular
ds = utils.rotated_ll_to_regular(ds, realization='time',
                                 llmm=safers_domain, res=safers_res,
                                 lon='longitude', lat='latitude')
logging.debug('done rotating')

if fctype == 'ENS':
    logging.debug('combine variables')
    datavars = list(ds.data_vars)
    cvars = [
        ['litota', 'litota3', 'litota6'],
        ['mn2', 'mn2t3', 'mn2t6'],
        ['mx2', 'mx2t3', 'mx2t6'],
        ['fg10', 'fg310', 'p10fg6'],
    ]
#    for ivar in range(len(cvars)):
#        v = cvars[ivar]
    for ivar, v in enumerate(cvars):
        if (v[1] in datavars) and (v[2] in datavars):
            ds[v[0]] = ds[v[1]].combine_first(ds[v[2]])
            ds[v[0] + '_p10'] = ds[v[1] + '_p10'].combine_first(ds[v[2] + '_p10'])
            ds[v[0] + '_p90'] = ds[v[1] + '_p90'].combine_first(ds[v[2] + '_p90'])

    dvars = [cvars[i][j] + k
        for k in ['', '_p10', '_p90']
        for j in [1, 2]
        for i in range(len(cvars))]
    ds = ds.drop(dvars, errors='ignore')
    logging.debug('combine variables done ')


bb = np.r_[ds['longitude'].min().values, ds['longitude'].max().values,
           ds['latitude'].max().values, ds['latitude'].min().values]


logging.debug('start converting')

# do other conversions
ds = utils.safers_convert(ds)
ds = utils.convertssr(ds)
ds = utils.convertssr(ds, ssrvar='ssr_p10')
ds = utils.convertssr(ds, ssrvar='ssr_p90')

# Add wind direction only for the mean
# Optionally drop u and v here
if variable_type in ('basic', 'all'):
    logging.debug('Calculate wind direction')
    ds = utils.uvtows(ds, dwi=True, ws=False, drop=False)
    if opts.dropuv:
        ds = ds.drop_vars(['u10', 'v10', 'u10_p10', 'v10_p10', 'u10_p90', 'v10_p90'], errors='ignore')
    logging.debug('Calculate wind direction done')

logging.debug('load tp24 for the first 24h')
ds = utils.addprevtp24(ds, opts.outdir, variable='tp24')
# ds = utils.addprevtp24(ds, opts.outdir, variable='tp24_std')
ds = utils.addprevtp24(ds, opts.outdir, variable='tp24_p10')
ds = utils.addprevtp24(ds, opts.outdir, variable='tp24_p90')

# do some renaming of the variables here
# see rename_vars at the beginning
if fctype == 'EXT':
    for variable in ds.data_vars:
        basevar = variable.partition('_')[0]
        fun = variable.partition('_')[2]
        newname = rename_vars.get(basevar)
        if newname is not None:
            if fun == '':
                ds = ds.rename({variable: newname})
            else:
                ds = ds.rename({variable: newname + '_' + fun})

# apply new attributes
ds = safers_attrs(ds)

logging.debug('done converting')

# dataid = utils.safers_data_id(fctype)
fctime = pd.to_datetime(ds.time.values[0])
fctime2 = pd.to_datetime(ds.time.values[-1])
i = -1
while pd.isnull(fctime2):
    i = i - 1
    fctime2 = pd.to_datetime(ds.time.values[i])
extra = 'several' if variable_type == 'all' else variable_type
ncname = utils.datafilename(origintime, fctime,
                            varid=dataid, fctime2=fctime2,
                            fctype=fctype, variable=extra,
                            filetype='nc')


logging.debug('start saving')
# ds.to_netcdf(ncfile)
utils.savenc(ds, ncfile, zlib=opts.zlib, discrete=opts.discrete)
logging.info('Wrote %s', ncfile)

variables = list(ds.data_vars.keys())

# dataid = utils.safers_data_id(fctype)
metadatafile = os.path.join(metadir, f'metadata_{fctype}_{variable_type}_{origintime.strftime("%Y%m%d%H%M")}.json')

fctimes = ds.time.values

utils.make_metadata_nc(origintime, variables, fctimes,
                       fctype=fctype,
                       template=metadatatemplate,
                       variable_type=variable_type,
                       dataid=dataid,
                       bb=bb,
                       funs=funs,
                       name=ncname,
                       file=metadatafile)
logging.info('Wrote %s', metadatafile)

if upload_to_datalake:
    logging.info('Upload to SAFRES datalake')
    upload_nc(metadatafile, ncfile)
    logging.info('Upload to SAFRES datalake DONE')

logging.info('Done all for %s %s', fctype, origintime)
