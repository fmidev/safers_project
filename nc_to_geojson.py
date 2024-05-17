#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Process netcdf file to several geojson files

# marko.laine@fmi.fi

import sys
import os
import logging
import argparse
# from datetime import datetime, timedelta

import multiprocessing
from joblib import Parallel, delayed

import numpy as np
import xarray as xr
import pandas as pd

import safers_utils as utils
from safers_geojson import da_to_geojson

from upload_EC import upload_files

parser = argparse.ArgumentParser()
parser.add_argument("--nc", type=str)
parser.add_argument("--log", default='stderr', type=str)
parser.add_argument("--fctype", default='HRES', type=str)
parser.add_argument("--metadir", default='/data/tmp/metadata/', type=str)
parser.add_argument("--jsondir", default='/data/tmp/safers_geojson/', type=str)
parser.add_argument("--level", default='INFO', type=str)
parser.add_argument("--ncores", default=-1, type=int)
parser.add_argument("--upload", action='store_true')
parser.add_argument("--convert", action='store_true')
opts = parser.parse_args(sys.argv[1:])

if opts.log == 'stderr':
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None))
else:
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None),
                        filename=opts.log)


# Capture warnings to log file (e.g. RuntimeWarning from xarray)
logging.captureWarnings(True)


# this does not work
def start_logger(lfile='stderr', level='INFO'):
    """."""
    logger = logging.getLogger("mylogger")
    if len(logger.handlers) == 0:
        if lfile == 'stderr':
            logger.basicConfig(level=getattr(logging, level.upper(), None))
        else:
            logger.basicConfig(level=getattr(logging, level.upper(), None),
                               filename=lfile)
    # Capture warnings to log file (e.g. RuntimeWarning from xarray)
    logger.captureWarnings(True)
    return logger


ncfile = opts.nc
ds = xr.open_dataset(ncfile)

origintime = pd.to_datetime(ds['forecast_reference_time'].values)

variables = list(ds.data_vars)
fctimes = ds.time.values
leads = ds.leadtime.values

ncores = opts.ncores
if ncores == 0:
    ncores = multiprocessing.cpu_count()

fctype = opts.fctype

# bb = utils.ds_bb(ds)
if fctype == 'HRES':
    bb = utils.safers_hres_domain
else:
    bb = utils.safers_ens_domain

# tmpdir =
outdir = opts.jsondir
metadir = opts.metadir
thisdir = os.path.dirname(os.path.realpath(__file__))

metadatatemplate = os.path.join(thisdir, 'metadata_template_ECMWF.json')

upload_to_datalake = opts.upload

logging.info('Origin time: %s', origintime)

# end times for each forecast
fctimes2 = np.append(fctimes[1:], fctimes[-1] + (fctimes[-1] - fctimes[-2]))

# generate output directory
# os.makedirs(tmpdir, exist_ok=True)
os.makedirs(outdir, exist_ok=True)
os.makedirs(metadir, exist_ok=True)

# convert u10 and v10 to wind speed and direction
#ds = utils.uvtows(ds)
#variables = list(ds.data_vars)

# pool = multiprocessing.Pool(processes=ncores)


# code to process one leadtime for one variable
# uses some global variables
def process_fc(da, origintime, variable, dataid, fctimes,
               interval, geojsonfiles, k):
    """Code to process one leadtime for one variable."""
    # need to redefine logging for each sub-process
    if opts.log == 'stderr':
        logging.basicConfig(level=getattr(logging, opts.level.upper(), None))
    else:
        logging.basicConfig(level=getattr(logging, opts.level.upper(), None),
                            filename=opts.log)
    fctime = pd.to_datetime(fctimes[k])
    if np.sum(np.isfinite(da.isel(time=k))).values < 10:
        logging.warning('%s in %s is mostly NaN', fctime, variable)
        return
    da_to_geojson(da.isel(time=k), os.path.join(outdir, geojsonfiles[k]),
                  interval=interval,
                  x='longitude', y='latitude')
    logging.info('wrote %s', os.path.join(outdir, geojsonfiles[k]))


# could make global shared memory in file
# folder = tempfile.mkdtemp()
# path = os.path.join(folder, 'shared.mmap')
# shareddata = np.memmap(path, dtype=float, shape=(data.shape), mode='w+')
# delete shared memory file
# shutil.rmtree(folder)

# loop over all variables
# for variable in variables[1:2]:  # !!!TEST!!!
for variable in variables:
    da = ds[variable]
    if opts.convert:
        da = utils.safers_conversion(da, variable)
    cfname = utils.ecmwftocf(variable)
    dataid = utils.safers_data_id(variable)
    metadatafile = os.path.join(metadir, f'metadata_{fctype}_{cfname}_{origintime.strftime("%Y%m%d%H%M")}.json')
    interval = utils.safers_isobar_interval(variable)
    logging.info('Processing %s', variable)
    nlead = len(fctimes)
#    geojsonfiles = [None] * nlead
    geojsonfiles = []
    # generate all output file names
    for k in range(nlead):
        fctime = pd.to_datetime(fctimes[k])
        fctime2 = pd.to_datetime(fctimes2[k])
        geojsonfiles.append(utils.datafilename(origintime, fctime, dataid,
                                               fctime2=fctime2,
                                               fctype=fctype,
                                               filetype='geojson',
                                               variable=variable))
    # remove Nones from list
    logging.info('Started parallel job')
    Parallel(n_jobs=ncores)(
        delayed(process_fc)(da, origintime, variable,
                            dataid, fctimes, interval,
                            geojsonfiles, k)
        for k in range(nlead))
    logging.info('Ended parallel job')
    # geojsonfiles = [x for x in geojsonfiles if x is not None]
    # should check here if files exists and fix fctimes, too
    nfiles = len(geojsonfiles)
    fileinds = np.arange(nfiles)
    exinds = [x for x in fileinds if
              os.path.exists(os.path.join(outdir, geojsonfiles[x]))]
    geojsonfiles = [geojsonfiles[x] for x in exinds]
    fctimes = fctimes[exinds]
    fctimes2 = fctimes2[exinds]

    # generate metadatafile
    if len(geojsonfiles) > 0:
        utils.make_metadata(origintime, fctimes, fctimes2, variable,
                            fctype=fctype,
                            template=metadatatemplate,
                            bb=bb,
                            fcfiles=geojsonfiles,
                            file=metadatafile)
        logging.info(f'Wrote {metadatafile}')

        if upload_to_datalake:
            logging.info('Upload to SAFRES datalake')
            upload_files(metadatafile, outdir)
            logging.info('Upload to SAFRES datalake DONE')

logging.info('Done all for %s %s', fctype, origintime)
