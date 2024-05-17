#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Generate fire weather index (FWI) from ECMWF HRES nc file.

marko.laine@fmi.fi
"""

import sys
import os
import logging
import argparse

import numpy as np
import xarray as xr
from pandas import to_datetime

import safers_utils as utils

from safers_data_tables import safers_hres_domain, safers_hres_res
from calculateFWI_xr import init_era5, init_prev
from calculateFWI_xr import calculate_FWI_ds, dstonoon

from FWI_functions_xr import add_daylength

from upload_EC import upload_nc

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--log", default='stderr', type=str)
parser.add_argument("--level", default='INFO', type=str)
parser.add_argument("--outdir", default='/data/tmp/safers_fwi/', type=str)
parser.add_argument("--metadir", default='/data/tmp/metadata/', type=str)
parser.add_argument("--fwidir", default='/data/tmp/safers_fwi', type=str)
parser.add_argument("--upload", action='store_true')
parser.add_argument("--era5", action='store_true')
parser.add_argument("--nolsm", action='store_true')
parser.add_argument("--datadir", default='/data/safers/data', type=str)
parser.add_argument("--ncdir", default='/data/tmp/safers_nc/', type=str)
parser.add_argument("--mapdata", default=None, type=str)
parser.add_argument("--ncin", type=str)
parser.add_argument("--nc", type=str)
opts = parser.parse_args(sys.argv[1:])

if opts.log == 'stderr':
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None))
else:
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None),
                        filename=opts.log)

# %%
# Capture warnings to log file (e.g. RuntimeWarning from xarray)
logging.captureWarnings(True)

#warnings.filterwarnings('ignore')
#np.seterr(all="ignore")
np.seterr(invalid="ignore", over="ignore")

# make outpur directories, if missing
os.makedirs(opts.outdir, exist_ok=True)
os.makedirs(opts.metadir, exist_ok=True)

fctype = 'HRES'
safers_domain = safers_hres_domain
safers_res = safers_hres_res

logging.info('opening nc %s', opts.ncin)

if opts.ncin is None:
    logging.error('ncin is None')
    sys.exit(1)
ds = xr.open_dataset(opts.ncin)
ds['ws'] = np.sqrt(np.square(ds['u10']) + np.square(ds['v10'])) * 3.6

forecast_reference_time = to_datetime(ds['forecast_reference_time'].values)

logging.info('Forecast reference time %s', forecast_reference_time)

logging.debug(ds)

# empty array
# HRES nc has latitudes from south to north
ds0 = xr.Dataset(coords=
                 {'time': [forecast_reference_time],
                  'latitude': np.arange(safers_domain[3], safers_domain[2] + safers_res, safers_res),
                  'longitude': np.arange(safers_domain[0], safers_domain[1] + safers_res, safers_res)}
                 )
ds0['forecast_reference_time'] = forecast_reference_time

logging.info('load mapdata')
mapdatafile = os.path.join(opts.datadir, f'Mapdata_{fctype}.nc')
mapdata = xr.open_dataset(mapdatafile)
mapdata['offset_summer'] = mapdata['offset_summer'].astype('timedelta64[h]')
mapdata['offset_winter'] = mapdata['offset_winter'].astype('timedelta64[h]')

# %%

logging.debug('initial values')

if opts.era5:
    ffmc0, dc0, dmc0 = init_era5(ds0)
else:
    ffmc0, dc0, dmc0 = init_prev(ds0, datadir=opts.fwidir, fctype=fctype)
logging.debug('FFMC0:')
logging.debug(ffmc0)

# %%

ds = add_daylength(ds, ds.time.dt.month.values[0])
ds = dstonoon(ds, mapdata)

# need to fix coordinates here
ds = ds.reindex_like(ds0.isel(time=0), method='nearest')

fwi = calculate_FWI_ds(ds, ffmc0, dc0, dmc0)
fwi = fwi.transpose("time", "latitude", "longitude")

# also here for some reason
mapdata = mapdata.reindex_like(fwi.isel(time=0), method='nearest')
if not opts.nolsm:
    fwi = fwi.where(mapdata.lsm != 0)

thisdir = os.path.dirname(os.path.realpath(__file__))
metadatatemplate = os.path.join(thisdir, 'metadata_template_ECMWF.json')

ncfile = opts.nc
logging.debug(fwi)

if ncfile is not None:
    ofile = os.path.join(opts.outdir, ncfile)
    fwi.to_netcdf(ofile)
    logging.info('saved to %s', ofile)
else:
    ofile = os.path.join(opts.outdir,
                         f'Fwi_{forecast_reference_time.strftime("%Y%m%d%H%M")}_{fctype}.nc')
    logging.info('NOT saved to %s', ofile)

# generate metadata
dataid = utils.safers_data_id(f'fwi_{fctype.lower()}')
metadatafile = os.path.join(opts.metadir,
                            f'metadata_fwi_{fctype}_{forecast_reference_time.strftime("%Y%m%d%H%M")}.json')

bb = utils.ds_bb(fwi)
fctimes = fwi.time.values
variables = list(fwi.data_vars.keys())

fwi.close()

utils.make_metadata_nc(forecast_reference_time, variables, fctimes,
                       fctype=fctype,
                       variable_type='fwi',
                       template=metadatatemplate,
                       dataid=dataid,
                       bb=bb,
                       funs=None,
                       name=os.path.basename(ofile),
                       file=metadatafile)
logging.info('Wrote %s', metadatafile)

# send to datalake
if opts.upload:
    logging.info('Upload to SAFRES datalake')
    upload_nc(metadatafile, ncfile)
    logging.info('Upload to SAFRES datalake DONE')

logging.info('Done all for %s %s', fctype, forecast_reference_time)
