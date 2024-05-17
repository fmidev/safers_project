#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FWI ens processing

Opens ECMWF ens grib file and calculates FWI for all members.
Optionally calulates summary statistics and anomalies against ERA5 climatology.
Gets initial values for ffmc, dc, and dmc from previous run or from ERA5 climatology.

For parallel processing the grib file has to be indexed before starting.
So opened once using `xarray` / `cfgrib`.

marko.laine@fmi.fi
"""


import sys
import os
import glob
import logging
import argparse
import time
from joblib import Parallel, delayed

import numpy as np
import xarray as xr

import safers_utils as utils
from calculateFWI_xr import init_era5, init_prev
from calculateFWI_xr import calculate_FWI_ENS_1, add_anomalies
from safers_data_tables import safers_ext_domain, safers_ext_res
from safers_data_tables import safers_ens_domain, safers_ens_res
from safers_s3 import parsefile_ENS
from upload_EC import upload_nc

# %% command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--log", default='stderr', type=str)
parser.add_argument("--level", default='INFO', type=str)
parser.add_argument("--outdir", default='/data/tmp/safers_fwi', type=str)
parser.add_argument("--fwidir", default='/data/tmp/safers_fwi', type=str)
parser.add_argument("--metadir", default='/data/tmp/metadata', type=str)
parser.add_argument("--datadir", default='/data/safers/data', type=str)
parser.add_argument("--nc", default=None, type=str)
parser.add_argument("--ncall", default=None, type=str)
parser.add_argument("--fctype", default='ENS', type=str)
parser.add_argument("--grib", default=None, type=str)
parser.add_argument("--upload", action='store_true')
parser.add_argument("--load", action='store_true')
parser.add_argument("--summary", default=True, action='store_true')
parser.add_argument("--nosummary", action='store_true')
parser.add_argument("--anomaly", action='store_true')
parser.add_argument("--noanomaly", action='store_true')
parser.add_argument("--median", action='store_true')
parser.add_argument("--era5", action='store_true')
parser.add_argument("--nens", default=50, type=int)
parser.add_argument("--ncores", default=-1, type=int)
opts = parser.parse_args(sys.argv[1:])
if opts.nosummary:
    opts.summary = False
if opts.noanomaly:
    opts.anomaly = False

# %% logging
if opts.log == 'stderr':
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None))
else:
    logging.basicConfig(level=getattr(logging, opts.level.upper(), None),
                        filename=opts.log)

# Capture warnings to log file (e.g. RuntimeWarning from xarray)
logging.captureWarnings(True)

# warnings.filterwarnings('ignore')
# np.seterr(all="ignore")
np.seterr(invalid='ignore', over='ignore', divide='ignore')

# %%

t00 = time.perf_counter()

# %%

fctype = opts.fctype

if opts.grib is None:
    file = max(glob.iglob(f'/data/tmp/{fctype.lower()}grib/*.grib'),
               key=os.path.getctime)
else:
    file = opts.grib

_, _, forecast_reference_time = parsefile_ENS(file)

logging.info('Using %s', file)
logging.info('Forecast reference time %s', forecast_reference_time)
logging.info('Nens %d, fctype %s', opts.nens, opts.fctype)

# %%

variables = ['t2m', 'd2m', 'u10', 'v10', 'tp']
fwi_variables = ['t2m', 'r2', 'ws', 'tp24']

if fctype == 'ENS':
    safers_domain = safers_ens_domain
    safers_res = safers_ens_res
elif fctype == 'EXT':
    safers_domain = safers_ext_domain
    safers_res = safers_ext_res
else:
    msg = f'fctype {fctype} not known'
    logging.error(msg)
    sys.exit(1)

# %%

thisdir = os.path.dirname(os.path.realpath(__file__))

# generate output directory
os.makedirs(opts.outdir, exist_ok=True)
os.makedirs(opts.metadir, exist_ok=True)

# %%
# empty array
ds0 = xr.Dataset(coords={'time': [forecast_reference_time],
                         'latitude': np.arange(safers_domain[2],
                                               safers_domain[3] - safers_res,
                                               -safers_res),
                         'longitude': np.arange(safers_domain[0],
                                                safers_domain[1] + safers_res,
                                                safers_res)}
                 )
ds0['forecast_reference_time'] = forecast_reference_time

#%%

logging.info('load mapdata')
mapdatafile = os.path.join(opts.datadir, f'Mapdata_{fctype}.nc')
mapdata = xr.open_dataset(mapdatafile)
mapdata['offset_summer'] = mapdata['offset_summer'].astype('timedelta64[h]')
mapdata['offset_winter'] = mapdata['offset_winter'].astype('timedelta64[h]')
mapdata['latitude'] = ds0['latitude']
mapdata['longitude'] = ds0['longitude']
#mapdata['latitude'] = np.arange(safers_domain[2], safers_domain[3] - safers_res, -safers_res)
#mapdata['longitude'] = np.arange(safers_domain[0], safers_domain[1] + safers_res, safers_res)

#%% initial values

logging.debug('initial values')

if opts.era5:
    ffmc0, dc0, dmc0 = init_era5(ds0)
else:
    ffmc0, dc0, dmc0 = init_prev(ds0, datadir=opts.fwidir, fctype=fctype)
logging.debug('FFMC0:')
logging.debug(ffmc0)
#ffmc0, dc0, dmc0 = init_default()
# sys.exit(0)

#%%
logging.info('looping over realizations')
chunks = {'step': 2}
chunks = None

# start from one to skip realization 0 which the reference forecast
fwi = Parallel(n_jobs=opts.ncores)(
    delayed(calculate_FWI_ENS_1)(file, inumber=inumber,
                                 mapdata=mapdata, chunks=chunks,
                                 fctype=fctype, level=opts.level.upper(),
                                 log=opts.log,
                                 ffmc0=ffmc0, dc0=dc0, dmc0=dmc0)
    for inumber in range(1, opts.nens + 1))

#fwi = [x for x in fwi if x is not None]
# print(fwi[0])
fwi = xr.concat(fwi, 'realization')
logging.info('looping done')

if opts.ncall is not None:
    ofile = os.path.join(opts.outdir, opts.ncall)
    # fwi.to_netcdf(ofile)
    utils.savenc(fwi, file=ofile, zlib=False, discrete=True)
    logging.info('saved to %s', ofile)


#%% calculate summary statistics

funs = ['mean', 'std', 'p10', 'p90']
if opts.median:
    funs.append('p50')
if opts.summary:
    logging.info('summary statistics')
    fwi2 = []
    for fun in funs:
        rename = not fun == 'mean'
        fwi2 += [utils.ensmean(fwi, fun=fun, rename=rename)]

    fwi = xr.merge(fwi2, compat='override')
    logging.info('summary statistics done')

fwi = fwi.drop_vars(['surface', 'depth', 'entireAtmosphere',
                     'quantile', 'ordinal_day'],
                    errors='ignore')

fwi['latitude'].attrs = {'units': 'degrees_north',
                         'standard_name': 'latitude',
                         'long_name': 'latitude'}
fwi['longitude'].attrs = {'units': 'degrees_east',
                          'standard_name': 'longitude',
                          'long_name': 'longitude'}
fwi['time'].attrs = {'standard_name': 'time',
                     'long_name': 'Day of forecast'}

# %% add anomalies
if opts.anomaly and opts.summary:
    logging.info('anomaly...')
    fwi = add_anomalies(fwi)
    logging.info('anomaly done')

# %% save to file
if opts.nc is not None:
    ofile = os.path.join(opts.outdir, opts.nc)
    # fwi.to_netcdf(ofile)
    utils.savenc(fwi, file=ofile, zlib=False, discrete=True)
    logging.info('saved to %s', ofile)
else:
    ofile = os.path.join(opts.outdir,
                         f'Fwi_{forecast_reference_time.strftime("%Y%m%d%H%M")}_{fctype}.nc')
    logging.info('NOT saved to %s', ofile)

#%% done

e1 = time.perf_counter() - t00
logging.info('Total time processing: %s',
             time.strftime("%H:%M:%S", time.gmtime(e1)))

# if not calculating summaries, do not make metadata or upload to datalake
if not opts.summary:
    sys.exit()

# generate metadata

origintime = forecast_reference_time
dataid = utils.safers_data_id(f'fwi_{fctype.lower()}')
metadatafile = os.path.join(opts.metadir,
                            f'metadata_fwi_{fctype}_{origintime.strftime("%Y%m%d%H%M")}.json')

bb = utils.ds_bb(fwi)
fctimes = fwi.time.values
variables = list(fwi.data_vars.keys())
metadatatemplate = os.path.join(thisdir, 'metadata_template_ECMWF.json')

utils.make_metadata_nc(origintime, variables, fctimes,
                       fctype=fctype,
                       variable_type='fwi',
                       dataid=dataid,
                       template=metadatatemplate,
                       bb=bb,
                       funs=funs if opts.summary else None,
                       name=os.path.basename(ofile),
                       file=metadatafile)
logging.info('Wrote %s', metadatafile)

if not opts.upload or not opts.summary:
    sys.exit()

# upload to datalake
if opts.upload:
    logging.info('Upload to SAFRES datalake')
    upload_nc(metadatafile, ofile)
    logging.info('Upload to SAFRES datalake DONE')

logging.info('Done all for fwi_%s %s', fctype, origintime)
e1 = time.perf_counter() - t00
logging.info('Total time: %s', time.strftime("%H:%M:%S", time.gmtime(e1)))
