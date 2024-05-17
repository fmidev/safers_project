#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process grib file for several variable for given thresholds.
One threshold per variable

./safers_threshold.py --variables t2m r2 ws10 --thresholds 30 30 8.33 \
    --thtypes larger smaller larger \
    --out /data/tmp/thresholds.nc \
    --grib /data/tmp/ensgrib/fc202210150000.grib

./plotnc.py /data/tmp/thresholds.nc --out /tmp/threshold.png --var t2m_prob --time 20 --cmap inferno_r

./animatenc.py /data/tmp/thresholds.nc --out /tmp/animation.gif --var t2m_prob  --cmap inferno_r

"""

# marko.laine@fmi.fi

import sys
import os
import logging
import argparse
import numpy as np
import dask

from pandas import to_datetime

import safers_utils as utils
import safers_grib
# from upload_EC import upload_nc
from safers_s3 import parsefile_ENS

from safers_data_tables import safers_ext_domain, safers_ext_res
from safers_data_tables import safers_ens_domain, safers_ens_res

from safers_data_tables import safers_attrs

from upload_EC import upload_nc

# parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--log", default='stderr', type=str)
parser.add_argument("--level", default='INFO', type=str)
parser.add_argument("--outdir", default='/data/tmp/safers_nc/', type=str)
parser.add_argument("--out", default=None, type=str)
parser.add_argument("--metadir", default='/data/tmp/metadata/', type=str)
parser.add_argument("--upload", action='store_true')
# parser.add_argument("--zlib", action='store_true')
# parser.add_argument("--discrete", action='store_true')
parser.add_argument("--le", action='store_true')
parser.add_argument("--grib", type=str)
parser.add_argument("--fctype", default="ENS", type=str)
parser.add_argument("--variables", type=str, nargs="+")
# parser.add_argument("--variable", type=str)
parser.add_argument("--thresholds", type=float, default=[24], nargs="+")
parser.add_argument("--thtypes", type=str, default=['larger'], nargs="+")
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
ths = opts.thresholds
thtypes = opts.thtypes
#variable = opts.variable
variables =opts.variables
outdir = opts.outdir
metadir = opts.metadir
thisdir = os.path.dirname(os.path.realpath(__file__))

# generate output directory
os.makedirs(outdir, exist_ok=True)
os.makedirs(metadir, exist_ok=True)

metadatatemplate = os.path.join(thisdir, 'metadata_template_ECMWF.json')

gribfile = opts.grib
f1, f2, origintime = parsefile_ENS(gribfile)

if fctype == 'ENS':
    safers_res = safers_ens_res
    safers_domain = safers_ens_domain
elif fctype == 'EXT':
    safers_res = safers_ext_res
    safers_domain = safers_ext_domain
else:
    logging.error('Unknown fctype %s', fctype)
    sys.exit(1)


logging.debug('Opening grib')
#if variable == "tp24":
#    ds = safers_grib.opengrib(gribfile, type=fctype, variable=['tp'])
#else:
#    ds = safers_grib.opengrib(gribfile, type=fctype, variable=[variable])
variables_grib = [i for i in variables if i not in ['r2', 'tp24', 'ws', 'rule30']]

if np.any(np.isin('rule30', variables)):
    variables_grib += ['t2m', 'd2m', 'u10', 'v10']
if np.any(np.isin('r2', variables)):
    variables_grib += ['t2m', 'd2m']
if np.any(np.isin('ws10', variables)):
    variables_grib += ['u10', 'v10']

variables_grib = list(np.unique(variables_grib))

ds = safers_grib.opengrib(gribfile, type=fctype, variable=variables_grib)

ds = (ds.swap_dims({'leadtime': 'time'}).
      rename({'lon': 'longitude', 'lat': 'latitude'}))


if np.any(np.isin('tp', variables)):
    logging.debug('Calculate tp24')
    ds = utils.addtp24(ds, drop=False)
    ds = ds.isel(time=(ds['leadtime'] >= np.timedelta64(24, 'h')).
                 values.nonzero()[0])
    logging.debug('Calculate tp24 done')

if np.any(np.isin('r2', variables)) | np.any(np.isin('rule30', variables)):
    ds = utils.addrh(ds, units='K')
    logging.debug('Added r2')

if np.any(np.isin('ws10', variables)) | np.any(np.isin('rule30', variables)):
    logging.debug('Calculate wind speed')
    ds = utils.uvtows(ds, dwi=False, drop=False)
    logging.debug('Calculate wind speed done')

ds = utils.safers_convert(ds)
ds = safers_attrs(ds)
logging.debug('Opening grib done')

nens = len(ds['realization'])


# process first variable
variable = variables[0]
th = ths[0]
thtype = thtypes[0]

if variable == 'tp':
    variable == 'tp24'
name = f'{variable}_prob'

if variable == 'rule30':
    da = (((ds['t2m'] > 30) &
           (ds['r2'] < 30) & 
           (ds['ws10'] > 8.333)).
          sum(dim='realization') / nens * 100)
    long_name = 't2m > 30 °C, r2 < 30 %, ws10 > 8.33 m/s'
    da.attrs['long_name'] = long_name
    da.attrs['units'] = '%'
    da.attrs['threshold'] = '30, 30, 8.33'
    da.attrs['threshold_type'] = f'larger, smaller, larger'
else:
    orig_name = ds[variable].attrs["long_name"]
    units = ds[variable].attrs["units"]
    logging.debug(name)

    if thtype == 'larger':
        da = (ds[variable] > th).sum(dim='realization') / nens * 100
        long_name = f'{orig_name}, prob > {th} {units}'
    elif thtype == 'larger':
        da = (ds[variable] < th).sum(dim='realization') / nens * 100
        long_name = f'{orig_name}, prob < {th} {units}'
    da.attrs['long_name'] = long_name
    da.attrs['units'] = '%'
    da.attrs['threshold'] = th
    da.attrs['threshold_type'] = f'{thtype} than'
    da.attrs['original_variable'] = variable

logging.debug(long_name)

da.name = name
ds2 = da.to_dataset()

# process rest of the thresholds
if len(variables) > 1:
    for i in range(1, len(variables)):
        variable = variables[i]
        th = ths[i]
        thtype = thtypes[i]
        units = ds[variable].attrs["units"]
        name = f'{variable}_prob'
        ii = 1
        name0 = name
        #while np.any(np.isin(name, list(ds2.data_vars.keys()))):
        #    name = f'{name0}{ii}'
        #    ii += 1
        #    pass
        orig_name = ds[variable].attrs["long_name"]
        if thtype == 'larger':
            long_name = f'{orig_name}, prob > {th} {units}'
            ds2[name] = (ds[variable] > th).sum(dim='realization') / nens * 100
        else:
            long_name = f'{orig_name}, prob < {th} {units}'
            ds2[name] = (ds[variable] < th).sum(dim='realization') / nens * 100
        ds2[name].attrs['long_name'] = long_name
        ds2[name].attrs['units'] = '%'
        ds2[name].attrs['threshold'] = th
        ds2[name].attrs['threshold_type'] = f'{thtype} than'
        ds2[name].attrs['original_variable'] = variable
        logging.debug(name)
        logging.debug(long_name)

ds2.load()

logging.debug('Thresholding done')

# rotate
#ds2 = (ds2.swap_dims({'leadtime': 'time'}).
#      rename({'lon': 'longitude', 'lat': 'latitude'}))
ds2 = utils.rotated_ll_to_regular(ds2, realization='time',
                                  llmm=safers_domain, res=safers_res,
                                  lon='longitude', lat='latitude')

logging.debug('Rotation done')

ds2 = ds2.drop_vars(['surface', 'depth', 'realization',
                     'entireAtmosphere', 'quantile'],
                    errors='ignore')

ds2.attrs = ds.attrs
ds2.attrs['FMI_note'] = 'Contains threshold exceedance percentages'

if opts.out is not None:
    ncfile = opts.out
else:
    ncfile = '/data/tmp/thresholds.nc'
utils.savenc(ds2, ncfile, zlib=False, int8=True)

logging.debug('Saving done')

#sys.exit()
# metadata

origintime = to_datetime(ds2.forecast_reference_time.values)
dataid = utils.safers_data_id('threshold')
metadatafile = os.path.join(opts.metadir,
                            f'metadata_threshold_{fctype}_{origintime.strftime("%Y%m%d%H%M")}.json')

bb = utils.ds_bb(ds2)
fctimes = ds2.time.values
variables = list(ds2.data_vars.keys())
metadatatemplate = os.path.join(thisdir, 'metadata_template_ECMWF.json')

extrameta = {}
for v in variables:
    extrameta[v] = {
        'name': v,
        'long_name': ds2[v].attrs.get('long_name'),
        'threshold': ds2[v].attrs.get('threshold'),
        'threshold_type': ds2[v].attrs.get('threshold_type'),
        'units': ds2[v].attrs.get('units'),
        }

utils.make_metadata_nc(origintime, variables, fctimes,
                       fctype=fctype,
                       variable_type='threshold',
                       dataid=dataid,
                       template=metadatatemplate,
                       bb=bb,
                       funs=['prob'],
                       name=os.path.basename(ncfile),
                       file=metadatafile,
                       extrameta=extrameta)
logging.info('Wrote %s', metadatafile)


# Send to SAFERS data lake

if opts.upload:
    logging.info('Upload to SAFRES datalake')
    upload_nc(metadatafile, ncfile)
    logging.info('Upload to SAFRES datalake DONE')
