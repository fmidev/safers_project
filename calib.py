# Functions for calibration

"""
Functions:
 `crps_ensemble`, `crps_gaussian`, `crps_truncated`, `crps_cencored`
Two first are from `xskillscore`, two other are similar but
defined using `scoringRules` R package.

Also loads `crch` from R as `r_crch`.
"""
from __future__ import annotations

import os
import warnings
from typing import Callable, List, Literal, Tuple, Union

# import glob
import pickle
import re
import datetime as dt

import pandas as pd
import xarray as xr
import numpy as np
import dask

from scipy.stats import norm
# from scipy.optimize import minimize
from scipy.stats import truncnorm

# from xskillscore import crps_ensemble, crps_gaussian

##

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
from rpy2.robjects import Formula

##

numpy2ri.activate()
pandas2ri.activate()

XArray = Union[xr.Dataset, xr.DataArray]
Dim = Union[List[str], str]

##

r_scoring = importr('scoringRules')
r_crch = importr('crch')

##

TIGGE_PATH = '/data/safers/tigge/leadtimes/tigge_new/'
# TIGGE_FILES = sorted(glob.glob(f'{TIGGE_PATH}tigge*.nc'))

REGIONS = {
    "France": 43.,
    "Germany": 121.,
    "Greece": 123.,
    "Spain": 132.,
    "Italy": 141.,
    "Finland": 151.,
    }

MASK_FILE = '/data/safers/countries_mask_ens.nc'


# this can load fwi ens files too
def load_tigge(files, funs=['mean', 'std'], variable='fwi', load=False):
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        dsall = (xr.open_mfdataset(files, combine="nested",
                                   concat_dim='forecast_reference_time',
                                   autoclose=True,
                                   preprocess=lambda ds: ds.swap_dims({'time': 'leadtime'})).
                 drop_vars(['heightAboveGround'], errors='ignore')
                 )

    obs = (dsall[variable].
           isel(leadtime=0).
           swap_dims({'forecast_reference_time': 'time'}).
           mean(dim='realization').
           drop_vars(['realization', 'heightAboveGround', 'step', 'leadtime'],
                     errors='ignore')
           )

    fc = []
    for fun in funs:
        fc += [ensmean(dsall[[variable]], fun=fun, rename=True)]
    fc = xr.merge(fc, compat='override')

    if load:
        fc.load()
        obs.load()

    return fc, obs


##
def load_tiggedata(file, funs=['mean', 'std', 'p50', 'p10', 'p90']):
    ds = xr.open_dataset(file)
    fc = ds.isel(step=1).drop_vars(['step', 'heightAboveGround'])
    obs = (ds.isel(step=0,  realization=0).
           drop_vars(['realization', 'heightAboveGround', 'step', 'leadtime']))
    # same as step
    fc['leadtime'] = pd.to_timedelta(fc.valid_time.values[0] - fc.time.values[0])

    if funs is not None:
        fc2 = []
        for fun in funs:
            rename = not fun == 'mean'
            fc2 += [ensmean(fc, fun=fun, rename=rename)]

        fcmean = xr.merge(fc2, compat='override')
        return fc, obs, fcmean

    return fc, obs


# copied from safers_utils.py
def ensmean(ds, fun='mean', ensvar='realization', rename=False):
    """Ensemble mean or other function."""
    if fun == 'p10':
        # logging.debug('calculate %s', fun)
        # da = ds.chunk({ensvar: -1}).quantile([0.1],
        da = ds.quantile([0.1],
                         dim=ensvar,
                         keep_attrs='True', skipna=False).isel(quantile=0)
    elif fun == 'p50':
        da = ds.quantile([0.5],
                         dim=ensvar,
                         keep_attrs='True', skipna=False).isel(quantile=0)
    elif fun == 'p90':
        da = ds.quantile([0.9],
                         dim=ensvar,
                         keep_attrs='True', skipna=False).isel(quantile=0)
    elif fun == 'mean':
        da = ds.mean(dim=ensvar, keep_attrs='True')
    elif fun == 'std':
        da = ds.std(dim=ensvar, keep_attrs='True')
    else:
        warnings.warn(f'Unknown function {fun}')
        da = None
        return da
    #for v in list(da.data_vars):
    #    da[v].attrs['FMI_ens_note'] = f'This is {variable_descriptions.get(fun)}'
    if rename:
        v = list(da.data_vars)
        da = da.rename(dict(zip(v, list(map(lambda x: x + '_' + fun, v)))))
    # da = da.astype(safers_datatype)
    return da


# Make new xskillscore type function. Uses R scoringRules or home made functions.
def crpst(obs, mu, sig):
    """Truncated CRPS using R package scoringRules"""
    obs = np.atleast_1d(obs)
    mu = np.atleast_1d(mu)
    sig = np.atleast_1d(sig)
#    dtype = obs.dtype
    out = r_scoring.crps_tnorm(robjects.FloatVector(obs.astype(float).ravel()),
                               location=robjects.FloatVector(mu.astype(float).ravel()),
                               scale=robjects.FloatVector(sig.astype(float).ravel()),
                               lower=0, upper=np.inf)
    return out.reshape(obs.shape)  # .astype(dtype)


def crpsc(obs, mu, sig):
    """Censored CRPS using R package scoringRules"""
    obs = np.atleast_1d(obs)
    mu = np.atleast_1d(mu)
    sig = np.atleast_1d(sig)
#    dtype = obs.dtype
    out = r_scoring.crps_cnorm(robjects.FloatVector(obs.astype(float).ravel()),
                               location=robjects.FloatVector(mu.astype(float).ravel()),
                               scale=robjects.FloatVector(sig.astype(float).ravel()),
                               lower=0, upper=np.inf)
    return out.reshape(obs.shape)  # .astype(dtype)


def crpsn(obs, mu, sig):
    """Gaussian CRPS using R package scoringRules"""
    obs = np.atleast_1d(obs)
    mu = np.atleast_1d(mu)
    sig = np.atleast_1d(sig)
    #    dtype = obs.dtype
    out = r_scoring.crps_norm(robjects.FloatVector(obs.astype(float).ravel()),
                              mean=robjects.FloatVector(mu.astype(float).ravel()),
                              sd=robjects.FloatVector(sig.astype(float).ravel()))
    return out.reshape(obs.shape)


def crps_trunc_gauss(obs, mu, sig):
    """Truncated Gaussian CRPS by Olle"""
    a = mu/sig
    z = (obs - mu)/sig
    Z = norm.cdf(z)
    Z0 = norm.cdf(a)
    Zsq = norm.cdf(np.sqrt(2)*a)
    zpdf = norm.pdf(z)
    crps = sig*(z*Z0*(2*Z+Z0-2) + 2*zpdf*Z0 - 1/np.sqrt(np.pi)*Zsq)/Z0**2
    return crps


##
# Code copied from xskillscore source
def crps_truncated(
    observations: XArray,
    mu: XArray | float | int,
    sig: XArray | float | int,
    dim: Dim = None,
    weights: XArray = None,
    keep_attrs: bool = False,
) -> XArray:
    """
    Truncated Gaussian CRPS for xskillscore
    """
    xmu = xr.DataArray(mu) if isinstance(mu, (int, float)) else mu
    xsig = xr.DataArray(sig) if isinstance(sig, (int, float)) else sig
    if xmu.dims != observations.dims:
        observations, xmu = xr.broadcast(observations, xmu)
    if xsig.dims != observations.dims:
        observations, xsig = xr.broadcast(observations, xsig)
    res = xr.apply_ufunc(
        crps_trunc_gauss,
        observations,
        xmu,
        xsig,
        input_core_dims=[[], [], []],
        # vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    if weights is not None:
        return res.weighted(weights).mean(dim, keep_attrs=keep_attrs)
    else:
        return res.mean(dim, keep_attrs=keep_attrs)


def crps_censored(
    observations: XArray,
    mu: XArray | float | int,
    sig: XArray | float | int,
    dim: Dim = None,
    weights: XArray = None,
    keep_attrs: bool = False,
) -> XArray:
    """
    Censored Gaussian CRPS for xskillscore
    """
    xmu = xr.DataArray(mu) if isinstance(mu, (int, float)) else mu
    xsig = xr.DataArray(sig) if isinstance(sig, (int, float)) else sig
    if xmu.dims != observations.dims:
        observations, xmu = xr.broadcast(observations, xmu)
    if xsig.dims != observations.dims:
        observations, xsig = xr.broadcast(observations, xsig)
    res = xr.apply_ufunc(
        crpsc,
        observations,
        xmu,
        xsig,
        input_core_dims=[[], [], []],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=keep_attrs,
    )
    if weights is not None:
        return res.weighted(weights).mean(dim, keep_attrs=keep_attrs)
    else:
        return res.mean(dim, keep_attrs=keep_attrs)


# copy dataarray and replace values from dataframe with matching index numbers
# Idea from https://github.com/pydata/xarray/issues/6377
def dftoda(da, df, fill=np.nan):
    df2 = pd.DataFrame()
    if fill is None:
        df2["values"] = da.values.ravel()
    else:
        df2["values"] = xr.ones_like(da).values.ravel() * fill
    df2["values"][df.index] = df
    return da.copy(data=df2["values"].values.reshape(da.shape))


def calibrate_train(fc, obs, mask=None, leadtime=1, modelfile=None, saveonly=False,
                    meanvar='mean', sdvar='sd', calibrate=False):
    """Calibrate forecasts"""
    # leadtime = np.atleast_1d(leadtime)

    # fci = (fc.
    #      isel(leadtime=leadtime).
    #      swap_dims({'forecast_reference_time': 'time'})
    #      )
    fci = fc.isel(leadtime=leadtime)

    if np.any(np.isin('forecast_reference_time', list(fci.dims))):
        fci = fci.swap_dims({'forecast_reference_time': 'time'})

    # Select only matching times
    times = np.intersect1d(fci['time'].values, obs['time'].values)
    fci = fci.sel(time=times)
    obs = obs.sel(time=times)

    variable = obs.name
    out = obs.to_dataset(name=variable)
    out[meanvar] = fci[meanvar]
    out[sdvar] = fci[sdvar]
    if mask is not None:
        out = out.where(mask)

    good = out[sdvar] > 1e-8
    out[sdvar] = out[sdvar].where(good)

    df = pd.DataFrame({
        'y': out[variable].values.ravel(),
        'mean': out[meanvar].values.ravel(),
        'sd': out[sdvar].values.ravel()
    }).dropna()

    r_model = r_crch.crch(Formula('y ~ mean | sd'), left=0.0, type='crps', data=df)

    if modelfile is not None:
        with open(modelfile, 'wb') as f:
            pickle.dump(r_model, f, protocol=-1)

    if saveonly:
        return
    if not calibrate:
        return r_model

    df['mean_cal'] = r_crch.predict_crch(r_model, type='location', new=df)
    df['sd_cal'] = r_crch.predict_crch(r_model, type='scale', new=df)

    out[meanvar + '_cal'] = dftoda(out[meanvar], df['mean_cal'], fill=None)
    out[sdvar + '_cal'] = dftoda(out[sdvar], df['sd_cal'], fill=None)
    out['leadtime'] = fc['leadtime'].isel(leadtime=leadtime)

    return out


def calibrate_predict(modelfile, fc, leadtime=1, mask=None,
                      meanvar='mean', sdvar='sd'):
    """Calibrate forecast given model in modefile."""

    if isinstance(modelfile, str):
        with open(modelfile, 'rb') as f:
            r_model = pickle.load(f)
    else:
        r_model = modelfile

    out = fc.isel(leadtime=leadtime).copy()

    if np.any(np.isin('forecast_reference_time', list(out.dims))):
        out = out.swap_dims({'forecast_reference_time': 'time'})

    if mask is not None:
        out = out.where(mask)

    df = pd.DataFrame({
        'mean': out[meanvar].values.ravel(),
        'sd': out[sdvar].values.ravel()
    }).dropna()

    df['mean_cal'] = r_crch.predict_crch(r_model, type='location', new=df)
    df['sd_cal'] = r_crch.predict_crch(r_model, type='scale', new=df)

    out[meanvar + '_cal'] = dftoda(out[meanvar], df['mean_cal'], fill=None)
    out[sdvar + '_cal'] = dftoda(out[sdvar], df['sd_cal'], fill=None)
    out['leadtime'] = fc['leadtime'].isel(leadtime=leadtime)
    out['forecast_reference_time'] = fc.isel(leadtime=leadtime)['forecast_reference_time']

    return out


def valid_files(folder_path, given_date, delta=-30):
    # Define a function to extract the date from the filename
    def extract_date_from_filename(filename):
        match = re.search(r'(\d{8})', filename)
        if match:
            date_str = match.group(1)
            return dt.datetime.strptime(date_str, '%Y%m%d')
        return None

    # Calculate the date 30 days ago
    date_30_days_ago = given_date + dt.timedelta(days=delta)
    #logging.debug(date_30_days_ago)
    # List all files in the folder
    if isinstance(folder_path, list):
        files_in_folder = folder_path
        path = os.path.dirname(folder_path[0])
    else:
        files_in_folder = os.listdir(folder_path)
        path = folder_path
    output = []
    # Filter and print files with dates within the last 30 days
    for filename in files_in_folder:
        file_date = extract_date_from_filename(filename)
        if (delta < 0 and file_date and file_date >= date_30_days_ago and
            file_date <= given_date) or (delta > 0 and file_date and
                                         file_date <= date_30_days_ago
                                         and file_date >= given_date):
            output += [path + '/' + os.path.basename(filename)]
    return output


def train_many(files, mask=None, leadtimes=np.arange(1, 15),
               variable='fwi',
               meanvar='fwi_mean', sdvar='fwi_std',
               fc=None, obs=None):

    if (fc is None) | (obs is None):
        fc, obs = load_tigge(files, variable=variable, load=True)
    model = dict(zip(leadtimes,
                     [calibrate_train(fc, obs, mask=mask, leadtime=l,
                                      meanvar=meanvar, sdvar=sdvar)
                      for l in leadtimes]))
    return model


def calibrate_many(ds, model, mask, leadtimes, type='MEAN',
                   meanvar='fwi_mean', sdvar='fwi_std'):
    #nlead = len(model)    
    #leadtimes = np.arange(firstlead, nlead-firstlead)

    if type == 'ENS':
        ds = xr.merge([ensmean(ds, fun='mean', rename=True),
                       ensmean(ds, fun='std', rename=True)])

    ds = ds.swap_dims({'time': 'leadtime'})

    ds = xr.concat([calibrate_predict(model[lt], ds, mask=mask, leadtime=lt,
                                      meanvar=meanvar, sdvar=sdvar)
                    for lt in leadtimes],
                   dim='leadtime')
    ds = ds.swap_dims({'leadtime': 'time'})

    # truncnorm conversion
    mean = ds[meanvar + '_cal'].values
    std = ds[sdvar + '_cal'].values
    a = -mean/std
    b = np.full(a.shape, fill_value=np.inf)  # np.inf
    ds[meanvar + '_cal'] = (ds.dims, truncnorm.mean(a, b, loc=mean, scale=std))
    ds[sdvar + '_cal'] = (ds.dims, truncnorm.std(a, b, loc=mean, scale=std))
    return ds


def getmask(region):
    """Get country mask"""
    if region is None:
        return None
    masks = xr.open_dataset(MASK_FILE)['region']
    mask = masks == REGIONS[region]
    return mask
