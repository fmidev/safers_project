"""
Data utilities for SAFERS
"""

import os
import glob
from uuid import uuid4
import json
import warnings
import logging
from datetime import datetime

import numpy as np
from scipy.spatial import KDTree

import xarray as xr
import cf2cdm

from pandas import to_datetime, isnull  # , to_timedelta

from safers_data_tables import safers_data_id, ds_to_grib
from safers_data_tables import safers_ens_domain, safers_hres_domain, safers_ext_domain
from safers_data_tables import safers_hres_res, safers_ens_res, safers_ext_res
from safers_data_tables import safers_unit
from safers_data_tables import variable_descriptions
from safers_data_tables import safers_datatype
from safers_data_tables import i16encoding, ui16encoding, int16vars, uint16vars
from safers_data_tables import i8encoding
from safers_data_tables import fwi_vars, fwiencoding

# this is set here, maybe move to other locations?
xr.set_options(keep_attrs=True)


def replace_minmax(ds, minvalue=None, maxvalue=None):
    """Replace values smaller than minvalue and similarly for maxvalue."""
    if minvalue is not None:
        ds = ds.where((ds > minvalue) | np.isnan(ds), minvalue)
    if maxvalue is not None:
        ds = ds.where((ds < maxvalue) | np.isnan(ds), maxvalue)
    return ds


def safers_conversion_old(da, variable=None):
    """Convert DataArray variable units from ECMWF to SAFERS."""
    tempvars = ['t2m', 'd2m', 'mn2t6', 'mx2t6', 'mn2t3', 'mx2t3', 'mn2', 'mx2']
    if variable is None:
        variable = da.name
    basevar = variable.partition('_')[0]
    if np.any(np.isin(basevar, tempvars)):
        da = da - 273.15
        da.attrs['units'] = 'C'
    return da


def safers_conversion(da, variable=None):
    """Convert DataArray variable units from ECMWF to SAFERS."""
    tempvars = ['t2m', 'd2m', 'mn2t6', 'mx2t6', 'mn2t3', 'mx2t3', 'mn2', 'mx2']
    if variable is None:
        variable = da.name
    # basevar = variable.partition('_')[0]
    # basevar = variable  # FIXME, do not convert _std etc
    basevar = variable.partition('_')[0]
    fun = variable.partition('_')[2]  # function applied
    if np.any(np.isin(basevar, tempvars)) and fun != 'std':
        da = da - 273.15
        da.attrs['units'] = 'C'
        da.attrs['FMI_note'] = 'Units converted at FMI'
    # TEMPORARY FIX FOR ERROR IN THE ENS DATA 'kg m**-2 s-2'
    if basevar == 'tp' and da.attrs['units'] == 'm':
        da = da * 1000
        da.attrs['units'] = 'mm'
        da.attrs['FMI_note'] = 'Units converted at FMI'
    if basevar == 'msl' and da.attrs['units'] == 'Pa':
        da = da / 100.0
        da.attrs['units'] = 'hPa'
        da.attrs['FMI_note'] = 'Units converted at FMI'
    return da


def safers_convert(ds):
    """Convert all data variables in a DataSet."""
    variables = ds.data_vars
    for variable in variables:
        ds[variable] = safers_conversion(ds[variable])
    return ds


def addtp24(ds, drop=False, ncdir=None):
    """Calculate tp24 from tp and add it to the dataset."""
    if not np.all(np.isin(['tp'], list(ds.data_vars))):
        warnings.warn('tp missing in DataSet')
        return ds
    # ds['tp'] = np.maximum(0.0, ds['tp'])  # force tp to be non-negative
    tp = ds['tp']
    time1 = tp.time[tp.time >= tp.time[0] + np.timedelta64(24, 'h')]
    time2 = time1 - np.timedelta64(24, 'h')
    tp24 = xr.full_like(tp, fill_value=np.nan)
    try:
        tp24.loc[dict(time=time1)] = tp.sel(time=time1).values - tp.sel(time=time2).values
    except Exception as e:
        warnings.warn('No suitable times found in DataSet')
        return ds
    if tp.attrs.get('units') == 'm':
        ds['tp24'] = tp24 * 1000.0  # convert from m to mm
    else:
        ds['tp24'] = tp24
    ds['tp24'] = np.maximum(0.0, ds['tp24'])  # force tp to be non-negative
    ds['tp24'].attrs['long_name'] = 'total precipitation in the last 24 hours'
    ds['tp24'].attrs['units'] = 'mm'
    ds['tp24'].attrs['FMI_note'] = 'Converted from tp at FMI'
    if drop:
        ds = ds.drop(['tp'])
    return ds


def addprevtp24(ds, ncdir, variable='tp24'):
    if variable not in list(ds.data_vars):
        warnings.warn(f'{variable} is not in dataset')
        return ds
    fprev = prevfcfile(ds=ds, ncdir=ncdir)
    if fprev is not None:
        try:
            tp24prev = xr.open_dataset(fprev)[variable].astype(safers_datatype)
            tp24new = ds[variable].combine_first(tp24prev)
            tp24new['leadtime'] = ds[variable]['leadtime']
            tp24new['forecast_reference_time'] = ds[variable]['forecast_reference_time']
            ds[variable] = tp24new.astype(safers_datatype)
            tp24prev.close()
        except Exception:
            warnings.warn(f'Could not load previous {variable}: {fprev}')
    else:
        warnings.warn(f'Did not find previous {variable}')
    return ds


def convertssr(ds, ssrvar='ssr'):
    """Convert ssr to W/m**2 from J/m**2."""
    if not np.all(np.isin([ssrvar], list(ds.data_vars))):
        warnings.warn(f'{ssrvar} missing in DataSet')
        return ds
    ssr = ds[ssrvar]
    # print(ssr)
    timediff = np.diff(ds['leadtime'].values).astype('timedelta64[s]').astype('float32')
    time1 = ssr.time[1:]
    time2 = ssr.time[:-1]
    ssrw = xr.full_like(ssr, fill_value=np.nan)
    try:
        ssrw.loc[dict(time=time1)] = ssr.sel(time=time1).values - ssr.sel(time=time2).values
    except Exception:
        warnings.warn('Problem with ssr times, no conversion!')
        return ds
    ssrw[0] = ssr[0]
    #  ssrw = ssrw / np.r_[timediff[0], timediff][:, None, None]
    ssrw = ssrw / np.r_[timediff[0], timediff].reshape(np.r_[-1, np.ones(len(ds[ssrvar].shape)-1, dtype=int)])
    ds[ssrvar] = ssrw

    #leadtime = ds['leadtime'].values.astype('timedelta64[s]').astype('float')
    #ds[ssrvar] = ds[ssrvar] / leadtime.reshape(np.r_[-1, np.ones(len(ds[ssrvar].shape)-1, dtype=int)])
    ds[ssrvar].attrs['units'] = 'W m**-2'
    ds[ssrvar].attrs['FMI_note'] = 'Units converted at FMI'
    return ds


def uvtows(ds, drop=True, ws=True, dwi=True, wsname='ws10'):
    """Convert u10 and v10 to ws10 and dwi10."""
    if ws:
        ds[wsname] = np.sqrt(np.square(ds['u10']) + np.square(ds['v10']))
        ds[wsname].attrs = ds['u10'].attrs.copy()
        ds[wsname].attrs['long_name'] = '10 metre wind speed'
    if dwi:
        ds['dwi10'] = np.mod(180 + np.rad2deg(np.arctan2(ds['u10'], ds['v10'])), 360)
        ds['dwi10'].attrs = ds['u10'].attrs.copy()
        ds['dwi10'].attrs['long_name'] = '10 metre wind direction'
        ds['dwi10'].attrs['units'] = 'degree'
    if drop:
        ds = ds.drop_vars(['u10', 'v10'])
    return ds


def addrh(ds, r2name='r2', units='C', limit=False):
    """Add relative humidity."""
    if not np.all(np.isin(['d2m', 't2m'], list(ds.data_vars))):
        warnings.warn('t2m or d2m missing in DataSet')
        return ds
    if units == 'C':
        ds[r2name] = 100 * (np.exp((17.625 * ds['d2m']) /
                                   (243.04 + ds['d2m'])) /
                            np.exp((17.625 * ds['t2m']) /
                                   (243.04 + ds['t2m'])))
    else:
        ds[r2name] = 100 * (np.exp((17.625 * (ds['d2m'] - 273.15)) /
                                   (ds['d2m'] - 30.11)) /
                            np.exp((17.625 * (ds['t2m'] - 273.15)) /
                                   (ds['t2m'] - 30.11)))
    if limit:  # force the value between 0 and 100
        ds[r2name] = np.minimum(100.0, np.maximum(0.0, ds[r2name]))
    ds[r2name].attrs = ds['t2m'].attrs.copy()
    ds[r2name].attrs['GRIB_paramId'] = 260242
    ds[r2name].attrs['GRIB_cfName'] = 'relative_humidity'
    ds[r2name].attrs['GRIB_cfVarName'] = 'r2'
    ds[r2name].attrs['standard_name'] = 'relative_humidity'
    ds[r2name].attrs['GRIB_shortname'] = '2r'
    ds[r2name].attrs['GRIB_name'] = '2 metre relative humidity'
    ds[r2name].attrs['long_name'] = '2 metre relative humidity'
    ds[r2name].attrs['GRIB_units'] = '%'
    ds[r2name].attrs['units'] = '%'
    ds[r2name].attrs['FMI_note'] = 'Calculated at FMI'
    return ds


# do not use this anymore
def opengrib(file, type='HRES', cf=True, variable=None, keys=None):
    """Read ECMWF grib file as dataset."""
    if type.upper() == 'ENS':
        gribargs = {'filter_by_keys': {'dataType': 'pf'}}
    else:
        gribargs = {}
    if variable is not None:
        variable = ds_to_grib(variable)
        gribargs.update({'filter_by_keys': {'shortName': variable}})
    # gribargs.update({'indexpath': ''})
    if keys is not None:
        gribargs.update(keys)
    ds = xr.open_dataset(file, backend_kwargs=gribargs,
                         engine='cfgrib', decode_cf=cf)
    if cf:
        ds = cf2cdm.translate_coords(ds, cf2cdm.CDS)
    return ds


def datafilename(origintime, fctime, varid=99999, hours=3, fctime2=None,
                 origin='EC', producer='FMI', fctype='ENS', variable='t2m',
                 fun='', filetype='nc', lang='EN'):
    """File name generator for SAFRES files."""
    timefmt = '%Y%m%dT%H%MZ'
    # forecast end
    # fcend = fctime + np.timedelta64(hours, 'h')
    # for netCDF files with many variables
    # variable = '' if variable == 'basic' or variable == 'all' else f'_{variable}_'
    if filetype == 'nc':
        filename = (f'{varid}_{producer}_{origin}_{fctype}_{variable}_' +
                    origintime.strftime(timefmt) + '_' +
                    fctime.strftime(timefmt) + '_' +
                    fctime2.strftime(timefmt) + '_' +
                    lang + '.' + filetype
                    )
        return filename
    # else geoJSON names
    if fctype == 'ENS':
        filename = (f'{varid}_{producer}_{origin}_{fctype}_{variable}_{fun}_' +
                    origintime.strftime(timefmt) + '_' +
                    fctime.strftime(timefmt) + '_' +
                    fctime2.strftime(timefmt) + '_' +
                    lang + '.' + filetype
                    )
    else:
        filename = (f'{varid}_{producer}_{origin}_{fctype}_{variable}_' +
                    origintime.strftime(timefmt) + '_' +
                    fctime.strftime(timefmt) + '_' +
                    fctime2.strftime(timefmt) + '_' +
                    lang + '.' + filetype
                    )
    return filename


def rotated_ll_to_regular(ds, llmm=safers_ens_domain, res=safers_ens_res,
                          variables=None,
                          fctype=None,
                          method='nearest',
                          realization='realization',
                          lon='lon',
                          lat='lat'):
    """Interpolate dataset to regular lonlat grid."""
    if fctype is not None:
        if fctype == 'ENS':
            llmm = safers_ens_domain
            res = safers_ens_res
        elif fctype == 'EXT':
            llmm = safers_ext_domain
            res = safers_ext_res
        else:
            logging.error('fctype %s not known', fctype)

    lat0 = ds[lat].values.ravel()
    lon0 = ds[lon].values.ravel()
    number = ds[realization].values.ravel()
    lon1 = np.arange(llmm[0], llmm[1]+res, res)
    lat1 = np.arange(llmm[2], llmm[3]-res, -res)
    lon2, lat2 = np.meshgrid(lon1, lat1, indexing='xy')

    #newdata = xr.Dataset(coords=dict(realization=ds[realization],
    #                                 lat=(["lat"], lat1),
    #                                 lon=(["lon"], lon1)))
    newdata = xr.Dataset(coords={realization: ds[realization],
                                 lat: ([lat], lat1),
                                 lon: ([lon], lon1)})

    # copy attributes
    newdata.attrs = ds.attrs.copy()
    newdata[lon].attrs = ds[lon].attrs.copy()
    newdata[lat].attrs = ds[lat].attrs.copy()
    newdata[realization].attrs = ds[realization].attrs.copy()
    newdata = newdata.assign_coords({'forecast_reference_time': ds.get('forecast_reference_time'),
                                     'leadtime': ds.get('leadtime')})
    newdata['time'].attrs = ds['time'].attrs.copy()
    newdata['leadtime'].attrs = ds['leadtime'].attrs.copy()

    # nearest neighbour indexes
    tree = KDTree(np.c_[lon0, lat0])
    _, ii = tree.query(np.c_[lon2.ravel(), lat2.ravel()], k=1)

    if variables is None:
        variables = list(ds.data_vars.keys())
    for variable in variables:
        logging.debug('  rotating %s', variable)
        y = np.empty((number.size, lat1.size, lon1.size),
                     dtype=ds[variable].dtype)
        for i in range(y.shape[0]):
            y[i] = ds[variable].values[i].ravel()[ii].reshape((lat1.size, lon1.size))
        newdata[variable] = ((realization, lat, lon), y)
        newdata[variable].attrs = ds[variable].attrs.copy()

    return newdata


def rotated_ll_to_regular2(ds, llmm=safers_ens_domain, res=0.2,
                           variables=None,
                           method='nearest',
                           leadtime='leadtime',
                           realization='realization'):
    """Interpolate dataset to regular lonlat grid."""
    lat0 = ds.lat.values.ravel()
    lon0 = ds.lon.values.ravel()

    leadt = ds[leadtime].values.ravel()
    number = ds[realization].values.ravel()
    lon1 = np.arange(llmm[0], llmm[1]+res, res)
    lat1 = np.arange(llmm[2], llmm[3]-res, -res)
    lon, lat = np.meshgrid(lon1, lat1, indexing='xy')

    newdata = xr.Dataset(coords=dict(realization=ds[realization],
                                     leadtime=ds[leadtime],
                                     lat=(["lat"], lat1),
                                     lon=(["lon"], lon1)))

    # copy attributes
    newdata.attrs = ds.attrs.copy()
    newdata["lon"].attrs = ds["lon"].attrs.copy()
    newdata["lat"].attrs = ds["lat"].attrs.copy()
    newdata[realization].attrs = ds[realization].attrs.copy()
    newdata[leadtime].attrs = ds[leadtime].attrs.copy()

    # nearest neighbour indexes
    tree = KDTree(np.c_[lon0, lat0])
    _, ii = tree.query(np.c_[lon.ravel(), lat.ravel()], k=1)

    if variables is None:
        variables = list(ds.data_vars.keys())
    for variable in variables:
        y = np.empty((number.size, leadt.size, lat1.size, lon1.size),
                     dtype=ds[variable].dtype)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y[i, j] = ds[variable].values[i, j].ravel()[ii].reshape((lat1.size, lon1.size))
        newdata[variable] = ((realization, leadtime, 'lat', 'lon'), y)
        newdata[variable].attrs = ds[variable].attrs.copy()

    return newdata


def rotated_ll_to_regular3(ds, llmm=safers_ens_domain, res=0.2,
                           variables=None,
                           method='nearest',
                           leadtime='leadtime',
                           time='time',
                           realization='realization',
                           lon='longitude',
                           lat='latitude'):
    """Interpolate dataset to regular lonlat grid."""
    """ this version for converting grib members"""

    lat0 = ds[lat].values.ravel()
    lon0 = ds[lon].values.ravel()

    times = ds[time].values.ravel()
    number = ds[realization].values.ravel()
    lon1 = np.arange(llmm[0], llmm[1]+res, res)
    lat1 = np.arange(llmm[2], llmm[3]-res, -res)
    lon2, lat2 = np.meshgrid(lon1, lat1, indexing='xy')

    newdata = xr.Dataset(coords=dict(realization=ds[realization],
                                     time=ds[time],
                                     latitude=([lat], lat1),
                                     longitude=([lon], lon1)))

    # copy attributes
    newdata.attrs = ds.attrs.copy()
    newdata[lon].attrs = ds[lon].attrs.copy()
    newdata[lat].attrs = ds[lat].attrs.copy()
    newdata[realization].attrs = ds[realization].attrs.copy()
    newdata[time].attrs = ds[time].attrs.copy()

    # nearest neighbour indexes
    tree = KDTree(np.c_[lon0, lat0])
    _, ii = tree.query(np.c_[lon2.ravel(), lat2.ravel()], k=1)

    if variables is None:
        variables = list(ds.data_vars.keys())
    for variable in variables:
        y = np.empty((number.size, times.size, lat1.size, lon1.size),
                     dtype=ds[variable].dtype)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y[i, j] = ds[variable].data[i, j].ravel()[ii].reshape((lat1.size, lon1.size))
        newdata[variable] = ((realization, time, lat, lon), y)
        newdata[variable].attrs = ds[variable].attrs.copy()

    return newdata


def da_to_lonlat(da, llmm=safers_ens_domain, res=0.2,
                 method='nearest', ii=None):
    """Interpolate datarray to regular lonlat grid."""
    lat0 = da.lat.values.ravel()
    lon0 = da.lon.values.ravel()

    lon = np.arange(llmm[0], llmm[1], res)
    lat = np.arange(llmm[2], llmm[3], -res)

    # nearest neighbour indexes
    # todo, make these global
    if ii is None:
        LON, LAT = np.meshgrid(lon, lat, indexing='xy')
        tree = KDTree(np.c_[lon0, lat0])
        _, ii = tree.query(np.c_[LON.ravel(), LAT.ravel()], k=1)

    y = da.values.ravel()[ii].reshape((lat.size, lon.size))

    newdata = xr.DataArray(coords=dict(lat=(["lat"], lat), lon=(["lon"], lon)),
                           data=y)
    newdata.name = da.name
    newdata.attrs = da.attrs.copy()

    return newdata


def generate_tree(lon2d, lat2d, lon=None, lat=None,
                  llmm=safers_ext_domain, res=0.2, file=None):
    """Generate tree-based nearest neighbour indeces."""
    # the target grid
    if lon is None:
        lon = np.arange(llmm[0], llmm[1], res)
        lat = np.arange(llmm[2], llmm[3], -res)
    LON, LAT = np.meshgrid(lon, lat, indexing='xy')
    tree = KDTree(np.c_[lon2d.ravel(), lat2d.ravel()])
    _, ii = tree.query(np.c_[LON.ravel(), LAT.ravel()], k=1)
    return ii


def make_metadata_nc(origintime, variables, fctimes,
                     fctype='HRES',
                     template='metadataECMWF.json',
                     variable_type='all',
                     bb=safers_hres_domain,
                     funs=None,
                     dataid=None,
                     uid=None,
                     name=None,
                     file=None,
                     extrameta=None):
    """Generate metadata for one nc file."""
    timefmt = '%Y-%m-%dT%H:%M:%S'

    creationtime = datetime.utcnow().strftime(timefmt)
    time0 = to_datetime(origintime).strftime(timefmt)
    time1 = to_datetime(fctimes[0]).strftime(timefmt)

    i = -1
    time2 = to_datetime(fctimes[i])
    while isnull(time2):
        i = i - 1
        time2 = to_datetime(fctimes[i])

    time2 = time2.strftime(timefmt)
    # time2 = to_datetime(fctimes[-1]).strftime(timefmt)

    #time1 = to_datetime(fctimes[0])
    #if time1 is pd.NaT:
    #    time1 = to_datetime(fctimes[1])
    #time1 = time1.strftime(timefmt)

    note = 'Forecasts from ECMWF processed at FMI'
    title = ''
    res = 0
    if fctype == 'HRES':
        title = f'Deterministic forecasts for between {time1} and {time2}'
        note = 'Deterministic high resolution forecast from ECMWF processed at FMI.'
        res = safers_hres_res
    elif fctype == 'ENS':
        title = f'Ensemble forecasts between {time1} and {time2}'
        note = 'Ensemble forecast from ECMWF processed at FMI.'
        res = safers_ens_res
    elif fctype == 'EXT':
        title = f'Extended range ensemble forecasts between {time1} and {time2}'
        note = 'Extended range ensemble forecast from ECMWF processed at FMI.'
        res = safers_ext_res
    elif fctype == 'FWI':
        title = f'Forest fire index between {time1} and {time2}'
        note = 'Forest fire index (FWI) using ECMWF data calculated at FMI.'
    else:
        title = f'Ensemble forecasts between {time1} and {time2}'
        note = 'Ensemble forecast from ECMWF processed at FMI for several variables.'
        res = safers_ens_res
    if variable_type == 'extra':
        title = title + ' (extra variables)'
    if variable_type == 'lightning':
        title = title + ' (variables related to lightning)'
    if variable_type == 'fwi':
        title = title + ' (Fire Weather Index)'
    if variable_type == 'threshold':
        title = title + ' (threshold probabilities)'
    if funs is not None:
        note = note + ' Includes the following ensemble statistics: '
        for i, fun in enumerate(funs):
            note = note + f'{fun} = {variable_descriptions.get(fun)}'
            if i == len(funs) - 1:
                note = note + '.'
            else:
                note = note + ', '

    if uid is None:
        uid = str(uuid4())
    # need separate tables for HRES and ENS
    if dataid is None:
        dataid = safers_data_id(fctype)

    names = {"title": title,
             "notes": note,
             "name": uid}

    spatial = {"spatial":
               {"type": "MultiPolygon",
                "coordinates": [[[[bb[0], bb[3]],
                                  [bb[1], bb[3]],
                                  [bb[1], bb[2]],
                                  [bb[0], bb[2]],
                                  [bb[0], bb[3]]]]]}}

    resolution = {
        "quality_and_validity_spatial_resolution_latitude": res,
        "quality_and_validity_spatial_resolution_longitude": res,
        "quality_and_validity_spatial_resolution_scale": 1,
        "quality_and_validity_spatial_resolution_measureunit": "degree",
    }

    dates = {"data_temporal_extent_begin_date": time1,
             "data_temporal_extent_end_date": time2,
             "temporalReference_dateOfPublication": creationtime,
             "temporalReference_dateOfLastRevision": creationtime,
             "temporalReference_dateOfCreation": creationtime,
             "temporalReference_date": creationtime}

    extra = {"external_attributes":
             {"__comment": "These fields describe the forecast variable and forecast times.",
              "variables": variables,
              "datatype_resource": dataid,
              "format": "netCDF",
              "origintime": time0,
              "fctimes": list(to_datetime(fctimes).strftime(timefmt)),
              # "fctimes_end": list(to_datetime(fctimes2).strftime(timefmt)),
              }}

    #if fcfiles is not None:
    #    extra["external_attributes"].update({'fcfiles': fcfiles})
    if name is not None:
        extra["external_attributes"].update({'name': name})

    if extrameta is not None:
        extra["external_attributes"].update(extrameta)

    with open(template, "r", encoding='UTF-8') as f:
        metadata = json.load(f)
    metadata.update(names)
    metadata.update(spatial)
    metadata.update(dates)
    metadata.update(resolution)
    metadata.update(extra)
    if file is not None:
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
    return metadata


def make_metadata(origintime, fctimes, fctimes2, variable,
                  template='metadataECMFF.json',
                  fcfiles=None,
                  fctype='HRES',
                  fun=None,
                  funs=None,
                  dataids=None,
                  bb=safers_hres_domain, uid=None, file=None):
    """Generate metedata for safers upload."""
    timefmt = '%Y-%m-%dT%H:%M:%S'

    creationtime = datetime.utcnow().strftime(timefmt)
    time0 = to_datetime(origintime).strftime(timefmt)
    time1 = to_datetime(fctimes[0]).strftime(timefmt)
    time2 = to_datetime(fctimes[-1]).strftime(timefmt)

    note = 'Forecast from ECMWF processed at FMI'
    title = ''
    if fctype == 'HRES':
        title = f'Deterministic forecasts for {variable} [{safers_unit(variable)}] between {time1} and {time2}'
        note = f'Deterministic high resolution forecast from ECMWF processed at FMI for {variable_descriptions.get(variable)}.'
    else:
        title = f'Ensemble forecasts for {variable} [{safers_unit(variable)}] between {time1} and {time2}'
        note = f'Ensemble forecast from ECMWF processed at FMI for {variable_descriptions.get(variable)}.'
    if funs is not None:
        note = note + ' Includes the following ensemble statistics: '
        for i, fun in enumerate(funs):
            note = note + f'{fun} = {variable_descriptions.get(fun)}'
            if i == len(fun) - 1:
                note = note + '.'
            else:
                note = note + ', '

    if uid is None:
        uid = str(uuid4())
    # need separate tables for HRES and ENS
    if dataids is not None:
        dataid = dataids
    else:
        dataid = safers_data_id(variable)

    names = {"title": title,
             "notes": note,
             "name": uid}

    spatial = {"spatial":
               {"type": "MultiPolygon",
                "coordinates": [[[[bb[0], bb[3]],
                                  [bb[1], bb[3]],
                                  [bb[1], bb[2]],
                                  [bb[0], bb[2]],
                                  [bb[0], bb[3]]]]]}}

    dates = {"data_temporal_extent_begin_date": time1,
             "data_temporal_extent_end_date": time2,
             "temporalReference_dateOfPublication": creationtime,
             "temporalReference_dateOfLastRevision": creationtime,
             "temporalReference_dateOfCreation": creationtime,
             "temporalReference_date": creationtime}

    extra = {"external_attributes":
             {"__comment": "These fields describe the forecast variable and forecast times.",
              "variable": variable,
              "unit": safers_unit(variable),
              "datatype_resource": dataid,
              "format": "geoJSON",
              "origintime": time0,
              "fctimes": list(to_datetime(fctimes).strftime(timefmt)),
              "fctimes_end": list(to_datetime(fctimes2).strftime(timefmt))}}

    if fcfiles is not None:
        extra["external_attributes"].update({'fcfiles': fcfiles})

    with open(template, "r", encoding='UTF-8') as f:
        metadata = json.load(f)
    metadata.update(names)
    metadata.update(spatial)
    metadata.update(dates)
    metadata.update(extra)
    if file is not None:
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
    return metadata


def ensmean(ds, fun='mean', ensvar='realization', rename=False):
    """Ensemble mean or other function."""
    if fun == 'p10':
        logging.debug('calculate %s', fun)
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
    for v in list(da.data_vars):
        da[v].attrs['FMI_ens_note'] = f'This is {variable_descriptions.get(fun)}'
    if rename:
        v = list(da.data_vars)
        da = da.rename(dict(zip(v, list(map(lambda x: x + '_' + fun, v)))))
    da = da.astype(safers_datatype)
    return da


def ds_bb(ds):
    """Dataset or dataArray bounding box.

    This function assumes origin in NE corner, which means
    y direction runs from north to south.
    """
    if np.any(np.isin('lon', list(ds.dims))):
        x = 'lon'
        y = 'lat'
    else:
        x = 'longitude'
        y = 'latitude'
    bb = np.r_[ds[x].min().values, ds[x].max().values,
               ds[y].max().values, ds[y].min().values]
    return bb


# https://stackoverflow.com/questions/40766037/specify-encoding-compression-for-many-variables-in-xarray-dataset-when-write-to

def savenc(ds, file, zlib=False, discrete=False, int8=False):
    """Save to netcdf with zlib or discrete compression."""
    if zlib:
        encoding = {}
        encoding_keys = ("_FillValue", "dtype", "scale_factor",
                         "add_offset", "grid_mapping")
        for data_var in ds.data_vars:
            encoding[data_var] = {key: value for key, value in ds[data_var].encoding.items() if key in encoding_keys}
            encoding[data_var].update(zlib=True, complevel=5)
        ds.to_netcdf(file, encoding=encoding)
    elif discrete:
        encoding = {key: i16encoding for key in list(ds.data_vars)
                    if key in int16vars}
        encoding.update({key: ui16encoding for key in list(ds.data_vars)
                        if key in uint16vars})
        encoding.update({key: fwiencoding for key in list(ds.data_vars)
                        if key in fwi_vars})
        # print(encoding)
        ds.to_netcdf(file, encoding=encoding)
    elif int8:
        encoding = {key: i8encoding for key in list(ds.data_vars)}
        ds.to_netcdf(file, encoding=encoding)
    else:
        ds.to_netcdf(file)


# works for EXT and ENS files
# fc202201010000_EXT.nc
def prevfcfile(ds=None, fctype=None, f=None, ncdir=None, olderthan=24, prefix='fc'):
    """Find previous fc file of same type."""
    if ds is not None:
        origin = ds['forecast_reference_time'].values
        fdir = ncdir
        if fctype is None:
            if max(ds['leadtime'].values.astype('timedelta64[h]')) > 360:
                fctype = 'EXT'
            elif max(ds['leadtime'].values.astype('timedelta64[h]')) > 72:
                fctype = 'ENS'
            else:
                fctype = 'HRES'
                # warnings.warn(f'no file {fprev}')
                # fprev = None
                # return fprev
    else:
        fbase = os.path.basename(f)
        fdir = os.path.dirname(f)
        fctype = fbase[-6:-3]
        origin = np.datetime64(datetime.strptime(fbase[len(prefix):(len(prefix)+12)], '%Y%m%d%H%M'))
    fprev = None
    files = [os.path.basename(i) for i in glob.glob(f'{fdir}/{prefix}*_{fctype}.nc')]
    if len(files) < 1:
        warnings.warn('no previous fc data')
        return fprev
    dates = np.sort([np.datetime64(datetime.strptime(i[len(prefix):(len(prefix)+12)], '%Y%m%d%H%M')) for i in files])
    if len(dates) < 1:
        warnings.warn('no previous fc data')
        return fprev
    dates = dates[dates < origin]
    if len(dates) < 1:
        warnings.warn('no previous fc data')
        return fprev
    idate = np.argmin(np.abs(dates - origin + np.timedelta64(olderthan, 'h')))
    if idate >= 0:
        prev = to_datetime(dates[idate])
        fprev = f'{fdir}/{prefix}{prev.strftime("%Y%m%d%H%M")}_{fctype}.nc'
        if not os.path.exists(fprev):
            warnings.warn(f'no file {fprev}')
            fprev = None
    else:
        warnings.warn('no previous fc data')
    return fprev
