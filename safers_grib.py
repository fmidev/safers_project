"""
GRIB utilities for SAFERS
"""

import warnings
import numpy as np
import xarray as xr
import cf2cdm

from safers_data_tables import ds_to_grib
from safers_utils import safers_convert


def opengrib(file, type='HRES', cf=True, variable=None, keys=None,
             noindex=False, number=None,
             redim=False, convert=False, array=True, minstep=0, maxvars=[],
             avgvars=[], ntime=4, rename={}, analysis=False,
             chunks=None):
    """Read ECMWF grib file as dataset.
    This function return None is variable is not found in GRIB file
    """
    if chunks is None:
        if ntime > 0:
            chunks = {'step': ntime}
    if type.upper() == 'ENS' or type.upper() == 'EXT':
        gribargs = {'filter_by_keys': {'dataType': 'pf'}}
    else:
        gribargs = {}
    if analysis:
        # gribargs['filter_by_keys'].update({'stepRange': 0})
        gribargs = {'filter_by_keys': {'stepRange': 0}}
    variable2 = variable.copy()
    variable = ds_to_grib(variable)
    if isinstance(variable, list):
        array = False
    else:
        variable = [variable]
        variable2 = [variable2]
    ds = []
    for v, v2 in zip(variable, variable2):
        if type.upper() == 'ENS' or type.upper() == 'EXT':
            gribargs = {'filter_by_keys': {'dataType': 'pf'}}
            if number is not None:
                gribargs['filter_by_keys'].update({'number': number})
                if number == 0:
                    gribargs['filter_by_keys'].update({'dataType': 'cf'})
        else:
            gribargs = {}
            if analysis:
                # gribargs['filter_by_keys'].update({'stepRange': 0})
                gribargs = {'filter_by_keys': {'stepRange': '0'}}

        if v is not None:
            if gribargs.get('filter_by_keys') is None:
                gribargs.update({'filter_by_keys': {'shortName': v}})
            else:
                gribargs['filter_by_keys'].update({'shortName': v})
            if np.any(np.isin(v, maxvars)):
                gribargs['filter_by_keys'].update({'stepType': 'max'})
            if np.any(np.isin(v, avgvars)):
                gribargs['filter_by_keys'].update({'stepType': 'avg'})
        if noindex:
            gribargs.update({'indexpath': ''})
        if keys is not None:
            gribargs.update(keys)
        dsi = _opengrib0(file, gribargs, cf=cf, chunks=chunks)
        if dsi is not None:
            if minstep > 0:
                if cf:
                    dsi = dsi.isel(leadtime=(dsi['leadtime'] >= np.timedelta64(minstep, 'h')).values.nonzero()[0])
                else:
                    dsi = dsi.isel(step=(dsi['step'] >= np.timedelta64(minstep, 'h')).values.nonzero()[0])
            v3 = rename.get(v2)
            if v3 is not None:
                dsi = dsi.rename({v2: v3})
            ds += [dsi]

    ds = xr.merge(ds, compat='override')
    #if ds.get('leadtime') is None:  # does not work
    #    return ds
    if redim:
        ds = ds.swap_dims({'leadtime': 'time'}).rename({'lon': 'longitude', 'lat': 'latitude'})
        # ds = ds.drop_vars(['surface', 'realization'], errors='ignore')
    if convert: # no not use!?
        ds = safers_convert(ds)
    if variable[0] is not None and array is True:
        return ds[list(ds.data_vars)[0]]
    return ds


def _opengrib0(file, gribargs, cf=True, chunks=None):
    """Internal helper."""
    try:
        ds = xr.open_dataset(file, backend_kwargs=gribargs,
                             engine='cfgrib', decode_cf=cf,
                             chunks=chunks)
    except KeyError:
        warnings.warn(f'{gribargs}', UserWarning)
        warnings.warn('KeyError in grib file, returning None', UserWarning)
        ds = None
        return ds
    except ValueError as e:
        warnings.warn(f'Exception during GRIB read {e}', UserWarning)
        ds = None
        return ds
    if cf:
        ds = cf2cdm.translate_coords(ds, cf2cdm.CDS)
    return ds
