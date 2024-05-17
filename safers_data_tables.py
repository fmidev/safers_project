"""SAFERS data tables and utility functions."""

from datetime import datetime

import numpy as np

from cfgrib import __version__ as cfgrib_version
from eccodes import __version__ as eccodes_version
from xarray import __version__ as xarray_version

safers_code_version = '1.1'

# EC HRES leadtimes (NOT valid with new HRES data)
# HRES_leadtimes = list(list(range(0, 145, 3)) + list(range(150, 241, 6)))

# ECMWF HRES leadtimes
HRES_leadtimes = np.r_[np.arange(0, 90 + 1, 1),
                       np.arange(93, 145 + 1, 3),
                       np.arange(150, 360 + 1, 6)]

HRES_leadtimes_ns = HRES_leadtimes.astype('timedelta64[h]').astype('timedelta64[ns]')

ENS_leadtimes = np.r_[np.arange(0, 145 + 1, 3),
                      np.arange(150, 360 + 1, 6)]

ENS_leadtimes_ns = ENS_leadtimes.astype('timedelta64[h]').astype('timedelta64[ns]')

# NOW west -> east, north -> south !
# safers_ireact_domain = [-26.0, 42.4, 73.4, 32.6]  # i-react grid
# safers_hres_domain = [-40.0, 72.5, 73.5, 27.5]  # HRES EU domain in MOS

safers_ens_domain = [-25.0, 39.8, 72.0, 25.6]
safers_hres_domain = [-25.0, 40.0, 72.0, 25.5]  # HRES EU domain
safers_ext_domain = [-25.0, 39.8, 72.0, 25.6]  # EXT domain

# data resolutions
safers_hres_res = 0.1
safers_ens_res = 0.2
safers_ext_res = 0.4

# datatype for safers data in saved netcdf files
safers_datatype = np.float32

# nc attributes
i16encoding = {'dtype': 'int16',
               'scale_factor': 0.01,
               'add_offset': 0.0,
               '_FillValue': -9999}
#               '_FillValue': -327.68}

# these variables can be saved with i16 encoding and two decimals
# i.e. they will be in the range -327.68 to 327.67
int16vars0 = ['t2m', 'r2', 'd2m', 'u10', 'v10', 'ws10',
              'p10fg6', 'mn2t6', 'mx2t6',
              'mn2', 'mx2', 'fg10']

int16vars = (int16vars0 +
             [v + '_p10' for v in int16vars0] +
             [v + '_p50' for v in int16vars0] +
             [v + '_p90' for v in int16vars0])

# int16_minvalue = -327.68
int16_minvalue = -327.67
int16_maxvalue = 327.67


# These variables can be in range from 0.0 to 655.35
# Two digits precision
uint16vars = [v + '_std' for v in int16vars0] + ['dwi10']

ui16encoding = {'dtype': 'int16',
                'scale_factor': 0.01,
                'add_offset': 327.68,
                '_FillValue': -9999}
#                '_FillValue': 655.35}

# uint16_maxvalue = 655.35
uint16_maxvalue = 655.34

# used for percentages in threshold data
i8encoding = {'dtype': 'int8',
              'scale_factor': 1.0,
              'add_offset': 0.0,
              'zlib': False,
              '_FillValue': -99
              }

# stores values from 0.0 to 6553.5
# one decimal precision
# used to save FWI variables
fwiencoding = {'dtype': 'int16',
               'scale_factor': 0.1,
               'add_offset': 3276.8,
               '_FillValue': -9999}
#               '_FillValue': 6553.5}

fwi_minvalue = 0.0
# fwi_maxvalue = 6553.5
fwi_maxvalue = 6553.4

# these can be saved with fwiencoding
fwi_vars = ['fwi', 'isi', 'bui', 'dc', 'dmc', 'ffmc']
fwi_vars += ([v + '_p10' for v in fwi_vars] +
             [v + '_p90' for v in fwi_vars] +
             [v + '_p50' for v in fwi_vars] +
             [v + '_std' for v in fwi_vars])

# fwi anomaly vars using int1
int16vars += [v + '_anomaly' for v in fwi_vars]

# table to convert ecmwf variables to CF standard
# some discussion here:
# https://www.ecmwf.int/sites/default/files/elibrary/2014/13704-parameter-naming-grib-and-cf.pdf
ecmwf_to_cf_variable = {
    't2m': 'air_temperature',
    'd2m': 'dewpoint_temperature',
    'si10': '10m_wind_speed',
    'dwi10': '10m_wind_direction',
    'tp': 'total_precipitation',
    'r': 'relative_humidity',
    'r2': 'relative_humidity',
    'u10': '10m_wind_u',
    'v10': '10m_wind_v',
    'msl': 'air_pressure_at_mean_sea_level',
}

# default colormaps
# for example:
# cmap=plt.get_cmap(safers_colormaps.get(name, 'RdYlBu_r'))
safers_colormaps = {
    't2m': 'RdYlBu_r',
    'd2m': 'RdYlBu_r',
    'si10': 'PuBu',
    'dwi10': 'PuBu',
    'tp': 'PuBu',
    'tp24': 'PuBu',
    'r': 'PuBu',
    'r2': 'PuBu',
    'u10': 'PuBu',
    'v10': 'PuBu',
    'msl': 'RdYlBu_r',
    'cape': 'inferno_r',
    'litota': 'inferno_r',
    'litota1': 'inferno_r',
    'litota3': 'inferno_r',
    'litota6': 'inferno_r',
}

fwi_levels = [0, 5.2, 11.2, 21.3, 38, 50, 70, 9999]
fwi_levelnames = ['very low', 'low', 'moderate', 'high',
                  'very high', 'extreme', 'very extreme']

# convert some grib names to DataSet names
ds_to_grib_table = {
    't2m': '2t',
    'd2m': '2d',
    'r2': '2r',
    'r2m': '2r',
    'u10': '10u',
    'v10': '10v',
    'si10': '10si',
    'p10fg6': '10fg6',
    'fg310': '10fg3',
}

# Descriptions for variables and summaries
variable_descriptions = {
    't2m': 'air temperature at 2 meters',
    'd2m': 'dew point temperature at 2 meters',
    'tp': 'total precipitation',
    'tp24': 'total precipitation in the last 24 hours',
    'mn2t6': 'minimum temperature at 2 metres in the last 6 hours',
    'mx2t6': 'maximum temperature at 2 metres in the last 6 hours',
    'mn2t3': 'minimum temperature at 2 metres in the last 3 hours',
    'mx2t3': 'maximum temperature at 2 metres in the last 3 hours',
    'mn2': 'minimum temperature at 2 metres since last forecast time',
    'mx2': 'maximum temperature at 2 metres since last forecast time',
    'r': 'relative humidity',
    'r2': '2 metre relative humidity',
    'maxgust': 'maximum wind speed',
    'u10': '10 metre U wind component',
    'v10': '10 metre V wind component',
    'si10': '10 metre wind speed',
    'ws10': '10 metre wind speed',
    'dwi10': '10 metre wind direction',
    'ssr': 'surface net solar radiation',
    'swvl1': 'volumetric soil water layer 1',
    'swvl2': 'volumetric soil water layer 2',
    'msl': 'mean sea level pressure',
    'p10fg6': '10 metre wind gust in the last 6 hours',
    'p10fg3': '10 metre wind gust in the last 3 hours',
    'fg10': '10 metre wind gust since the last forecast',
    'hres': 'deterministic forecast',
    'none': 'deterministic forecast',
    'ens': 'ensemble forecast',
    'ext': 'extended range ensemble forecast',
    'mean': 'mean of ensemble forecast',
    'std': 'standard deviation of ensemble forecast',
    'cstd': 'circular standard deviation of wind direction ensemble forecast',
    'p10': 'lower limit of 80% probability forecast',
    'p90': 'upper limit of 80% probability forecast',
    'p50': 'median of ensemble forecast',
    'anomaly': 'relative anomaly against historical distribution',
    'lightning': 'variables related to lightning',
    'litota': 'averaged total lightning flash density since the last forecast',
    'litota1': 'averaged total lightning flash density in the last hour',
    'litota3': 'averaged total lightning flash density in the last 3 hours',
    'litota6': 'averaged total lightning flash density in the last 6 hours',
    'cape': 'convective available potential energy',
    'fwi': 'Fire Weather Index',
    'ffmc': 'Fine Fuel Moisture Code',
    'dc': 'Drought Code',
    'dmc': 'Duff Moisture Code',
    'isi': 'Initial Spread Index',
    'bui': 'Buildup Index',
    'ffmc0': 'Initial Fine Fuel Moisture Code',
    'dc0': 'Initial Drought Code',
    'dmc0': 'Initial Duff Moisture Code',
    'latitude': 'latitude',
    'longitude': 'longitude',
    'time': 'forecast validity time',
    'leadtime': 'leadtime',
    'prob': 'threshold exceedance probability [%]',
    'prob*': 'threshold exceedance probability [%]',
}

isobar_interval = {
    't2m': 1,
    'd2m': 1,
    'mn2t6': 1,
    'mx2t6': 1,
    'r': 5,
    'r2': 5,
    'si10': 2,
    'maxgust': 2,
    'dwi10': 45,
    'u10': 1,
    'v10': 1,
    'ssr': 'auto',
    'tp': 'auto',
}

isobar_interval_pct = 5.0

# these are now the converted units
# see safers_utils.safers_conversion
variable_units = {
    't2m': '°C',
    'd2m': '°C',
    'mn2t6': '°C',
    'mx2t6': '°C',
    'mn2t3': '°C',
    'mx2t3': '°C',
    'mn2': '°C',
    'mx2': '°C',
    'tp': 'mm',
    'tp24': 'mm',
    'r': '%',
    'r2': '%',
    'maxgust': 'm/s',
    'p10fg6': 'm/s',
    'p10fg3': 'm/s',
    'fg10': 'm/s',
    'u10': 'm/s',
    'v10': 'm/s',
    'si10': 'm/s',
    'ws10': 'm/s',
    'dwi10': 'degree',
    'ssr': 'W/m**2',
    'swvl1': 'm**3/m**3',
    'swvl2': 'm**3/m**3',
    'msl': 'hPa',
    'litota': '1/km**2/day',
    'litota1': '1/km**2/day',
    'litota3': '1/km**2/day',
    'litota6': '1/km**2/day',
    'cape': 'J/kg',
    'ffmc': 'Numeric',
    'dc': 'Numeric',
    'dmc': 'Numeric',
    'isi': 'Numeric',
    'bui': 'Numeric',
    'fwi': 'Numeric',
    'ffmc0': 'Numeric',
    'dc0': 'Numeric',
    'dmc0': 'Numeric',
    'latitude': 'degrees_north',
    'longitude': 'degrees_east',
}


# table of SAFERS data ids, always 5 digits
safers_data_ids = {
    'hres': 31001,
    'hres_basic': 31001,
    'hres_all': 31001,
    'ens': 31002,
    'ens_basic': 31002,
    'ens_all': 31002,
    'ext': 31003,
    'ext_basic': 31003,
    'ext_all': 31003,
    'fwi_ens': 31005,  # new numbering
    'fwi_ext': 31006,
    'era5': 31504,
    'fwi': 31505,
    'fwi_hres': 31505,
    'era5fwi': 31506,
    'hres_extra': 31007,
    'ens_extra': 31008,
    'ext_extra': 31009,
    'ens_lightning': 31010,
    'ext_lightning': 31011,
    'threshold': 31012,
    'fwi_climatology': 31013,
    'fwi_climatology_std': 31014,
    't2m': 31101,
    'si10': 31102,
    'tp': 31103,
    'r': 31104,
    'r2': 31104,
    'd2m': 31105,
    'dwi10': 31106,
    'u10': 31107,
    'v10': 31108,
    'maxgust': 31109,
    'ssr': 31110,
    'litota': 31111,
    't2m_mean': 31201,
    't2m_std': 31202,
    't2m_p10': 31203,
    't2m_p20': 31204,
    'tp_mean': 31205,
    'tp_std': 31206,
    'tp_p10': 31207,
    'tp_p90': 31208,
    'd2m_mean': 31209,
    'd2m_std': 31210,
    'd2m_p10': 31211,
    'd2m_p80': 31212,
    'u10_mean': 31213,
    'u10_std': 31214,
    'u10_p10': 31215,
    'u10_p80': 31216,
    'v10_mean': 31217,
    'v10_std': 31218,
    'v10_p10': 31219,
    'v10_p80': 31220,
    'maxgust_mean': 31221,
    'maxgust_std': 31222,
    'maxgust_p10': 31223,
    'maxgust_p80': 31224,
    'ssr_mean': 31225,
    'ssr_std': 31226,
    'ssr_p10': 31227,
    'ssr_p80': 31228,
    'mn2t6_mean': 31229,
    'mn2t6_std': 31230,
    'mn2t6_p10': 31231,
    'mn2t6_p80': 31232,
    'mx2t6_mean': 31233,
    'mx2t6_std': 31234,
    'mx2t6_p10': 31235,
    'mx2t6_p20': 31236,
    'unknown': 99999,
}


safers_dataset_attrs = {
    'institution': 'European Centre for Medium-Range Weather Forecasts',
    'institution2': 'Finnish Meteorological Institute',
    'project': 'SAFERS EU Horizon',
    'contact': 'marko.laine@fmi.fi',
    'history': (f'ECMWF forecasts processed at FMI {datetime.utcnow().strftime("%Y-%m-%dT%H:%M")}'
                f' using SAFERS code version {safers_code_version} and cfgrib-{cfgrib_version}/ecCodes-{eccodes_version}'
                f'/xarray-{xarray_version}'
                ),
}

safers_variable_attrs = {
    'projection': 'lonlat',
    'grid_type': 'regular_ll',
    'crs': 'epsg4326',
}

# drop these attributes
safers_drop_keys = ['standard_name', 'GRIB_', 'Conventions']

def safers_attrs(ds):
    """Apply safers attributes to dataset."""
    attrs = ds.attrs
    newattrs = {k: attrs[k] for k in attrs.keys()
                if all([k.find(k2) < 0 for k2 in safers_drop_keys])}
    newattrs.update(safers_dataset_attrs)
    ds.attrs = newattrs
    for v in ds.data_vars:
        attrs2 = ds[v].attrs
        newattrs = {k: attrs2[k] for k in attrs2.keys()
                    if all([k.find(k2) < 0 for k2 in safers_drop_keys])}
        newattrs.update(safers_variable_attrs)
        basevar = v.partition('_')[0]
        fun = v.partition('_')[2]
        attrs3 = {
            'long_name': variable_descriptions.get(basevar, 'unknown'),
            'units': safers_unit(v),
        }
        newattrs.update(attrs3)
        ds[v].attrs = newattrs
    return ds


def ds_to_grib(variable, default=None):
    """Return grib name of the variable."""
    if variable is None:
        return None
    if isinstance(variable, list):
        vout = variable.copy()
        for i in range(len(vout)):
            if isinstance(default, list):
                d = default[i]
            else:
                d = vout[i]
            vout[i] = ds_to_grib_table.get(vout[i], d)
    else:
        if default is None:
            default = variable
        vout = ds_to_grib_table.get(variable, default)
    return vout


def safers_data_id(variable, fun=None):
    """SAFERS data variable id."""
    unknown = safers_data_ids.get('unknown', 99999)
    if fun is None:
        varid = safers_data_ids.get(variable.lower(), unknown)
    else:
        varid = safers_data_ids.get(f'{variable}_{fun}'.lower(), unknown)
    return varid


def safers_unit(variable):
    """SAFERS data unit."""
    basevar = variable.partition('_')[0]
    return variable_units.get(basevar, '')


def safers_isobar_interval(variable):
    """SAFERS default isobar intervals."""
    basevar = variable.partition('_')[0]
    return isobar_interval.get(basevar, 'auto')


def ecmwftocf(variable):
    """Convert ECMWF variable name to CF standard."""
    cfname = ecmwf_to_cf_variable.get(variable, variable)
    return cfname
