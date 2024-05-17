# Forest fire index (FWI) calculations

#
# The FWI code is originally from
#
# Y.Wang, K.R. Anderson, and R.M. Suddaby:
# Updated source code for calculating fire danger indices in
# the Canadian Forest Fire Weather Index System, 2015
#
# Canadian Forest Service Publications
# https://cfs.nrcan.gc.ca/publications?id=36461
#
# Modified for vectorized calculations at FMI
# Xarray version of the code

import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr

# original maxvalue was 10000
# now might be using 6553.5 to allow int16 compression
from safers_data_tables import fwi_minvalue, fwi_maxvalue
# from safers_utils import replace_minmax


def daylength_dmc(lat, mth):
    """
    Monthly day length adjustment factors (Le) for DMC in relation to
    reference latitudes
    reference: B.D. Lawson and O.B. Armitage, 2008: Weather guide for the
    Canadian Forest Fire Danger Rating System
    Table A3.1
    """

    # Canadian standard
    el_ge30 = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
    el_ge10_lt30 = [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8]
    # de Groot,2007
    el_ge10S_lt10 = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    el_ge30S_lt10S = [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2]
    el_lt30S = [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8]

    el = np.where(lat >= 30, el_ge30[mth-1],
                  np.where((lat >= 10) & (lat < 30),
                           el_ge10_lt30[mth-1],
                           np.where((lat >= -10) & (lat < 10),
                                    el_ge10S_lt10[mth - 1],
                                    np.where((lat >= -30) & (lat < -10),
                                             el_ge30S_lt10S[mth-1],
                                             np.where(lat < -30, el_lt30S[mth - 1],
                                                      np.nan)))))
    # el = el.astype('float32')
    return el


def daylength_dc(lat, mth):
    """
    Monthly day length adjustment factors (Lf) for DC for northern and
    southern hemispheres
    reference: B.D. Lawson and O.B. Armitage, 2008: Weather guide for the
    Canadian Forest Fire Danger Rating System
    Table A3.2
    """
    # Northern Hemisphere
    flN = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
    # around Equator between 10S and 10N
    flEq = [1.39, 1.39, 1.39, 1.39, 1.39, 1.39, 1.39, 1.39, 1.39, 1.39, 1.39, 1.39]
    # around Equator between 10S and 10N
    # flEq = [1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4]
    # Southern Hemisphere
    flS = [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8]

    fl = np.where(lat >= 10, flN[mth-1],
                  np.where((lat >= -10) & (lat < 10), flEq[mth-1],
                           np.where(lat < -10, flS[mth-1], np.nan)))

    # fl = fl.astype('float32')
    return fl


def add_daylength(ds, mth):
    """Add daylength to dataset for given month"""
    lat = ds['latitude']
    dmc = xr.DataArray(coords={'latitude': lat}, data=daylength_dmc(lat, mth))
    dc = xr.DataArray(coords={'latitude': lat}, data=daylength_dc(lat, mth))
    ds['daylength_dmc'] = dmc.broadcast_like(ds)
    ds['daylength_dc'] = dc.broadcast_like(ds)
    return ds


# def vFFMCcalc(temp, rhum, wind, prcp, ffmc0):
def vFFMCcalc(ds, ffmc0):
    """
    Fine Fuel Moisture Code (FFMC),
    represents the moisture content of litter and other
    cured fine fuels in a forest stand,
    in a layer of dry weight about 0.25 kg/m2;
    Parameters:
    temp(float): Local noon temperature (°C)
    rhum(float): Local noon relative humidity (%)
    wind(float): local noon wind speed(km/h)
    prcp(float): 24h precipitation (mm)
    ffmc0(float): previous days ffmc ()
    """

    temp = ds['t2m']
    rhum = np.maximum(0.0, ds['r2'])
    wind = np.maximum(0.0, ds['ws'])
    prcp = np.maximum(0.0, ds['tp24'])

    mo = (147.2*(101.0 - ffmc0)) / (np.maximum((59.5 + ffmc0), 1.0E-8))

    ed = (0.942*(rhum**.679) + (11.0*np.exp((rhum-100.0)/10.0)) +
          0.18*(21.1-temp) * (1.0 - 1.0/np.exp(0.1150 * rhum)))

    ew = (0.618*(rhum**.753) + (10.0*np.exp((rhum-100.0)/10.0)) +
          0.18*(21.1-temp) * (1.0 - 1.0/np.exp(0.115 * rhum)))

    mo = xr.where(prcp > 0.5,
                  xr.where(mo > 150.0,
                           (mo + 42.5*(prcp - 0.5) *
                            np.exp(-100.0/(251.0-mo)) *
                            (1.0 - np.exp(-6.93/(prcp - 0.5))) +
                            (0.0015 * (mo - 150.0)**2) *
                            np.sqrt((prcp - 0.5))),
                           (mo + 42.5*(prcp - 0.5) *
                            np.exp(-100.0/(251.0-mo)) *
                            (1.0 - np.exp(-6.93/(prcp - 0.5))))),
                  mo)
    mo = np.minimum(250.0, mo)

    m = xr.where(mo < ed,
                 xr.where(mo <= ew,
                          (ew - (ew - mo) /
                           10.0**((0.424*(1.0-((100.0-rhum)/100.0)**1.7) +
                                   (0.0694*np.sqrt(wind)) *
                                   (1.0 - ((100.0 - rhum)/100.0)**8)) *
                                  (0.581 * np.exp(0.0365 * temp)))),
                          mo),
                 xr.where(mo == ed,
                          mo,
                          (ed + (mo-ed)/10.0**((0.424*(1.0-(rhum/100.0)**1.7) +
                                                (0.0694*np.sqrt(wind)) *
                                                (1.0-(rhum/100.0)**8)) *
                                               (0.581*np.exp(0.0365*temp))))))

    ffmc = (59.5 * (250.0 - m)) / (147.2 + m)  # *Eq. 10*#
    ffmc = np.minimum(101.0, ffmc)
    ffmc = np.maximum(0.0, ffmc)
    return ffmc


# def vDMCcalc(temp, rhum, prcp, dmc0, mth, lat):
def vDMCcalc(ds, dmc0):
    """
    Duff Moisture Code (DMC),
    represents the moisture content of loosely compacted,
    decomposing organic matter weighing about 5 kg/m2 when dry
    note: checked against documentation by Dowdy,2009
    """

    temp = ds['t2m']
    rhum = np.maximum(0.0, ds['r2'])
    prcp = np.maximum(0.0, ds['tp24'])
    elm = ds['daylength_dmc']

    # effective rainfall
    re = xr.where(prcp > 1.5,
                  0.92 * prcp - 1.27,
                  0.0)

    # moisture content of the duff layer, initial uses dmc0
    m0 = 20.0 + np.exp(5.6348 - dmc0 / 43.43)

    b = xr.where(dmc0 <= 33,
                 100/(0.5 + 0.3*dmc0),
                 xr.where((dmc0 > 33) & (dmc0 <= 65),
                          14 - 1.3*np.log(dmc0),
                          6.2*np.log(dmc0) - 17.2))

    # initial moisture content modified by effective rainfall
    mr = m0 + ((1000*re)/(48.77 + b*re))
    # dmc modified by rainfall
    dmcr = 244.72 - 43.43*np.log(mr - 20)
    dmcr = np.maximum(dmcr, 0)  # check that it's not negative

    # approximation of evaporation from duff layer for temp > -1.1
    dmcd = xr.where(temp > -1.1,
                    (1.894 * (temp + 1.1) * (100.0 - rhum) * elm * 1E-4),
                    0.0)

    dmc = dmcr + dmcd
    # limit dmc to 10000 (or to fwi_maxvalue)
    dmc = np.minimum(np.maximum(dmc, 0.0), fwi_maxvalue)
    return dmc


# def vDCcalc(temp, prcp, dc0, mth, lat):
# def vDCcalc(temp, prcp, dc0, flm):
def vDCcalc(ds, dc0):
    """
    Drought Code (DC),
    represents a deep layer of compact organic matter
    weight perhaps 25 kg/m2 when dry
    note: checked against documentation by Dowdy,2009
    """

    temp = ds['t2m']
    prcp = np.maximum(0.0, ds['tp24'])
    flm = ds['daylength_dc']

    # effective rainfall, first 2.8mm daily rainfall is assumed to be lost
    # (canopy interception, surface runoff, upper duff layer)
    re = xr.where(prcp > 2.8,
                  0.83 * prcp - 1.27,
                  0.0)
    # moistue equivalent scale
    q0 = 800 * np.exp(-dc0/400)
    # rainfall modified moisture equivalent scale
    qr = q0 + 3.937 * re
    # rainfall modified dc
    dcr = 400 * np.log(800/qr)

    # seasonal day length adjustment
    # time, lat, lon
    # flm = np.array(np.broadcast_to(daylength_dc(lat, mth), temp.shape))

    # moisture loss/evaporation from the deep duff layer
    dcd = xr.where(temp > -2.8,
                   0.5 * (0.36 * (temp + 2.8) + flm),
                   0.0)

    dc = np.minimum(np.maximum(dcr + dcd, 0.0), fwi_maxvalue)

    return dc


# def vISIcalc(wind, ffmc):
def vISIcalc(ds, ffmc):
    """
    Initial spread index (ISI),
    a combination of wind and the FFMC (fine fuel moisture code)
    that represents the rate of spread alone without the influence
    of variable quantities of fuel
    """
    wind = np.maximum(0.0, ds['ws'])
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 91.9 * np.exp(-0.1386 * m) * (1.0 + m**5.31 / 49300000.0)
    isi = 0.208 * np.exp(0.05039 * wind) * ff

    isi = np.minimum(np.maximum(isi, fwi_minvalue), fwi_maxvalue)

    return isi


def vBUIcalc(dmc, dc):
    """
    Buildup index (BUI),
    a combination of DMC and DC that represents the total fuel
    available to spreading fire
    """
    bui = xr.where(dmc <= 0.4*dc,
                   (0.8 * dc * dmc) / (dmc + 0.4 * dc),
                   (dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) *
                    (0.92 + (0.0114 * dmc)**1.7)))
    bui = bui.fillna(0.0)
    # bui = np.maximum(0.0, bui)
    bui = np.minimum(np.maximum(bui, fwi_minvalue), fwi_maxvalue)
    return bui


def vFWIcalc(isi, bui):
    """
    Fire weather index (FWI),
    a combination of the ISI and the BUI that represents the intensity of
    the spreading fire as energy output rate per unit lenght of fire front
    """
    bb = xr.where(bui <= 80.0,
                  0.1 * isi * (0.626 * bui**0.809 + 2.0),
                  0.1 * isi * (1000.0 / (25.0 + 108.64 / np.exp(0.023 * bui)))
                  )
    fwi = xr.where(bb <= 1.0,
                   bb,
                   np.exp(2.72 * (0.434 * np.log(bb))**0.647)
                   )
    fwi = np.minimum(np.maximum(fwi, fwi_minvalue), fwi_maxvalue)
    return fwi


""" other utility functions for FWI calculation"""


def solar_noon(dtime, lon):
    """
    calculate the solar noon for all locations
    according to NOAA Global Monitoring Division
    (https://gml.noaa.gov/grad/solcalc/solareqns.PDF)
    return: solar noon in minutes
    """
    doy = dtime.timetuple().tm_yday  # day of year
    hour = dtime.hour

    # fractional year in rad
    f = (2 * np.pi) / 365 * (doy - 1 + (hour - 12) / 24)
    # equation of time in minutes
    eqtime = 229.18*(0.000075 + 0.001868*np.cos(f) - 0.032077*np.sin(f) -
                     0.014615*np.cos(2*f) - 0.040849*np.sin(2*f))
    # solar noon
    solnoon = 720 - 4 * lon - eqtime
    snoon = dt.datetime.combine(dtime.date(), dt.time(int(solnoon/60),
                                                      int((solnoon) % 60),
                                                      int((solnoon*60) % 60)))
    return snoon


solar_noon_vect = np.vectorize(solar_noon)


def closest_dates(dates, pivot, n):
    return sorted((d for d in dates if d < pivot),
                  key=lambda t: abs(t - pivot))[:n]


def modified_ordinal_day(ds_time):
    """ ds_time: datetime coordinate
        return: ordinal day, considering leap years
    (same date has always same number)
    NOT equal to dayofyear
    """
    # if isinstance(ds_time,xr.core.dataarray.DataArray) == True:
    #     not_leap_year = xr.DataArray(~pd.DatetimeIndex(ds_time).is_leap_year,coords=ds_time.coords)
    # else:
    not_leap_year = ~pd.DatetimeIndex(ds_time).is_leap_year
    march_or_later = ds_time.dt.month >= 3
    ordinal_day = ds_time.dt.dayofyear
    modified_ordinal_day = ordinal_day + (not_leap_year & march_or_later)
    modified_ordinal_day = modified_ordinal_day.rename('mod_ordinal_day')
    return modified_ordinal_day


def interpolate_grid(ds1, ds2, output=None):
    """
    ds1: dataset to be interpolated
    ds2: dataset to interpolate grid to
    returns ds1 interpolated onto the grid of ds2
    ds1 and ds3 can only have lat,lon and time dimension
    """
    if np.any(np.isin('longitude', list(ds1.dims), invert=True)):
        ds1 = ds1.rename({'lon': 'longitude', 'lat': 'latitude'})

    time_var = list(set(list(ds1.dims)) - set(['latitude', 'longitude']))[0]
    ds1 = ds1.rename({time_var: 'times'})

    time_var2 = list(set(list(ds2.dims)) - set(['latitude', 'longitude']))[0]
    ds2 = ds2.rename({time_var2: 'times'})

    days = ds1.times.values.astype('datetime64[h]').tolist()
    dataset = []
    for i in range(len(ds1.times)):
        ds1_interp = ds1[dict(times=i)].interp_like(ds2.isel(times=0),
                                                    method='linear')
        dataset.append(ds1_interp)

    dst_final = xr.concat(dataset, 'times')
    dst_final = dst_final.rename({'times': time_var})

    if output is not None:
        dst_final.to_netcdf(output)

    return dst_final
