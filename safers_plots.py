"""
SAFERS plotting utilities. See plotnc.py, animatenc.py and tsplot.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cpf

from pandas import to_datetime
import geopandas as gpd

from IPython.display import HTML

from safers_data_tables import safers_colormaps
from safers_data_tables import fwi_levels, fwi_levelnames

p0 = ccrs.PlateCarree()
ecglobe = ccrs.Globe(semimajor_axis=6367470,
                     semiminor_axis=6367470,
                     ellipse='sphere')
p_regular = ccrs.PlateCarree(globe=ecglobe)
p_rotated = ccrs.RotatedPole(pole_longitude=180.0,
                             pole_latitude=30,
                             central_rotated_longitude=0.0,
                             globe=ecglobe)

cmap_temp = plt.cm.RdYlBu.reversed()


def fwi_cmap(levels=fwi_levels, levelnames=fwi_levelnames):
    cmap = plt.get_cmap('YlOrRd')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    return cmap, norm


def plotmap(da, p_map=p0, p_data=p0, file=None,
            cb_shrink=0.6, ax=None, title=None,
            cmap=None, add_colorbar=True,
            vmin=None, vmax=None, show=True, norm=None,
            u=None, v=None, label=None):
    """Plot Data Array on a map."""

    if label is None:
        label = f"{da.attrs.get('long_name')} [{da.attrs.get('units')}]"

    if np.any(np.isin('lon', list(da.dims))):
        xvar = 'lon'
        yvar = 'lat'
    else:
        xvar = 'longitude'
        yvar = 'latitude'

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1, projection=p_map)
    else:
        fig = ax.get_figure()
    if cmap is None:
        if da.name is not None:
            cmap = plt.get_cmap(safers_colormaps.get(da.name.partition('_')[0],
                                                     'RdYlBu_r'))
        else:
            cmap = cmap_temp

    m = ax.pcolormesh(da[xvar].values, da[yvar].values, da.values,
                      vmin=vmin, vmax=vmax,
                      shading='auto',
                      transform=p_data,
                      cmap=cmap,
                      norm=norm)

    # add wind direction from u and v
    if u is not None and v is not None:
        nlon = len(da[xvar])
        nlat = len(da[yvar])
        dx = nlon // 20
        dy = nlat // 20
        ax.quiver(da[xvar][0::dx].values,
                  da[yvar][0::dy].values,
                  u[0::dy, 0::dx].values,
                  v[0::dy, 0::dx].values,
                  angles='xy',
                  transform=p_data)

    plt.title(title)
    if add_colorbar:
        fig.colorbar(m, ax=ax, label=label, shrink=cb_shrink)
    plt.margins(0)
    ax.coastlines()
    ax.add_feature(cpf.BORDERS, linestyle=':')
    gl = ax.gridlines(dms=False, draw_labels=True,
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    if file is not None:
        plt.savefig(file, bbox_inches='tight')
        plt.close()
    elif show:
        plt.show()
    return m


def animate(da, interval=300, cmap=None, show=True,
            vmax=None, vmin=None, repeat=False,
            dpi=100, cb_shrink=0.8, figsize=(6, 4),
            timevar='time', label=None):
    """Generate animation."""
    nt, ny, nx = da.shape
    if vmax is None:
        vmax = da.max()
    if vmin is None:
        vmin = da.min()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    fig.set_dpi(dpi)
    # fig.set_size_inches(6, 4)
    p = plotmap(da.isel({timevar: 0}), ax=ax, vmin=vmin, vmax=vmax,
                cb_shrink=cb_shrink, label=label,
                cmap=cmap, show=False)

    def animate_iter(i):
        """Iterator."""
        p.set_array(da.isel({timevar: i}).values.ravel())
        ax.set_title(to_datetime(da.isel({timevar: i})[timevar].values))
        return p,

    anim = animation.FuncAnimation(fig, animate_iter, frames=nt,
                                   interval=interval, blit=True,
                                   repeat=repeat)
    plt.close()
    if show:
        return HTML(anim.to_jshtml())

    return anim


def plotts(ds, lon, lat, variable=None, file=None, df=None,
           dx=0, dy=0, decdiff=1.0, title=None, median=False,
           show=True):
    """Plot time series at a location."""
    variables = list(ds.data_vars)
    if variable is None:
        variable = variables[0]
    da = ds[variable]  # .isel(time=time)

    xi = np.argmin(np.abs(da['longitude'].values - lon))
    yi = np.argmin(np.abs(da['latitude'].values - lat))

    if median and np.all(np.isin([variable + '_p50'], variables)):
        m = ds[variable + '_p50'].isel(longitude=xi, latitude=yi)
    else:
        # average over ±dx pixels
        dai = da.isel(longitude=slice(xi - dx, xi + dx + 1),
                      latitude=slice(yi - dy, yi + dy + 1))
        m = dai.mean(dim=['longitude', 'latitude'], skipna=True)
        m.attrs = dai.attrs

    # print(m)
    # lx = (da['longitude'][1]-da['longitude'][0]).values
    p = m.plot(linestyle='--', marker='o', markersize=3, linewidth=1)
    # plt.title(f'{lon}, {lat} (±{dx} pixels of size {lx:.2})')
    # upper and lower
    if np.all(np.isin([variable+'_p10', variable+'_p90'], variables)):
        low = ds[variable + '_p10'].isel(longitude=xi, latitude=yi)
        up = ds[variable + '_p90'].isel(longitude=xi, latitude=yi)
        plt.fill_between(ds['time'], low, up, color=np.ones(3)*0.8, alpha=0.6)

    rt = da.coords.get("forecast_reference_time")
    rt = '' if rt is None else to_datetime(rt.values)
    if title is None:
        title = f'Origin {rt}'
        title = f'{np.abs(lon)}°{"W" if lon < 0 else "E"}, {lat}°N'
    plt.title(title)
    if df is not None:
        # add observations
        inds = ((np.abs(df['latitude'] - lat) < decdiff) &
                (np.abs(df['longitude'] - lon) < decdiff))
        fdi = df[inds].set_index('time')
        if len(fdi) > 0:
            fdi['DOC'].plot(marker='o', linestyle='none',
                            markersize=4)
    plt.grid()
    if file is not None:
        plt.savefig(file, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        if show:
            plt.show()
        return p


def plotgeojson(gfile, p_map=p0, column=None, file=None,
                cb_shrink=0.85, ax=None, title=None):
    """Plot geojson file containing forecasts over a map."""
    df = gpd.read_file(gfile)

    if column is None:
        column = df.columns[0]
    label = column

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection=p_map)

    df.plot(column=column, cmap=cmap_temp, ax=ax, transform=p0, legend=True,
            legend_kwds={'label': label, 'shrink': cb_shrink})

    plt.title(title)
    plt.margins(0)
    ax.coastlines()
    ax.add_feature(cpf.BORDERS, linestyle=':')
    gl = ax.gridlines(dms=False, draw_labels=True,
                      x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    if file is not None:
        plt.savefig(file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# fix broken geoJSON, not used
def loadgeojson(f):
    import fiona
    import json
    import pandas as pd
    from shapely.geometry import shape
    with fiona.open(f, driver='GeoJSON', mode='r') as c:
        df1 = pd.DataFrame(c)

        # Check Geometry
    def isvalid(geom):
        try:
            shape(geom)
            return 1
        except Exception:
            return 0
    df1['isvalid'] = df1['geometry'].apply(lambda x: isvalid(x))
    df1 = df1[df1['isvalid'] == 1]
    collection = json.loads(df1.to_json(orient='records'))
    # Convert to geodataframe
    df = gpd.GeoDataFrame.from_features(collection)
    return df


# this plots ensemble based time series of FWI data
def plot_fwi_ts(ds, lon, lat, file=None, dpi=100):
    """Plot FWI time series."""
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    for ax, v1 in zip(axs.flat, ['fwi', 'ffmc', 'dmc', 'dc', 'bui', 'isi']):
        v2 = v1 + '_p10'
        v3 = v1 + '_p90'
        # v4 = v1 + '_p50'
        z = ds[v1].interp(latitude=lat, longitude=lon)
        low = ds[v2].interp(latitude=lat, longitude=lon)
        up = ds[v3].interp(latitude=lat, longitude=lon)
        ax.fill_between(z['time'], low, up, color=np.ones(3)*0.8, alpha=0.6)
        (ds[v1].interp(latitude=lat, longitude=lon).
            plot(label='mean', ax=ax, color='green', marker='o'))
        ax.set_title(f'{v1} ({np.abs(lon)}°{"W" if lon < 0 else "E"}, {lat}°N)')
        ax.set_ylabel('')
        ax.set_xlabel('')
    timestr = to_datetime(ds["forecast_reference_time"].values)
    plt.suptitle(f'Forecast reference time {timestr}')
    plt.tight_layout()
    if file is not None:
        plt.savefig(file, dpi=dpi)
        plt.close()
    else:
        plt.show()


# FWI anomaly at a location
def plot_fwi_anomaly(ds, lon, lat, file=None, dpi=100):
    """Plot FWI anomalies."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    for ax, v in zip(axs.flat, ['fwi', 'ffmc', 'dmc', 'dc']):
        v1 = v + '_anomaly'
        ax.fill_between(ds['time'].values, -2, 2, color='lightgreen', alpha=0.2)
        (ds[v1].interp(latitude=lat, longitude=lon).
            plot(label='anomaly', ax=ax, marker='o'))
        ax.set_title(f'{v1} ({np.abs(lon)}°{"W" if lon < 0 else "E"}, {lat}°N)')
        ax.set_ylabel('relative anomaly')
        ax.set_xlabel('')
        ax.grid()
    timestr = to_datetime(ds["forecast_reference_time"].values)
    plt.suptitle(f'FWI anomalies, forecast reference time {timestr}')
    plt.tight_layout()
    if file is not None:
        plt.savefig(file, dpi=dpi)
        plt.close()
    else:
        plt.show()
