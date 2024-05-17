#!/usr/bin/env python
# coding: utf-8

# In[72]:


import xarray as xr
import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm 
import cartopy.crs as ccrs
import cartopy.feature as cpf
from pandas import to_datetime, to_timedelta

def plotmap(da,
            var,
            p0 = ccrs.PlateCarree(),
            cmap=plt.cm.RdYlBu.reversed(),
            levels=None,            
            file=None,
            title=None,
            label=None,
            under=None,
            ax=None,
            xvar=None,
            yvar=None,
            vmin=None,
            vmax=None,
            vcenter=None,
            skip=None):
    
    p_map=p0    
    p_data=p0
    
    if xvar == None:
        xvar = 'longitude'
    else:
        xvar = xvar
        
    if yvar == None:
        yvar = 'latitude'
    else:
        yvar = yvar    

    my_cmap = mpl.cm.get_cmap(cmap).copy()
    if under != None:
        my_cmap.set_under(under)

    if vcenter != None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm=None
        
    if ax == None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1, projection=p_map)        
    else:
        fig = plt.sca(ax)
        
    m = ax.pcolormesh(da[xvar].values, da[yvar].values, da[var].values,
                      vmin=vmin, vmax=vmax,norm = norm,
                      shading='auto',
                      transform=p_data,
                      cmap=my_cmap)

    if title != None:
        try:
            plt.title(f"{da[var].attrs.get('long_name')} [{da[var].attrs.get('units')}]")
        except KeyError:
            plt.title('')
            
    
    if label == None:
            label = f"{da[var].attrs.get('long_name')} [{da[var].attrs.get('units')}]"
    else:
        label=label
    cb = plt.colorbar(m, ax=ax)
    cb.set_label(label, multialignment='center')
    
    plt.margins(0)
    ax.coastlines()
    ax.add_feature(cpf.BORDERS, linestyle=':')
    gl = ax.gridlines(dms=False, draw_labels=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    if file != None:
        plt.savefig(file, bbox_inches='tight')
        
    return    
    
    
def plotPressureMap(da,
                    var,
                    p0 = ccrs.PlateCarree(),
                    cmap = None,
                    levels=10,
                    file=None,
                    title=None,
                    label=None,
                    under=None,
                    ax=None,
                    xvar=None,
                    yvar=None,
                    vmin=None,
                    vmax=None,
                    vcenter=None,
                    skip=2):
    p_map=p0    
    p_data=p0
    
    if da[var].attrs.get('units') == 'Pa':
        da = da/100
        da[var].attrs['units'] = 'hPa'   
        da[var].attrs['long_name'] = da[var].attrs.get('long_name')  
    
    bold = 1010
    
    if xvar == None:
        xvar = 'longitude'
    else:
        xvar = xvar
    if yvar == None:
        yvar = 'latitude'
    else:
        yvar = yvar    

    if ax == None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1, projection=p_map)
    else:
        fig = plt.sca(ax)
        
    if title != None:
        plt.title(title)
    else:
        plt.title(f"Mean sea level pressure [{da[var].attrs.get('units')}]")
    
    n = ax.contour(da[xvar].values, da[yvar].values, da[var].values,levels=np.array([bold]),linewidths=2,colors='k')    # increase width of 1010hPa line
    ax.clabel(n, inline=True, fontsize=10)#,colors='#FF000000')
    m = ax.contour(da[xvar].values, da[yvar].values, da[var].values,levels=levels, colors='k',linewidths=1)
    ax.clabel(m, inline=True, fontsize=10)
    
    cb = plt.colorbar(m,shrink=0)
    cb.ax.tick_params(size=0,labelsize=0,labelcolor='w')#, labelright=False) #Remove ticks
    cb.outline.set_visible(False) #Remove outline
    plt.margins(0)
    ax.add_feature(cpf.LAND)
    ax.coastlines()
    ax.add_feature(cpf.BORDERS, linestyle=':')
    gl = ax.gridlines(dms=False, draw_labels=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    if file != None:
        plt.savefig(file, bbox_inches='tight')
            
    return    
    
    
def plotWindMap(da,
                var,
                u='u10',
                v='v10',
                skip=2,
                p0 = ccrs.PlateCarree(),
                cmap=plt.cm.RdYlBu.reversed(),
                levels=None,
                file=None,
                title=None,
                label=None,
                under=None,
                ax=None,
                xvar=None,
                yvar=None,
                vmin=None,
                vmax=None,
                vcenter=None):
    
    p_map=p0    
    p_data=p0
    
    label = f" Windspeed [m/s]"

    xvar = 'longitude'
    yvar = 'latitude'

    my_cmap = mpl.cm.get_cmap(cmap).copy()
    if under != None:
        my_cmap.set_under(under)

    ds = da[dict(latitude=slice(None,None, skip), longitude=slice(None,None, skip))]
    #da = da.assign(ws= np.sqrt(np.square(da[v])+np.square(da[u]))*3.6)
    
    if ax == None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1, projection=p_map)

    else:
        fig = plt.sca(ax)
   
    m = ax.pcolormesh(da[xvar].values, da[yvar].values, da[var].values,
                      vmin=vmin, vmax=vmax,
                      shading='auto',
                      transform=p_data,
                      cmap=my_cmap)
    
    Q = ds.plot.quiver(x="longitude",y="latitude",u=u,v=v,scale=200,alpha=0.8,add_guide=False)
    qk = 10
    ax.quiverkey(Q, 1.05, 1.05, qk, str(int(qk))+' m/s', labelpos='W')
    
    if title != None:
        plt.title(f"{to_datetime(da.time.values)} to {to_timedelta(da.leadtime.values)}")
    else:
        plt.title('')
   
    plt.colorbar(m, ax=ax,label=label)
    plt.margins(0)
    ax.coastlines()
    ax.add_feature(cpf.BORDERS, linestyle=':')
    gl = ax.gridlines(dms=False, draw_labels=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    if file != None:
        plt.savefig(file, bbox_inches='tight')    
        
        
# def plot_ensemble_timeseries(ds,variable,latitude,longitude,file=None,ax=None):
#     ds = ds.sel(latitude=latitude,longitude=longitude,method='nearest') 
#     x = (ds.leadtime/3.6E12).astype('int')
    
#     if ax == None:
#         fig, ax = plt.subplots(figsize=(12,6))
#     else:
#         fig = plt.sca(ax)
    
#     ax.fill_between(x,ds[variable]+2*(ds[f'{variable}_std']+273.15),ds[variable]-2*(ds[f'{variable}_std']+273.15),color='lightgrey',alpha=0.5,
#     label='95% prediction interval')
#     ax.plot(x,ds[variable],c='k',label='member mean')
#     ax.set_title(f'latitude={latitude}°N, longitude={longitude}°E')
#     #ax.set_xticks(rotation=360)
#     ax.set_xlabel('Lead time [hours]')
#     ax.grid(color='grey', linestyle=':', linewidth=1,alpha=0.5)
#     ax.set_ylim([-5,25])
#     ax.set_xlim([0,1105]) #1105 365
#     ax.legend(loc='best')
    
#     if file != None:
#         plt.savefig(file, bbox_inches='tight')     
        

plotmapfunction = {
    't2m': {'map':plotmap,'cb':'RdYlBu_r'},
    'd2m': {'map':plotmap,'cb':'RdYlBu_r'},
    'mn2t6': {'map':plotmap,'cb':'RdYlBu_r'}, 
    'mx2t6': {'map':plotmap,'cb':'RdYlBu_r'},
    'mn2t3': {'map':plotmap,'cb':'RdYlBu_r'}, 
    'mx2t3': {'map':plotmap,'cb':'RdYlBu_r'},
    'tp': {'map':plotmap,'cb':'YlGnBu','vmin':0.01,'under':'w'},
    'r': {'map':plotmap,'cb':LinearSegmentedColormap.from_list("mycmap", ["peru", "peachpuff", "yellow", "lime","darkgreen"]),'vmin':0,'vmax':100},
    'r2': {'map':plotmap,'cb':LinearSegmentedColormap.from_list("mycmap", ["peru", "peachpuff", "yellow", "lime","darkgreen"]),'vmin':0,'vmax':100},
    'p10fg6': {'map':plotmap,'cb':'viridis_r'},
    'fg310': {'map':plotmap,'cb':'viridis_r'},
    'maxgust': {'map':plotmap,'cb':'viridis_r'},
    'u10': {'map':plotmap,'cb':'bwr','vcenter':0},
    'v10': {'map':plotmap,'cb':'bwr','vcenter':0},
    'ws': {'map':plotWindMap,'cb':'viridis_r'},
    'ssr': {'map':plotmap,'cb':'RdYlBu_r'},
    'swvl1': {'map':plotmap,'cb':'YlGn','vmin':0,'under':'w'},
    'swvl2': {'map':plotmap,'cb':'YlGn','vmin':0,'under':'w'},
    'msl': {'map':plotPressureMap,'cb':'RdYlBu_r'},
    'litota1': {'map':plotmap,'cb':'hot_r'},
    'litota3': {'map':plotmap,'cb':'hot_r'},
    'litota6': {'map':plotmap,'cb':'hot_r'},
    'cape': {'map':plotmap,'cb':'turbo','vmin':0.01,'under':'w'},
    'fwi': {'map':plotmap,'cb':'YlOrRd'},
}
   
def safers_plot(da,variable,discrete=None,ax=None,levels=None,skip=None,title=None):
    """SAFERS color maps."""
    basevar = variable.partition('_')[0]
    colormap = plt.cm.get_cmap(plotmapfunction.get(basevar, '')['cb'],discrete)
    
    variables = ['vmin','vmax','vcenter','under']

    for v in variables:
        if not str(v) in plotmapfunction.get(basevar, str(v)).keys():
            globals()[v] = None
        else:
            globals()[v] = plotmapfunction.get(basevar, '')[str(v)]  
      
    return plotmapfunction.get(basevar, '')['map'](da,basevar,cmap=colormap,ax=ax,levels=levels,skip=skip,vmin=vmin,vmax=vmax,vcenter=vcenter,under=under,title=title)    


# In[73]:


input_path = '/data/tmp/safers_nc/'
output_path = '../../example_out/deliverable_plots/'
fctype= 'EXT'

list_of_files = sorted(glob.glob(f'{input_path}*_{fctype}.nc')) 
file = max(list_of_files, key=os.path.getctime) 
fctype = os.path.basename(file).split(".")[0].split('_')[1]
ds = xr.open_dataset(file)
ds = ds.assign(ws= np.sqrt(np.square(ds['u10'])+np.square(ds['v10'])))
da = ds.isel(time=0)
print(list_of_files)

forecast_reference_time = to_datetime(da.forecast_reference_time.values)
valid_time = to_datetime(da.time.values)


# In[66]:


var_list = [e for e in list(da.keys()) if '_' not in e]
print(var_list)

cols = 2
rows = int(np.ceil(len(var_list)/cols))
gs = gridspec.GridSpec(rows, cols)
fig = plt.figure(figsize=(15,35))
fig.tight_layout()
plt.suptitle(f'{fctype}   forecast reference time: {forecast_reference_time.strftime("%Y-%m-%d %H:%M")}, valid time: {valid_time.strftime("%Y-%m-%d %H:%M")}',
             x=0.5, y=.90, horizontalalignment='center', verticalalignment='top', fontsize = 15)
for n,v in zip(range(len(var_list)),var_list):
    ax = fig.add_subplot(gs[n],projection=ccrs.PlateCarree())
    
    safers_plot(da,v,ax=ax,levels=10,skip=10)#,discrete=10)   
plt.savefig(f'{output_path}forecasts{forecast_reference_time.strftime("%Y%m%d%H%M")}_{valid_time.strftime("%Y%m%d%H%M")}_{fctype}.png')    


# In[75]:


safers_plot(da,'t2m',discrete=None,ax=None,levels=10)


# In[67]:


file = '../../example_out/Fwi/new5/Fwi_202204071200_ENS.nc'
ds = xr.open_dataset(file)
da = ds.isel(time=0)
da
# forecast_reference_time = to_datetime(da.forecast_reference_time.values).strftime("%Y-%m-%d %H:%M")
valid_time = to_datetime(da.time.values).strftime("%Y-%m-%d %H:%M")


# In[68]:


safers_plot(da,'fwi',discrete=12,ax=None,levels=10,title=True)


# In[ ]:




