import xskillscore
import xarray as xr
import glob
import os
import logging
from datetime import datetime, timedelta

logging.getLogger().setLevel(logging.INFO)

def parsefile_FWI(f):
    """Parse forecast filename to get fc times."""
    f1 = os.path.basename(f)
    f0 = os.path.splitext(f1)[0].split('_')[1]
    time = datetime.strptime(f0[-12:], '%Y%m%d%H%M')
    return([f, f1, time])

def calc_crps(fwi,obs):
    """ 
    obs: dataset of last 15 observations in the directory
    fwi: file path of ENS forecast 15 days ago 
    """
    ens_fwi = xr.open_dataset(fwi)    
    ens_fwi = ens_fwi.rename({'realization':'member'}).drop_vars(['surface','quantile'])
    # matching hres analysis and ens forecast with lat/lon and dates
    obs = obs.reindex_like(ens_fwi.isel(time=0,member=1),method='nearest').drop_vars(['forecast_reference_time','leadtime','ordinal_day'])
    dst_obs = obs.where(obs.time.isin(ens_fwi.time), drop=True)
    dst_ens =ens_fwi.sel(time=dst_obs.time)
    # calculate crps and save
    crps= xskillscore.crps_ensemble(dst_obs, dst_ens, dim=[])
    return crps

def filesBetweenDates(start_date,end_date,file_path,fileending=None,filebeginning=None):
    all_files = [datetime.strptime(x.split('_')[1], "%Y%m%d%H%M")for x in os.listdir(file_path) if x.endswith(fileending)]
    correct_date_files = [os.path.join(file_path, filebeginning + x.strftime('%Y%m%d%H%M') + fileending) for x in all_files 
                            if x >=  start_date and x <= end_date]
    return correct_date_files


def crps_calculation(file,analysis_dir='/data/safers/fwi_analysis/',ens_dir='/data/safers/fwi_ens/',crps_outdir='/data/safers/fwi_crps',ndays=15):
    """
    collects the correct files and saves the crps 
    """
    fref_new = parsefile_FWI(file)[2]
    start = fref_new - timedelta(days=ndays)
    end = fref_new
    obs = xr.open_mfdataset(filesBetweenDates(start,end,analysis_dir,fileending='_HRES.nc',filebeginning='Fwi_'))
    # -> open ENS file that is x (15) days old, do matching and calculate crps, save crps in another folder
    fwi = sorted(glob.glob(f'{ens_dir}*{(fref_new - timedelta(days=ndays)).strftime("%Y%m%d")}*.nc'))[0]
    logging.info(f'Open {fwi}')
    crps = calc_crps(fwi,obs)
    crps.to_netcdf(f'{crps_outdir}/crps_{parsefile_FWI(fwi)[2].strftime("%Y%m%d%H%M")}.nc')
    logging.info(f'Saved to {crps_outdir}/crps_{parsefile_FWI(fwi)[2].strftime("%Y%m%d%H%M")}')  


if __name__ == "__main__":
    analysis_dir = '/data/safers/fwi_analysis/'
    ens_dir = '/data/safers/fwi_ens/'
    crps_outdir = '/data/safers/fwi_crps'
    ndays = 15

    newest_ana_file = '/data/safers/fwi_analysis/Fwi_202211220600_HRES.nc'

    crps_calculation(newest_ana_file,analysis_dir,ens_dir,crps_outdir,ndays)