# SAFERS EU project meteorological data scripts

This repository contains scripts made at FMI for SAFERS project <https://safers-project.eu> (2021-2024). This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 869353.

The scripts load ECMWF forecast data in [grib](https://confluence.ecmwf.int/display/CKB/What+are+GRIB+files+and+how+can+I+read+them) format from a S3 bucket, convert it to netcdf using [xarray](http://xarray.pydata.org/en/stable/) and send the generated files to SAFERS datalake at  <https://datalake-test.safers-project.cloud>. In addition to raw (or slightly modified) meteorological variables, the code calculates Canadian fire weather index and related variables, see below for more info.

SAFERS wiki has information for project members: <https://wiki.safers-project.cloud/en/api-broker/wcf> (password required). At FMI the computations and development is done is virtual machine


## Fire Weather Index

This directory contains code to calculate Canadian FWI. The python code is based to Y.Wang, K.R. Anderson, and R.M. Suddaby: Updated source code for calculating fire danger indices in the Canadian Forest Fire Weather Index System, 2015. Canadian Forest Service Publications https://cfs.nrcan.gc.ca/publications?id=36461. It has been modified for vectorized calculations at FMI using Python xarray package. There are some additional code for local noon calculations, grib file processing etc.

The code takes as input a grib file or an xarray dataset that contains local noon values of variables 't2m', 'r2', 'ws', and 'tp24', for two-metre temperature (°C), relative humidity (%), 10-metre wind speed (km/h) and 24 h precipitation (mm).

FWI calculations need initial values for ffmc, dc, and dmc (Fine Fuel Moisture Code, Drought Code, and Duff Moisture Code). These can be read from a previous output file or from pre-calculated ERA5 climatology.

Functions

|                                              |                                |
|----------------------------------------------|--------------------------------|
| [`FWI_functions_xr.py`](FWI_functions_xr.py) | FWI functions                  |
| [`calculateFWI_xr.py`](calculateFWI_xr.py)   | ECMWF ens functions            |
| [`process_fwi_hres.py`](process_fwi.py)      | Process deterministic forecast |
| [`process_fwi_ens.py`](process_fwi_ens.py)   | Process ensembles              |

## Directories in `safers.fmi.fi`

|                                        |                                              |
|----------------------------------------|----------------------------------------------|
| `/data/safers/scripts/safers_scripts/` | Operational code folder                      |
| `/data/tmp/`                           | Downloaded grib files and generated nc files |
| `/data/safers/log`                     | Log files                                    |
| `/data/safers/data`                    | Extra data files, e.g. land sea masks        |
| `/data/git/safers_scripts.git/`        | Local code repository                        |

## SAFERS shell scripts

These are invoked by cron: [`crontab.in`](crontab.in). Edit `SCRIPTS_DIR` to the location of the scripts and `LOG_DIR` to log directory. The SAFERS scripts poll S3 bucket for arrival of new EC data.

|                                          |                                 |
|------------------------------------------|---------------------------------|
| [`HRES_to_SAFERS.sh`](HRES_to_SAFERS.sh) | Load and process HRES data      |
| [`ENS_to_SAFERS.sh`](ENS_to_SAFERS.sh)   | Load and process ENS data       |
| [`EXT_to_SAFERS.sh`](EXT_to_SAFERS.sh)   | Load and process EXT data       |
| [`tmpcleanup.sh`](tmpcleanup.sh)         | Clean up `/data/tmp/` directory |


## Grib to netcdf

Python scripts to process grib data and produce nc files.

|                                                              |                                                              |
|--------------------------------------------------------------|--------------------------------------------------------------|
| [`process_fc_hres.py`](process_fc_hres.py)                   | Produce NC file from HRES grib and send it to datalake       |
| [`process_fc_hres_extra.py`](process_fc_hres_extra.py)       | HRES extra variables                                         |
| [`process_fc_hres_analysis.py`](process_fc_hres_analysis.py) | HRES analysis                                                |
| [`process_fc_ens.py`](process_fc_ens.py)                     | Produce NC file from ENS or EXT grib and send it to datalake |
| [`process_fwi_hres.py`](process_fwi_hres.py)                 | FWI processing                                               |
| [`process_fwi_ens.py`](process_fwi_ens.py)                   | FWI processing                                               |
|                                                              |                                                              |

## S2 buckets

The input ECMWF forecast data is handled by STU and it is stored in the following buckets:

|                                 |                         |
|---------------------------------|-------------------------|
| `s3://safers-ecmwf/ec_hres/`    | High resolution         |
| `s3://safers-ecmwf/ec_eps_raw/` | Ensemble                |
| `s3://safers-ecmwf/ec_erf/`     | Extended range ensemble |

There is expiration rule of 3 days for files inside the buckets defined in [`ecmwf-lifecycle.xml`](ecmwf-lifecycle.xml).
