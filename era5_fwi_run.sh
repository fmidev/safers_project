#!/bin/sh

# process era5 data:
# pick variables needed for FWI calculations
# crop EU region and save as netcdf
# intended to be run in eslogin/voima with ERA5 data

module load cray-python

YEARS=$(seq 1979 1999)
# YEARS=$(seq 2000 2019)

for i in ${YEARS} ; do
    d=/lustre/tmp/weto/era5/$i
    # d=/lustre/tmp/weto/era5/postprocessed/$i
    for f in $d/*.grib ; do
	if [ ! -z $(echo $f | grep pressure ) ] ; then
	    # if [ -z $(echo $f | grep surface ) ] ; then
	    continue
	fi
	f2=$(basename $f)
	nc="$(basename $f .grib).nc"
	echo Processing $f2
	nice grib_copy -w shortName=2t/2d/10u/10v/tp $f out/$f2
	nice ./era5_fwi_data.py out/$f2 out/$nc
	rm -f out/$f2*
    done
done
echo done
