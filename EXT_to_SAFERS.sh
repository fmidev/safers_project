#!/bin/bash
# Script to send ENS data to SAFRES datalake
# this is run from a cronjob!

## loops to wait for data to arrive
## looks only for the latest data

# need to activate python virtual environment here
source /data/python/safers/bin/activate

# PROXY settings
export HTTPS_PROXY=http://wwwproxy.fmi.fi:8080
export HTTP_PROXY=http://wwwproxy.fmi.fi:8080
export ALL_PROXY=http://wwwproxy.fmi.fi:8080
export http_proxy=http://wwwproxy.fmi.fi:8080
export https_proxy=http://wwwproxy.fmi.fi:8080

# where this script is located
PROGDIR="$(realpath "$(dirname "$BASH_SOURCE")")"
TMPDIR=/data/tmp/extgrib
LOGDIR=/data/safers/logs
NCDIR=/data/tmp/safers_nc
FWIDIR=/data/tmp/safers_fwi
FWISAVEDIR=/data/safers/fwi_ens

mkdir -p ${TMPDIR}
mkdir -p ${FWISAVEDIR}

if [ $# -lt 2 ]; then
    echo give date and hour
    exit 1
fi

UPLOAD="--upload"

DATE=`date --date="${1}" +%Y-%m-%d`
HOUR="${2}"
# sleep 5 minutes before next try
SLEEP=300

# wait for data 5 hours
MAXTRIES=`expr \( 5 \* 3600 \) / ${SLEEP}`
# echo $MAXTRIES

LOGFILE=${LOGDIR}/${DATE}T${HOUR}_EXT.log

ntries=0
while true; do
    ntries=`expr $ntries + 1`
    echo "`date`: try n:o ${ntries}"
    if [ "${ntries}" -gt "${MAXTRIES}" ]; then
	echo "`date`: No data found after ${MAXTRIES} tries"
	exit 1
    fi
    # get latest data
    LATEST=`${PROGDIR}/get_latest_fc.py ec_erf`
    if [ "$LATEST" == "${DATE}T${HOUR}" ]; then
	echo "`date`: processing for EXT ${DATE}T${HOUR} start"
	# sleep couple of seconds just in case
	sleep 10
	GRIBFILE=fc${DATE:0:4}${DATE:5:2}${DATE:8:2}${HOUR}00.grib
	# copy file
	s3cmd get --no-progress --skip-existing \
	      s3://safers-ecmwf/ec_erf/${GRIBFILE} ${TMPDIR}
	${PROGDIR}/process_fc_ens.py --variables basic ${UPLOAD} --fctype EXT \
		  --grib ${TMPDIR}/${GRIBFILE} --log ${LOGFILE} --dropuv
	${PROGDIR}/process_fc_ens.py --variables extra ${UPLOAD} --fctype EXT \
		  --grib ${TMPDIR}/${GRIBFILE} --log ${LOGFILE}
	${PROGDIR}/process_fc_ens.py --variables lightning ${UPLOAD} \
		  --fctype EXT \
		  --grib ${TMPDIR}/${GRIBFILE} --log ${LOGFILE}
	echo "`date`: processing for EXT-NC ${DATE}T${HOUR} done"
	NCFILE=fc${DATE:0:4}${DATE:5:2}${DATE:8:2}${HOUR}00_EXT.nc
	FWIFILE=Fwi_${DATE:0:4}${DATE:5:2}${DATE:8:2}${HOUR}00_EXT.nc
	# ${UPLOAD} puuttuu vielä
	${PROGDIR}/process_fwi_ens.py --summary --anomaly \
		  --fctype EXT  \
		  --grib ${TMPDIR}/${GRIBFILE} \
		  --nc ${FWIDIR}/${FWIFILE} \
		  --ncall ${FWISAVEDIR}/${FWIFILE} \
		  --log ${LOGFILE}
	echo "`date`: processing for EXT-FWI ${DATE}T${HOUR} done"
	exit 0
    fi
    sleep ${SLEEP}
done
