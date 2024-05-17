#!/bin/bash
# Script to send HRES data to SAFRES datalake
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
TMPDIR=/data/tmp/hresgrib
LOGDIR=/data/safers/logs
NCDIR=/data/tmp/safers_nc
FWIDIR=/data/tmp/safers_fwi
FWIADIR=/data/safers/fwi_analysis

mkdir -p ${TMPDIR}
mkdir -p ${FWIDIR}
mkdir -p ${FWIADIR}

if [ $# -lt 2 ]; then
    echo give date and hour
    exit 1
fi

UPLOAD="--upload"
if [ $# -gt 2 ]; then
    if [ "x${3}" == "xTEST" ]; then
	UPLOAD=""
	echo "No upload, just testing"
    fi
fi

DATE=`date --date="${1}" +%Y-%m-%d`
HOUR="${2}"
# sleep 5 minutes before next try
SLEEP=300

# wait for data 4 hours
MAXTRIES=`expr \( 4 \* 3600 \) / ${SLEEP}`
# echo $MAXTRIES

LOGFILE=${LOGDIR}/${DATE}T${HOUR}_HRES.log
GRIBFILE=fc${DATE:0:4}${DATE:5:2}${DATE:8:2}${HOUR}00.grib

ntries=0
while true; do
    if [ -f "${TMPDIR}/${GRIBFILE}" ]; then
	echo Grib file already exists, continuing
	break
    fi
    ntries=`expr $ntries + 1`
    echo "`date`: try n:o ${ntries}"
    if [ "${ntries}" -gt "${MAXTRIES}" ]; then
	echo "`date`: No data found after ${MAXTRIES} tries"
	exit 1
    fi
    # get latest data
    LATEST=`${PROGDIR}/get_latest_fc.py`
    if [ "$LATEST" == "${DATE}T${HOUR}" ]; then
	# sleep couple of seconds just in case
	sleep 10
	s3cmd get --no-progress --skip-existing \
	      s3://safers-ecmwf/ec_hres/${GRIBFILE} ${TMPDIR}
	break
    fi
    sleep ${SLEEP}
done

echo "`date`: processing for HRES ${DATE}T${HOUR} start"
${PROGDIR}/process_fc_hres.py ${UPLOAD} --discrete \
	  --grib ${TMPDIR}/${GRIBFILE} --log ${LOGFILE}
echo "`date`: processing for HRES ${DATE}T${HOUR} done"

if [ "${HOUR}" == "06" ]; then
    NCFILE=fc${DATE:0:4}${DATE:5:2}${DATE:8:2}${HOUR}00_HRES.nc
    FWIFILE=Fwi_${DATE:0:4}${DATE:5:2}${DATE:8:2}${HOUR}00_HRES.nc
    echo "`date`: processing for HRES-FWI ${DATE}T${HOUR} start"
    # no UPLOAD here ${UPLOAD} 
    ${PROGDIR}/process_fwi_hres.py --ncin ${NCDIR}/${NCFILE} \
	      --nc ${FWIDIR}/${FWIFILE} --log ${LOGFILE}
    ${PROGDIR}/fwi_analysis.py -i ${FWIDIR}/${FWIFILE} \
	      -o ${FWIADIR}/${FWIFILE} \
	      --log ${LOGFILE}
    echo "`date`: processing for HRES-FWI ${DATE}T${HOUR} done"
fi
