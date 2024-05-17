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
TMPDIR=/data/tmp/hresextragrib
LOGDIR=/data/safers/logs

mkdir -p ${TMPDIR}

if [ $# -lt 2 ]; then
    echo give date and hour
    exit 1
fi

DATE=`date --date="${1}" +%Y-%m-%d`
HOUR="${2}"
# sleep 5 minutes before next try
SLEEP=300

# wait for data 4 hours
MAXTRIES=`expr \( 4 \* 3600 \) / ${SLEEP}`
# echo $MAXTRIES

LOGFILE=${LOGDIR}/${DATE}T${HOUR}_HRESEXTRA.log
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
    LATEST=`${PROGDIR}/get_latest_fc.py ec_hres_raw`
    if [ "$LATEST" == "${DATE}T${HOUR}" ]; then
	echo "`date`: processing for ${DATE}T${HOUR} start"
	# sleep couple of seconds just in case
	sleep 5
	# process the NC file
	s3cmd get --no-progress --skip-existing \
	      s3://safers-ecmwf/ec_hres_raw/${GRIBFILE} ${TMPDIR}
	break
    fi
    sleep ${SLEEP}
done

${PROGDIR}/process_fc_hres_extra.py --discrete \
	  --grib ${TMPDIR}/${GRIBFILE} --log ${LOGFILE}
${PROGDIR}/process_fc_hres_analysis.py --discrete \
	  --grib ${TMPDIR}/${GRIBFILE} --log ${LOGFILE}
echo "`date`: processing for HRES extra ${DATE}T${HOUR} done"
