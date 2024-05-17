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
TMPDIR=/data/tmp/ensgrib
LOGDIR=/data/safers/logs
NCDIR=/data/tmp/safers_nc
FWIDIR=/data/tmp/safers_fwi
FWISAVEDIR=/data/safers/fwi_ens
FWIADIR=/data/safers/fwi_analysis

mkdir -p ${TMPDIR}
mkdir -p ${FWISAVEDIR}
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

LOGFILE=${LOGDIR}/${DATE}T${HOUR}_ENS.log
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
    LATEST=`${PROGDIR}/get_latest_fc.py ec_eps_raw`
    if [ "$LATEST" == "${DATE}T${HOUR}" ]; then
	# sleep couple of seconds just in case
	sleep 10
	s3cmd get --no-progress --skip-existing \
	      s3://safers-ecmwf/ec_eps_raw/${GRIBFILE} ${TMPDIR}
	break
    fi
    sleep ${SLEEP}
done

echo "`date`: processing for ENS ${DATE}T${HOUR} start"

# processing in two (or three) parts, basic and extra
echo "`date`: processing for ENS-basic ${DATE}T${HOUR}"
${PROGDIR}/process_fc_ens.py --variables basic ${UPLOAD} --discrete \
	  --fctype ENS --grib ${TMPDIR}/${GRIBFILE} --log ${LOGFILE} \
	  --dropuv

echo "`date`: processing for ENS-extra ${DATE}T${HOUR}"
${PROGDIR}/process_fc_ens.py --variables extra ${UPLOAD} --discrete \
	  --fctype ENS --grib ${TMPDIR}/${GRIBFILE} --log ${LOGFILE}

echo "`date`: processing for ENS-lightning ${DATE}T${HOUR}"
${PROGDIR}/process_fc_ens.py --variables lightning ${UPLOAD} --discrete \
	  --fctype ENS --grib ${TMPDIR}/${GRIBFILE} --log ${LOGFILE}

echo "`date`: processing for ENS-NC ${DATE}T${HOUR} done"

# do FWI for 00 hour
if [ "${HOUR}" == "00" ]; then
    NCFILE=fc${DATE:0:4}${DATE:5:2}${DATE:8:2}${HOUR}00_ENS.nc
    FWIFILE=Fwi_${DATE:0:4}${DATE:5:2}${DATE:8:2}${HOUR}00_ENS.nc
    echo "`date`: processing for ENS-FWI ${DATE}T${HOUR} start"
    ${PROGDIR}/process_fwi_ens.py --summary --anomaly \
	      ${UPLOAD} \
	      --fctype ENS \
              --ncall ${FWISAVEDIR}/${FWIFILE} \
	      --grib ${TMPDIR}/${GRIBFILE} \
	      --nc ${FWIDIR}/${FWIFILE} \
	      --log ${LOGFILE}
    ${PROGDIR}/fwi_analysis_ens.py -i ${FWIDIR}/${FWIFILE} \
              -o ${FWIADIR}/${FWIFILE} \
	      --log ${LOGFILE}
    echo "`date`: processing for ENS-FWI ${DATE}T${HOUR} done"
    # do FWI calibrations
    # removed calibration 2024-02-22
    #echo "`date`: processing for FWI calibration ${DATE}T${HOUR}"
    #FWICALIN=${FWIDIR}/${FWIFILE}
    #FWICALOUT=${FWICALIN:0:-3}_cal.nc
    #${PROGDIR}/FWI_calibration.sh "${FWICALIN}" "${FWICALOUT}" "${LOGFILE}"
    #echo "`date`: processing for FWI calibration ${DATE}T${HOUR} done"
fi

# threshold calculations
# 2024-03-26, skip this as there are some memory problems
exit 0

echo "`date`: processing for ENS thresholds ${DATE}T${HOUR} start"
THFILE=${NCDIR}/threshold_ENS_${DATE:0:4}${DATE:5:2}${DATE:8:2}${HOUR}00.nc

${PROGDIR}/safers_threshold.py \
          ${UPLOAD} \
          --fctype ENS \
          --variables rule30 \
          --grib ${TMPDIR}/${GRIBFILE} \
          --out ${THFILE} \
          --log ${LOGFILE}
echo "`date`: processing for ENS thresholds ${DATE}T${HOUR} done"
