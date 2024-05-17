#!/bin/bash

# Call FWI calibration routine
# input: FWIIN FWIOUT LOGFILE

# need to activate python virtual environment here
source /data/python/calib/bin/activate

# PROXY settings
export HTTPS_PROXY=http://wwwproxy.fmi.fi:8080
export HTTP_PROXY=http://wwwproxy.fmi.fi:8080
export ALL_PROXY=http://wwwproxy.fmi.fi:8080
export http_proxy=http://wwwproxy.fmi.fi:8080
export https_proxy=http://wwwproxy.fmi.fi:8080

PROGDIR="$(realpath "$(dirname "$BASH_SOURCE")")"
# PROGDIR=/home/users/lainema/safers_research/calibration

if [ $# -lt 3 ]; then
    echo Needs 3 inputs
    exit 1
fi

cd ${PROGDIR}

./fwi_calibration.py --fwiin "${1}" \
   --fwiout "${2}" \
   --log "${3}"

exit 0
