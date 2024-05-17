#!/bin/bash

# need to activate python virtual environment here
source /data/python/safers/bin/activate

# PROXY settings
export HTTPS_PROXY=http://wwwproxy.fmi.fi:8080
export HTTP_PROXY=http://wwwproxy.fmi.fi:8080
export ALL_PROXY=http://wwwproxy.fmi.fi:8080
export http_proxy=http://wwwproxy.fmi.fi:8080
export https_proxy=http://wwwproxy.fmi.fi:8080

PROGDIR="$(realpath "$(dirname "$BASH_SOURCE")")"

METADATA=$PROGDIR/metadata_test.json

if [ $# -lt 1 ]; then
    SIZE=100
#    echo give size in MB
#    exit 1
else
    SIZE=$1
fi

echo generating file of size $SIZE MB

DATAFILE=/data/tmp/testfile.nc
dd if=/dev/urandom of=$DATAFILE bs=1MB count=$SIZE

echo ./upload_EC.py $METADATA $DATAFILE
