#!/bin/sh

# sync this directory with safers:

echo "Not in use"
exit 1

SAFERS=safers:safers/scripts/

if [ ! -z "`hostname|egrep safers`" ]; then
    echo "Do not sync in safers"
    exit 1
fi

if [ "$1" == "get" ]; then
    rsync -auv --delete --exclude '*~' ${SAFERS} .
    exit $?
fi

if [ "$1" == "put" ]; then
    # rsync -auv --delete --exclude '*~' . ${SAFERS}
    rsync -auv --exclude '*~' . ${SAFERS}
    exit $?
fi

echo "Usage ${0} get|put"
exit 1
