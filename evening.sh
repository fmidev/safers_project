#!/bin/bash
# safers evening routines
# this is called from crontab!

echo `date "+%Y-%m-%d %H:%M:%S"` ${0} evening run started >> /data/safers/logs/runs.log

PROGDIR="$(realpath "$(dirname "$BASH_SOURCE")")"

/usr/sbin/logrotate -s /tmp/safers.logrotate.status ${PROGDIR}/logrotate.conf

DIR=/data/safers/logs

# do not erase README.txt
if [ -f ${DIR}/README.txt ]; then
    touch ${DIR}/README.txt
fi

find ${DIR} -name '*.log' -mtime +35 -delete

exit 0
