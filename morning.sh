#!/bin/bash
# safers morning routines
# this is called from crontab!
echo `date "+%Y-%m-%d %H:%M:%S"` ${0} morning run started >> /data/safers/logs/runs.log

PROGDIR="$(realpath "$(dirname "$BASH_SOURCE")")"

${PROGDIR}/tmpcleanup.sh
exit 0
