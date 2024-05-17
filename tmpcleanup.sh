#!/bin/sh
# clean up temporary files
# -mtime is modification time in days

DIR=/data/tmp
DIR2=/data/safers/fwi_ens

mkdir -p ${DIR}
touch ${DIR}

mkdir -p ${DIR2}
touch ${DIR2}

# do not erase README.txt
if [ -f ${DIR}/README.txt ]; then
    touch ${DIR}/README.txt
fi

if [ -f ${DIR2}/README.txt ]; then
    touch ${DIR2}/README.txt
fi

find ${DIR} -mtime +3 -delete
find ${DIR2} -mtime +35 -delete
