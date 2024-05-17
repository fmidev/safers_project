#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from pandas import to_datetime
from safers_s3 import latest_time

args = sys.argv
if len(args) > 1:
    dir = args[1] + '/'
else:
    dir = 'ec_hres/'

d = to_datetime(latest_time(dir))

print(d.strftime('%Y-%m-%dT%H'))
