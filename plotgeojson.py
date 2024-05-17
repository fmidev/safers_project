#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# plot data in geojson file

import sys
from safers_plots import plotgeojson

args = sys.argv
if len(args) < 2:
    print('usage:', args[0], 'geojsonfile [graphics_file]')
    sys.exit(1)

file = args[1]
gfile = args[2] if len(args) > 2 else None

plotgeojson(file, file=gfile)

