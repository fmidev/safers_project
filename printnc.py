#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# display contents of nc file

import sys
import xarray as xr

args = sys.argv
if len(args) < 2:
    print('usage:', args[0], 'nc file')
    sys.exit(1)

file = args[1]
variable = args[2] if len(args) > 2 else None

if variable is None:
    print(xr.open_dataset(file))
else:
    print(xr.open_dataset(file)[variable])
