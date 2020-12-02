#!/usr/bin/env python

import xarray as xr

with xr.open_dataset(geo_file_path) as ds:
    df = ds.to_dataframe()

    df[ df['link'] in ]
