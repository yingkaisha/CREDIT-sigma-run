'''
This script creates 1 deg ERA5 training data
for CREDIT stage 2

Yingkai Sha
ksha@ucar.edu
'''

import os
import sys
import yaml
import dask
import zarr
import numpy as np
import xarray as xr
from glob import glob

import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu
import interp_utils as iu

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

# get year from input
year = int(args['year'])

base_dir_mlevel = '/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_1deg/'
base_dir_plevel = '/glade/derecho/scratch/ksha/CREDIT_data/ERA5_plevel_1deg/'
base_dir_output = base_dir_mlevel

fn_fmt_mlevel = base_dir_mlevel + 'upper_air/ERA5_mlevel_1deg_6h_{}_conserve.zarr'
fn_fmt_cloud = base_dir_mlevel + 'cloud/ERA5_mlevel_1deg_6h_cloud_{}_conserve.zarr'
fn_fmt_plevel = base_dir_plevel + 'all_in_one/ERA5_plevel_1deg_6h_{}_conserve.zarr'
fn_fmt_static = base_dir_plevel + 'static/ERA5_plevel_1deg_6h_conserve_static.zarr'
fn_mean_std = '/glade/derecho/scratch/ksha/CREDIT_data/mean_6h_1979_2018_16lev_0.25deg.nc'

# mlevel_picks = [1, 2, 4, 6, 9, 12, 15, 18, 21, 24, 27, 
#                 30, 33, 36, 39, 43, 47, 51, 54, 58, 62,
#                 66, 70, 74, 77, 81, 84, 87, 90, 93, 97, 
#                 100, 104, 107, 111, 114, 116, 119, 122, 
#                 124, 126, 128, 131, 133, 136, 137]

mlevel_picks = [  1,   9,  19,  29,  39,  49,  59,  69,  79,
                 89,  97, 104, 111, 116, 122, 126, 131, 136]

var_mlevel = {
    'specific_humidity': mlevel_picks,
    'temperature': mlevel_picks,
    'u_component_of_wind': mlevel_picks,
    'v_component_of_wind': mlevel_picks
}

var_cloud = {
    'specific_cloud_liquid_water_content': mlevel_picks,
    'specific_rain_water_content': mlevel_picks,
}

chunk_size_3d = {
    'time': 10,
    'latitude': 181,
    'longitude': 360
}

chunk_size_4d = {
    'time': 10,
    'level': 46,
    'latitude': 181,
    'longitude': 360
}

encode_size_3d = dict(
    chunks=(
        chunk_size_3d['time'],
        chunk_size_3d['latitude'],
        chunk_size_3d['longitude']
    )
)

encode_size_4d = dict(
    chunks=(
        chunk_size_4d['time'],
        chunk_size_4d['level'],
        chunk_size_4d['latitude'],
        chunk_size_4d['longitude']
    )
)

ds_mlevel = xr.open_zarr(fn_fmt_mlevel.format(year))
ds_mlevel_sub = vu.ds_subset_everything(ds_mlevel, var_mlevel)

ds_cloud = xr.open_zarr(fn_fmt_cloud.format(year))
ds_cloud_sub = vu.ds_subset_everything(ds_cloud, var_cloud)

Q = ds_mlevel_sub['specific_humidity'] + \
    ds_cloud_sub['specific_cloud_liquid_water_content'] + \
    ds_cloud_sub['specific_rain_water_content']

ds_mlevel_sub['specific_total_water'] = Q

ds_plevel = xr.open_zarr(fn_fmt_plevel.format(year))
ds_plevel_sub = ds_plevel.drop_vars(['U', 'V', 'T', 'Q', 'Z', 'specific_total_water'])

ds_plevel_sub = ds_plevel_sub.drop_vars(['level',])
ds_merge = xr.merge([ds_mlevel_sub, ds_plevel_sub])

varnames = list(ds_merge.keys())
varname_4D = [
    'specific_humidity',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'specific_total_water'
]

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        ds_merge[var] = ds_merge[var].chunk(chunk_size_4d)
    else:
        ds_merge[var] = ds_merge[var].chunk(chunk_size_3d)

dict_encoding = {}

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        dict_encoding[var] = {'compressor': compress, **encode_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **encode_size_3d}

save_name = base_dir_output + 'all_in_one/ERA5_mlevel_1deg_6h_subset_{}_conserve.zarr'.format(year)
ds_merge.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)

