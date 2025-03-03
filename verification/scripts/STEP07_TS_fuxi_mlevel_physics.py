'''
TS

Yingkai Sha
ksha@ucar.edu
'''

import os
import sys
import yaml
import argparse
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu
import score_utils as su

config_name = os.path.realpath('../verif_config.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('verif_lead', help='verif_lead')
args = vars(parser.parse_args())
verif_lead = int(args['verif_lead'])

# ====================== #
model_name = 'fuxi_mlevel_physics'
ind_lead = verif_lead
# ====================== #

verif_ind_start = 0; verif_ind_end = 2*(365+366)
path_verif = conf[model_name]['save_loc_verif']+'combined_TS_{:03d}d_{}.npy'.format(ind_lead, model_name)

list_thres = 1e-3*np.array([0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 25])
result_ = np.empty((len(list_thres), 3)) # 1st col = all grids; 2nd col = land grids

ds_static = xr.open_zarr(
    '/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_1deg/static/ERA5_mlevel_1deg_static_for_interp.zarr'
)
lsm = ds_static['land_sea_mask'].values

variable_levels = {
    'total_precipitation': None,
}

# ---------------------------------------------------------------------------------------- #
# ERA5 verif target
filename_ERA5 = sorted(glob('/glade/campaign/cisl/aiml/ksha/IMERG_V7/daily/gather_yearly/year_*_1deg_interp.zarr'))

# pick years
year_range = conf['ERA5_ours']['year_range']
years_pick = np.arange(year_range[0], year_range[1]+1, 1).astype(str)
filename_ERA5 = [fn for fn in filename_ERA5 if any(year in fn for year in years_pick)]

# merge yearly ERA5 as one
ds_ERA5 = [vu.get_forward_data(fn) for fn in filename_ERA5]
ds_ERA5_merge = xr.concat(ds_ERA5, dim='time')

# ---------------------------------------------------------------------------------------- #
# forecast
filename_OURS = sorted(glob(conf[model_name]['save_loc_gather']+'*.nc'))

# pick years
year_range = conf[model_name]['year_range']
years_pick = np.arange(year_range[0], year_range[1]+1, 1).astype(str)
filename_OURS = [fn for fn in filename_OURS if any(year in fn for year in years_pick)]

L_max = len(filename_OURS)
assert verif_ind_end <= L_max, 'verified indices (days) exceeds the max index available'

filename_OURS = filename_OURS[verif_ind_start:verif_ind_end]
filename_OURS = [fn for fn in filename_OURS if '00Z' in fn]

# latitude weighting
lat = xr.open_dataset(filename_OURS[0])["latitude"]
w_lat = np.cos(np.deg2rad(lat))
w_lat = w_lat / w_lat.mean()

# ---------------------------------------------------------------------------------------- #
# RMSE compute
ds_ours_all = []
ds_ERA5_all = []

for fn_ours in filename_OURS:
    # detect 00Z vs 12Z
    ini = int(fn_ours[-6:-4])
    
    ds_ours = xr.open_dataset(fn_ours)
    ds_ours = vu.ds_subset_everything(ds_ours, variable_levels)
    
    # ------------------------------------------- #
    # convert neg precip to 0 before accumulation 
    ds_ours['total_precipitation'] = xr.where(
        ds_ours['total_precipitation'] < 0, 0, ds_ours['total_precipitation']
    )
    
    ds_ours_24h = vu.accum_6h_24h(ds_ours, ini)
    ds_ours_24h = ds_ours_24h.isel(time=ind_lead)
    ds_ours_all.append(ds_ours_24h)
    
    ds_target = ds_ERA5_merge.sel(time=ds_ours_24h['time'])
    ds_target = vu.ds_subset_everything(ds_target, variable_levels)
    ds_ERA5_all.append(ds_target)

ds_ours_concat = xr.concat(ds_ours_all, dim='time')
ds_ERA5_concat = xr.concat(ds_ERA5_all, dim='time')

for i_thres, thres in enumerate(list_thres):
    ts_full = su.compute_ts(ds_ours_concat, ds_ERA5_concat, threshold=thres)
    ts_land = su.compute_ts_land_only(ds_ours_concat, ds_ERA5_concat, lsm, threshold=thres)
    ts_mid = su.compute_ts_midlat(ds_ours_concat, ds_ERA5_concat, threshold=thres)
    result_[i_thres, 0] = ts_full
    result_[i_thres, 1] = ts_land
    result_[i_thres, 2] = ts_mid

np.save(path_verif, result_)






