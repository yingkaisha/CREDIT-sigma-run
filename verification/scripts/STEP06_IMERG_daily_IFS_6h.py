'''
SEEPS using IMERG

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
import seeps_utils as seeps

config_name = os.path.realpath('../verif_config.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('verif_ind_start', help='verif_ind_start')
parser.add_argument('verif_ind_end', help='verif_ind_end')
args = vars(parser.parse_args())

verif_ind_start = int(args['verif_ind_start'])
verif_ind_end = int(args['verif_ind_end'])

# ====================== #
model_name = 'IFS'
lead_range = conf[model_name]['lead_range']
verif_lead_range = conf[model_name]['verif_lead_range']

leads_exist = list(np.arange(lead_range[0], lead_range[-1]+lead_range[0], lead_range[0]))
leads_verif = list(np.arange(verif_lead_range[0], verif_lead_range[-1]+verif_lead_range[0], verif_lead_range[0]))
ind_lead = vu.lead_to_index(leads_exist, leads_verif)

print('Verifying lead times: {}'.format(leads_verif))
print('Verifying lead indices: {}'.format(ind_lead))
# ====================== #

path_verif = conf[model_name]['save_loc_verif']+'combined_IMERG_{:04d}_{:04d}_{:03d}h_{:03d}h_{}.nc'.format(
    verif_ind_start, verif_ind_end, verif_lead_range[0], verif_lead_range[-1], model_name)

ds_clim = xr.open_dataset(
    '/glade/campaign/cisl/aiml/ksha/CREDIT_physics/VERIF/ERA5_clim/IMERG_clim_2000_2020_SEEPS.nc'
)
#/glade/campaign/cisl/aiml/ksha/CREDIT_physics/VERIF/ERA5_clim/ERA5_clim_1990_2019_SEEPS.nc
seeps_calc = seeps.SpatialSEEPS(climatology=ds_clim, precip_name='total_precipitation')

# ---------------------------------------------------------------------------------------- #
# ERA5 verif target
filename_ERA5 = sorted(glob('/glade/campaign/cisl/aiml/ksha/IMERG_V7/GPM_3B_V07/year_*_1deg_interp.zarr'))

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

variable_levels = {'total_precipitation': None,}
variable_levels_IFS = {'total_precipitation_6hr': None,}
rename_IFS_to_ERA5 = {'total_precipitation_6hr': 'total_precipitation'}

# ---------------------------------------------------------------------------------------- #
# RMSE compute
verif_results = []

for fn_ours in filename_OURS:
    # detect 00Z vs 12Z
    ini = int(fn_ours[-6:-4])
    
    ds_ours = xr.open_dataset(fn_ours)
    ds_ours = ds_ours.isel(time=ind_lead)
    ds_ours = vu.ds_subset_everything(ds_ours, variable_levels_IFS)
    ds_ours = ds_ours.rename(rename_IFS_to_ERA5)
    
    ds_ours_24h = vu.accum_6h_24h(ds_ours, ini)
    ds_ours_24h = ds_ours_24h.compute()
    
    ds_target = ds_ERA5_merge.sel(time=ds_ours_24h['time'])
    ds_target = vu.ds_subset_everything(ds_target, variable_levels)
    ds_target_24h = ds_target.compute()
    
    # SEEPS
    seeps_score = seeps_calc.compute_chunk(ds_ours_24h, ds_target_24h)
    seeps_score_mean = (w_lat * seeps_score).mean(['latitude', 'longitude'])
    
    verif_results.append(seeps_score_mean.drop_vars('time'))
    
# Combine verif results
ds_verif_24h = xr.concat(verif_results, dim='days')

# Save the combined dataset
print('Save to {}'.format(path_verif))
ds_verif_24h.to_netcdf(path_verif, mode='w')



