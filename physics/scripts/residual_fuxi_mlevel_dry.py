import os
import sys
import xarray as xr
import numpy as np
from glob import glob

sys.path.insert(0, os.path.realpath('../../libs/'))
from physics_utils import grid_area, pressure_integral, weighted_sum

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('ind_start', help='verif_ind_start')
parser.add_argument('ind_end', help='verif_ind_end')
args = vars(parser.parse_args())

ind_start = int(args['ind_start'])
ind_end = int(args['ind_end'])
# ====================== #
key = 'fuxi_mlevel_dry'

# Earth's radius
RAD_EARTH = 6371000 # m
RVGAS = 461.5 # J/kg/K
RDGAS = 287.05 # J/kg/K
GRAVITY = 9.80665 # m/s^2
RHO_WATER = 1000.0 # kg/m^3
LH_WATER = 2.501e6  # J/kg
LH_ICE = 333700 # J/kg
CP_DRY = 1004.64 # J/kg K
CP_VAPOR = 1810.0 # J/kg K
CP_LIQUID = 4188.0 # J/kg K
CP_ICE = 2117.27 # J/kg K

N_seconds = 3600 * 6  # 6-hourly data

def model_level_integral(q, level_p):
    '''
    Compute the model level integral of a given quantity using xarray integration.

    Args:
        q: xarray.DataArray
        level_p: xarray.DataArray

    Returns:
        xarray.DataArray: integrals of q
    '''
    q_coord_name = 'level'
    p_coord_name = 'half_level'
    dp = level_p.diff(dim=p_coord_name)
    dp = dp.rename({p_coord_name: q_coord_name})
    q = q.assign_coords(level=dp[q_coord_name])
    q_int = (q * dp).sum(dim=q_coord_name)
    return q_int

def water_budget_residual(q, precip, evapor, N_seconds, area, level_p):
    '''
    Compute water budget residuals using xarray DataArrays.

    Args:
        q: xarray.DataArray of specific total water (time, level, latitude, longitude)
        precip: xarray.DataArray of total precipitation (time, latitude, longitude), units m
        evapor: xarray.DataArray of evaporation (time, latitude, longitude), units m
        N_seconds: Number of seconds between time steps
        area: xarray.DataArray of grid cell areas (latitude, longitude), units m^2
        level_p: xarray.DataArray of pressure levels, units Pa

    Returns:
        residual: xarray.DataArray of water budget residuals over time
    '''
    # Convert increments to fluxes (kg/m^2/s)
    precip_flux = precip.isel(time=slice(1, None)) * RHO_WATER / N_seconds  # kg/m^2/s
    evapor_flux = evapor.isel(time=slice(1, None)) * RHO_WATER / N_seconds  # kg/m^2/s

    # Compute Total Water Content (TWC) at each time step
    TWC = model_level_integral(q, level_p) / GRAVITY  # kg/m^2

    # Compute time derivative of TWC (difference over time)
    dTWC_dt = TWC.diff('time') / N_seconds  # kg/m^2/s
    
    # Compute weighted sums over area
    dTWC_sum = weighted_sum(dTWC_dt, area, dims=('latitude', 'longitude'))  # kg/s
    E_sum = weighted_sum(evapor_flux, area, dims=('latitude', 'longitude'))  # kg/s
    P_sum = weighted_sum(precip_flux, area, dims=('latitude', 'longitude'))  # kg/s

    TWC_sum = weighted_sum(TWC, area, dims=('latitude', 'longitude'))
    
    # Compute residual
    residual = -dTWC_sum - E_sum - P_sum

    return residual, dTWC_sum, E_sum, P_sum


fcst_dir = f'/glade/campaign/cisl/aiml/ksha/CREDIT_cp/GATHER/{key}/'
verif_dir = f'/glade/campaign/cisl/aiml/ksha/CREDIT_cp/VERIF/{key}/'

# Load datasets
base_dir_plevel = '/glade/derecho/scratch/ksha/CREDIT_data/ERA5_plevel_1deg/'
base_dir_mlevel = '/glade/derecho/scratch/ksha/CREDIT_data/ERA5_mlevel_1deg/'

# Static dataset
filename_static = base_dir_mlevel + 'static/ERA5_mlevel_1deg_static_subset.zarr'
ds_static = xr.open_zarr(filename_static)

x = ds_static['longitude']
y = ds_static['latitude']
lon, lat = np.meshgrid(x, y)
# Compute grid cell areas
area = grid_area(lat, lon)

# Years to process
years = np.arange(2020, 2022, 1)

GPH_surf = ds_static['geopotential_at_surface']
coef_a = ds_static['coef_a']
coef_b = ds_static['coef_b']

ds_water_collect = []

# ============================================================= #
fcst_files = sorted(glob(fcst_dir + '*00Z.nc'))
fcst_files = fcst_files[ind_start:ind_end]

for fn in fcst_files:
    ds_fcst = xr.open_dataset(fn)
    
    q = ds_fcst['specific_total_water_mlevel']
    T = ds_fcst['T_mlevel']
    u = ds_fcst['U_mlevel']
    v = ds_fcst['V_mlevel']
    precip = ds_fcst['total_precipitation']
    evapor = ds_fcst['evaporation']
    sp = ds_fcst['SP']
    
    q = q.drop_vars(['level'])
    T = T.drop_vars(['level'])
    u = u.drop_vars(['level'])
    v = v.drop_vars(['level'])

    level_p = coef_a + coef_b * sp

    water_residual, water_tendency, evapor_sum, precip_sum = water_budget_residual(
        q, precip, evapor, N_seconds, area, level_p)
    
    ds_water = xr.Dataset({
        'water_residual': water_residual,
        'water_tendency': water_tendency,
        'evapor': evapor_sum,
        'precip': precip_sum,
    })
    
    ds_water_collect.append(ds_water.drop_vars('time'))

ds_water_all = xr.concat(ds_water_collect, dim='days')
save_name_water = verif_dir + key + '_water_residual_subset_{:05d}_{:05d}.nc'
ds_water_all.to_netcdf(save_name_water.format(ind_start, ind_end), compute=True)



