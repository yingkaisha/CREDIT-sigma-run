'''
A collection of functions for computing verification scores
-------------------------------------------------------
Content:
    - bootstrap_confidence_intervals
    - zonal_energy_spectrum_sph
    - compute_ts
    - compute_ts_land_only
    - compute_ts_midlat
    
Yingkai Sha
ksha@ucar.edu
'''

import numba as nb
import numpy as np
import xarray as xr
import pyshtools

def bootstrap_confidence_intervals(rmse_t2m, 
                                   num_bootstrap_samples=1000, 
                                   lower_quantile=0.05, 
                                   upper_quantile=0.95,
                                   random_seed=None):
    """
    Compute confidence intervals over the 'day' dimension of rmse_t2m using bootstrapping.

    Parameters:
    - rmse_t2m: numpy array of shape (n_days, n_lead_times)
    - num_bootstrap_samples: int, number of bootstrap samples to draw
    - lower_percentile, upper_percentile: float, confidence intervals
    - random_seed: int or None, seed for the random number generator for reproducibility

    Returns:
    - ci_lower: numpy array of shape (n_lead_times,), lower bounds of the confidence intervals
    - ci_upper: numpy array of shape (n_lead_times,), upper bounds of the confidence intervals
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # call the numba-optimized function
    bootstrap_data = bootstrap_core(rmse_t2m, num_bootstrap_samples)
    
    # Compute confidence intervals outside numba
    ci_lower = np.quantile(bootstrap_data, lower_quantile, axis=0)
    ci_upper = np.quantile(bootstrap_data, upper_quantile, axis=0)
    mean_score = np.mean(bootstrap_data, axis=0)
    
    return mean_score, ci_lower, ci_upper

@nb.njit()
def bootstrap_core(rmse_t2m, num_bootstrap_samples):
    n_days, n_lead_times = rmse_t2m.shape
    bootstrap_data = np.empty((num_bootstrap_samples, n_lead_times))
    
    for i in range(num_bootstrap_samples):
        ind = np.random.randint(0, n_days)
        bootstrap_data[i, :] = rmse_t2m[ind, :]  # Shape: (n_days, n_lead_times)
        
    return bootstrap_data

def zonal_energy_spectrum_sph(ds_input: xr.Dataset, 
                              varname: str,
                              grid_type: str ='DH',
                              rescale=False)-> xr.DataArray:
    '''
    Computes the Zonal Energy Spectrum of a variable in an xarray.Dataset 
    using spherical harmonic transform. The output is rescaled by the 
    mean circumference per longitude grid.
    
    Parameters:
    - ds_input: xarray.Dataset containing the data.
    - varname: Name of the variable to compute the spectrum for.
    - grid_type: 'GLQ' or 'DH'
    - rescale: produce m * unit result based on circumference

    Returns:
    - spectrum: xarray.DataArray containing the zonal energy spectrum.
    '''
    RAD_EARTH = 6371000
    
    data = ds_input[varname]

    # check 'latitude' and 'longitude' cooridnate names
    if 'latitude' not in data.dims or 'longitude' not in data.dims:
        raise ValueError("Data must have 'latitude' and 'longitude' dimensions")
        
    latitudes = data['latitude'].values
    longitudes = data['longitude'].values

    # check latitudes for [90, -90] descending order
    # if not flip data and latitude
    if latitudes[0] < latitudes[-1]:
        data = data.isel(latitude=slice(None, None, -1))
        latitudes = data['latitude'].values
        
    # check longitudes for [0, 360] order
    # if not re-organize
    if np.any(longitudes < 0):
        longitudes = (longitudes + 360) % 360
        sorted_indices = np.argsort(longitudes)
        data = data.isel(longitude=sorted_indices)
        longitudes = data['longitude'].values

    # number of grids
    nlat = len(latitudes)
    nlon = len(longitudes)
    
    # max wavenumber is half of the latitude grids -1
    max_wavenum = (nlat - 1) // 2  # int divide
    
    # allocate zonal wavenumbers ranges
    zonal_wavenumbers = np.arange(max_wavenum + 1)

    def compute_power_m(data_array_2d):
        '''
        Computes the power spectrum for a 2D data array using spherical harmonics.

        Parameters:
        - data_array_2d: 2D numpy array of shape (nlat, nlon)

        Returns:
        - power_m: 1D numpy array of power corresponding to each zonal wavenumber m
        '''
        # initialize SHGrid
        grid = pyshtools.SHGrid.from_array(data_array_2d, grid=grid_type)
        
        # expand the grid to spherical harmonic coefs
        coeffs = grid.expand(normalization='ortho', lmax_calc=max_wavenum)

        # power per degree per order. shape=(lmax+1, lmax+1)
        coeffs_squared = coeffs.coeffs[0]**2 + coeffs.coeffs[1]**2
        
        # allocate power array for each zonal wavenumber m
        power_m = np.zeros(max_wavenum + 1)
        
        # sum over degrees l > m for each order m to get the total power
        # -l < m < l
        for l in range(max_wavenum + 1):
            power_m[l] = np.sum(coeffs_squared[l:, l])
        
        return power_m

    # xr.apply_ufunc scope
    spectrum = xr.apply_ufunc(
        compute_power_m,
        data,
        input_core_dims=[['latitude', 'longitude']],
        output_core_dims=[['zonal_wavenumber']],
        vectorize=True,
        dask='parallelized',  # <-- dask parallelization
        output_dtypes=[float],
    )

    # assign new coordinate 'zonal_wavenumber'
    spectrum = spectrum.assign_coords(zonal_wavenumber=zonal_wavenumbers)

    if rescale:
        # re-scale power spectrum based on the mean circumference per longitude
        cos_latitudes = np.cos(np.deg2rad(latitudes))
        normalization_factor = (RAD_EARTH * np.sum(cos_latitudes)) / nlon
        
        spectrum = spectrum * normalization_factor
    
    return spectrum

def compute_ts(ds_fcst, ds_obs, threshold=0.1, dims=None):
    """
    Compute the Threat Score (TS, also known as the Critical Success Index) for 
    precipitation forecast verification between forecast and observation datasets.
    
    TS is defined as:
    
        TS = Hits / (Hits + Misses + False Alarms)
    
    where:
        - Hits:        number of times both forecast and observation exceed the threshold.
        - Misses:      number of times the observation exceeds the threshold but the forecast does not.
        - False Alarms: number of times the forecast exceeds the threshold but the observation does not.
    
    Parameters:
    -----------
    ds_fcst : xarray.Dataset
        Forecast dataset containing the variable 'total_precipitation'.
    ds_obs : xarray.Dataset
        Observation (target) dataset containing the variable 'total_precipitation'.
    threshold : float, optional
        Threshold value for defining a precipitation event (default is 0.1).
    dims : list of str, optional
        List of dimensions over which to aggregate the contingency counts.
        If None (default), TS is computed over all dimensions of 'total_precipitation'.
    
    Returns:
    --------
    ts : xarray.DataArray or float
        The Threat Score. If aggregation is performed over all dimensions,
        a scalar float is returned; otherwise, a DataArray with the remaining dims.
    """
    # Extract the precipitation fields.
    fcst = ds_fcst["total_precipitation"]
    obs = ds_obs["total_precipitation"]
    
    # Create binary event arrays: True where precipitation is at least the threshold.
    fcst_event = fcst >= threshold
    obs_event = obs >= threshold
    
    # If no dims are specified, aggregate over all dimensions.
    if dims is None:
        dims = list(fcst.dims)
    
    # Compute the contingency table components.
    hits = ((fcst_event) & (obs_event)).sum(dim=dims)
    misses = ((~fcst_event) & (obs_event)).sum(dim=dims)
    false_alarms = ((fcst_event) & (~obs_event)).sum(dim=dims)
    
    # Compute TS = Hits / (Hits + Misses + False Alarms)
    denominator = hits + misses + false_alarms
    ts = xr.where(denominator == 0, np.nan, hits / denominator)
    
    # If the result is a 0-dim DataArray, convert it to a scalar.
    if ts.ndim == 0:
        # If the underlying data is a Dask array, compute first.
        if hasattr(ts.data, "compute"):
            ts = ts.compute().item()
        else:
            ts = ts.item()
    
    return ts

def compute_ts_land_only(ds_fcst, ds_obs, lsm, threshold=0.1, dims=None):
    """
    Compute the Threat Score (TS) for precipitation forecast verification 
    on land grid points only (where lsm==1).
    
    TS (or Critical Success Index) is defined as:
    
        TS = Hits / (Hits + Misses + False Alarms)
    
    Parameters
    ----------
    ds_fcst : xarray.Dataset
        Forecast dataset containing the variable 'total_precipitation'.
    ds_obs : xarray.Dataset
        Observation (target) dataset containing the variable 'total_precipitation'.
    lsm : numpy.ndarray
        Land–sea mask with dimensions (latitude, longitude) where gridpoints with lsm==1 
        are considered land.
    threshold : float, optional
        Precipitation threshold to consider an event (default is 0.1).
    dims : list of str, optional
        List of dimensions over which to aggregate the contingency counts.
        If None (default), TS is computed over all dimensions of 'total_precipitation'.
    
    Returns
    -------
    ts : xarray.DataArray or float
        The Threat Score computed over the specified dimensions (only for land gridpoints).
        If aggregation is performed over all dimensions, a scalar float is returned.
    """
    # Convert the numpy land-sea mask into an xarray DataArray.
    # (Assumes that the forecast dataset has 'latitude' and 'longitude' coordinates.)
    land_mask = xr.DataArray(lsm == 1, 
                             dims=["latitude", "longitude"],
                             coords={"latitude": ds_fcst.latitude, "longitude": ds_fcst.longitude})
    
    # Extract the precipitation fields from forecast and observation datasets.
    fcst = ds_fcst["total_precipitation"]
    obs = ds_obs["total_precipitation"]
    
    # Create binary event arrays: True if precipitation >= threshold.
    fcst_event = fcst >= threshold
    obs_event = obs >= threshold
    
    # If dims is None, aggregate over all dimensions in the variable.
    if dims is None:
        dims = list(fcst.dims)
    
    # When performing the logical operations, include only grid points where land_mask is True.
    # Because land_mask is defined only on ('latitude', 'longitude'), it will automatically
    # broadcast along any extra dimensions (e.g., 'time').
    hits = ((fcst_event) & (obs_event) & land_mask).sum(dim=dims, skipna=True)
    misses = ((~fcst_event) & (obs_event) & land_mask).sum(dim=dims, skipna=True)
    false_alarms = ((fcst_event) & (~obs_event) & land_mask).sum(dim=dims, skipna=True)
    
    # Compute TS = Hits / (Hits + Misses + False Alarms), avoiding division by zero.
    denominator = hits + misses + false_alarms
    ts = xr.where(denominator == 0, np.nan, hits / denominator)
    
    # If the result is 0-dimensional, convert to a Python scalar.
    if ts.ndim == 0:
        if hasattr(ts.data, "compute"):
            ts = ts.compute().item()
        else:
            ts = ts.item()
    
    return ts

def compute_ts_midlat(ds_fcst, ds_obs, threshold=0.1, dims=None):
    # Restrict the datasets to latitudes between -60 and 60.
    # This approach works regardless of whether the latitude coordinate is ascending or descending.
    ds_fcst = ds_fcst.sel(latitude=ds_fcst.latitude.where((ds_fcst.latitude >= -60) & (ds_fcst.latitude <= 60), drop=True))
    ds_obs  = ds_obs.sel(latitude=ds_obs.latitude.where((ds_obs.latitude >= -60) & (ds_obs.latitude <= 60), drop=True))
    
    # Extract the precipitation fields.
    fcst = ds_fcst["total_precipitation"]
    obs  = ds_obs["total_precipitation"]
    
    # Create binary event arrays: True where precipitation is at least the threshold.
    fcst_event = fcst >= threshold
    obs_event  = obs  >= threshold
    
    # If no dims are specified, aggregate over all dimensions.
    if dims is None:
        dims = list(fcst.dims)
    
    # Compute the contingency table components.
    hits         = ((fcst_event) & (obs_event)).sum(dim=dims)
    misses       = ((~fcst_event) & (obs_event)).sum(dim=dims)
    false_alarms = ((fcst_event) & (~obs_event)).sum(dim=dims)
    
    # Compute TS = Hits / (Hits + Misses + False Alarms)
    denominator = hits + misses + false_alarms
    ts = xr.where(denominator == 0, np.nan, hits / denominator)
    
    # If the result is a 0-dimensional DataArray, convert it to a Python scalar.
    if ts.ndim == 0:
        if hasattr(ts.data, "compute"):
            ts = ts.compute().item()
        else:
            ts = ts.item()
    
    return ts