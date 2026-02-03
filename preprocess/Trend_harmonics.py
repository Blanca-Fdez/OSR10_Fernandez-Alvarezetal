
#%%
import numpy      as np
import xarray     as xr
import dask
import sys
import os
from dask.diagnostics import ProgressBar


sys.path.append('/home/bfernandez/Escritorio/MHW/Estadistica')
import blanca_tools         as btools

"""
Created on Thu Jan 30 12:10 2025

@author: Blanca-Fdez

Code to analyse the long-term warming of the Balearic Sea

Analysis done with SST

Code adapted from Bàrbara Barceló-Llull (bbarcelo)
As used in Barceló-Llull et al. (2019)  https://doi.org/10.1029/2018JC014636 (harmonic anlysis) and
Barceló-Llull et al. (2024) https://doi.org/10.48550/arXiv.2406.08014 (mk trends)

"""



def harmonic_model(t, T, n_harmonics=1):
    """
    Creates a matrix of harmonic components for regression.

    IN:
    - t: Time array
    - T: Base period (e.g., 365 for annual cycle)

    OPTIONS:
    - n_harmonics: Number of harmonics to include (default is 1 for annual)

    OUT:
    - Harmonic matrix with [1, sin(2πt/T), cos(2πt/T), sin(4πt/T), cos(4πt/T), ...]
    """
    harmonics = [np.ones_like(t)]  # Intercept term

    for n in range(1, n_harmonics + 1):
        harmonics.append(np.sin(2 * np.pi * n * t / T))
        harmonics.append(np.cos(2 * np.pi * n * t / T))

    return np.stack(harmonics, axis=1)  # Shape: (len(t), 2*n_harmonics + 1)



def harmonic_model_with_trend(t, T, n_harmonics=1):
    """
    Constructs a harmonic regression matrix with trend.
    """
    harmonics = [np.ones_like(t), t]  # Intercept and trend

    for n in range(1, n_harmonics + 1):
        harmonics.append(np.sin(2 * np.pi * n * t / T))
        harmonics.append(np.cos(2 * np.pi * n * t / T))

    return np.stack(harmonics, axis=1)  # Shape: (len(t), 2*n_harmonics + 2)



def fit_harmonic(sst_time_series, T, n_harmonics=1):
    """
    Fits a harmonic regression model to an SST time series.

    IN:
    - sst_time_series: 1D NumPy array of SST values over time
    - T: Base period for harmonics (e.g., 365.25 for annual)

    OPTIONS:
    - n_harmonics: Number of harmonics to include

    OUT:
    - mean (float): SST mean
    - amplitude (float): Amplitude of the first harmonic
    - phase (float): Phase shift of the first harmonic
    - trend (float): Linear trend component
    """
    mask = ~np.isnan(sst_time_series).compute()
    if mask.sum() <= 20:  # Skip if not enough valid data
        return np.nan, np.nan, np.nan, np.nan

    t = np.arange(len(sst_time_series))[mask]
    t = xr.DataArray(np.arange(len(sst_time_series)), dims=sst_time_series.dims, coords=sst_time_series.coords)

    sst_m = sst_time_series.where(mask, drop=True)

    X = harmonic_model(t.values, T, n_harmonics)
    coeffs, _, _, _ = np.linalg.lstsq(X, sst_m, rcond=None)

    # Extract components
    mean = coeffs[0]
    As, Ac = coeffs[1], coeffs[2]  # First harmonic coefficients
    amplitude = np.sqrt(As**2 + Ac**2)
    phase = np.arctan2(-As, Ac)

    return mean, amplitude, phase

def compute_residuals(sst_time_series, T, n_harmonics=1):
    """Compute residuals using fit_harmonic_and_trend."""

    # Mask non-NaN values
    if isinstance(sst_time_series, xr.DataArray) and isinstance(sst_time_series.data, dask.array.core.Array):
        sst_time_series = sst_time_series.compute()
    mask = ~np.isnan(sst_time_series)

    t = np.arange(len(sst_time_series))[mask]
    # t = xr.DataArray(np.arange(len(sst_time_series)), dims=sst_time_series.dims, coords=sst_time_series.coords)
    sst_m = sst_time_series[mask]

    if mask.sum() <= 20:  # Skip if not enough valid data
        return np.full(sst_time_series.shape, np.nan)
    
    # Fit harmonics
    X = harmonic_model(t, T, n_harmonics)
    coeffs, _, _, _ = np.linalg.lstsq(X, sst_m, rcond=None)

    # Compute fitted values
    fitted_harmonics = X @ coeffs
    residuals = sst_m - fitted_harmonics

    # Create a residuals array with the same shape as the input, filled with NaNs
    residuals_full = np.full(sst_time_series.shape, np.nan)

    # Place the computed residuals in the positions where data is valid (mask is True)
    residuals_full[mask] = residuals

    return residuals_full

def fit_harmonic_and_trend(sst_time_series, T, n_harmonics=1):
    """
    Fits a harmonic regression model to an SST time series separating the trend
    THIS FUNCTION IS M0DIFIED TO APPLY THE MK TEST AND COMPUTE THE THEIL-SEN SLOPE

    IN:
    - sst_time_series: 1D NumPy array of SST values over time
    - T: Base period for harmonics (e.g., 365.25 for annual)

    OPTIONS:
    - n_harmonics: Number of harmonics to include
    
   OUT:
    - mean: Mean SST value from the harmonic regression [units]
    - amplitude: Amplitude of first harmonic
    - phase: Phase of first harmonic
    - harmonic_trend: Trend from harmonic regression [units per T]
    - slope: Theil-Sen Slope from the SST timeseries after removing the seasonal 
    component [units per T]
    - SE : Standard error (SE) of the Theil-Sen slope, using the effective 
    sample size (ESS) to account for correlation in the data [units per T]
    - intercept: The intercept of the residual trend after removing the seasonal 
    and harmonic components [units]
    
    Notes:
    ------
    - If fewer than 20 valid (non-NaN) points exist, the function returns NaNs to avoid unreliable estimates.
    - The harmonic trend comes from a linear regression that includes a trend term.
    - The residual trend is estimated separately using the Mann-Kendall test and Theil-Sen slope.
    
    """
    mask = ~np.isnan(sst_time_series)
    if mask.sum() <= 20:  # Skip if not enough valid data
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    t = np.arange(len(sst_time_series))[mask]
    # t = xr.DataArray(np.arange(len(sst_time_series)), dims=sst_time_series.dims, coords=sst_time_series.coords)
    sst_m = sst_time_series[mask]

    # Fit harmonics
    X = harmonic_model(t, T, n_harmonics)
    coeffs, _, _, _ = np.linalg.lstsq(X, sst_m, rcond=None)

    # Extract main harmonic components
    mean = coeffs[0]
    As, Ac = coeffs[1], coeffs[2]
    amplitude = np.sqrt(As**2 + Ac**2)
    phase = np.arctan2(-As, Ac)

    # Fit harmonics with trend 
    X_with_trend = harmonic_model_with_trend(t, T, n_harmonics)
    coeffs_with_trend, _, _, _ = np.linalg.lstsq(X_with_trend, sst_m, rcond=None)

    # Extract main harmonic components
    mean = coeffs_with_trend[0]
    As, Ac = coeffs_with_trend[2], coeffs_with_trend[3]
    amplitude = np.sqrt(As**2 + Ac**2)
    phase = np.arctan2(-As, Ac)
    harmonic_trend = coeffs_with_trend[1] * T # Assuming trend is part of first coefficient

    # Compute residuals
    fitted_harmonics = X @ coeffs
    residuals = sst_m - fitted_harmonics

    # # Fit a linear model to residuals to extract trend
    # X_residuals = np.column_stack((np.ones_like(t), t))  # [1, time]
    # residual_coeffs, _, _, _ = np.linalg.lstsq(X_residuals, residuals, rcond=None)
    # residual_trend = residual_coeffs[1] * T  # Slope of the residuals

    slope,  p_value, intercept, se  = btools.mk_test(residuals)

    return mean, amplitude, phase, harmonic_trend, slope*T, se*T, intercept, p_value

def fit_harmonic_with_stats(sst_time_series, T, n_harmonics=1):
     
    """
    Fits a harmonic regression model with a trend component and accounts for statistical validation.
    The function estimates seasonal (harmonic) patterns while also extracting a long-term trend.

    IN:
    - sst_time_series: 1D NumPy array of SST values over time
    - T: Base period for harmonics (e.g., 365.25 for annual)

    OPTIONS:
    - n_harmonics: Number of harmonics to include

    OUT:
    - mean: Mean SST [units]
    - amplitude: Amplitude of first harmonic
    - phase: Phase of first harmonic
    - trend: Estimated linear trend in SST [units per T]
    - diff_trend : Difference between the estimated trend and its confidence interval, provinding
    an uncertanty estimate [units per T]
    - error trend: Standard error of the trend, adjusted for autocorrelation using the effective sample size
    - diff_trend: Theil-Sen Slope from the SST timeseries - seasonal component [units per T]
    - N_eff : Effective sample size (ESS), which corrects for serial correlation in the time series.

    Notes:
    ------
    - If fewer than 20 valid (non-NaN) points exist, the function returns NaNs to avoid unreliable estimates.
    - The harmonic model is fitted using linear regression with skill-based criteria.
    """

    mask = ~np.isnan(sst_time_series)
    if mask.sum() <= 20:  # Skip if not enough valid data
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    t = np.arange(len(sst_time_series))[mask]
    # t = xr.DataArray(np.arange(len(sst_time_series)), dims=sst_time_series.dims, coords=sst_time_series.coords)
    sst_m = sst_time_series[mask]
    
    X = harmonic_model_with_trend(t, T, n_harmonics)

    B, y_est, S, N, S_crit, dB, N_eff = btools.lin_regression_with_skillcrit(sst_m, X)
    
    mean = B[0]
    trend = B[1]
    amplitude = np.sqrt(B[2]**2 + B[3]**2)
    phase = np.arctan2(-B[2], B[3])
    
    diff_As = np.abs(B[2]) - np.abs(dB[2])
    diff_Ac = np.abs(B[3]) - np.abs(dB[3])
    diff_trend = (np.abs(B[1]) - np.abs(dB[1])) * T

    error_trend = dB[1] * T
    
    return mean, amplitude, phase, trend, diff_trend, error_trend, N_eff




##################################################################################
# Define the folder paths
fold = "/home/bfernandez/Escritorio/OSR10/data/"
out_fold = "/home/bfernandez/Escritorio/OSR10/harmonics_trends/"
if not os.path.exists(out_fold):
    os.makedirs(out_fold)

# Load the dataset
files = btools.dirtodict(fold)[".files"]
ds = xr.open_mfdataset(files, combine="by_coords")
sst    = ds['analysed_sst']  # original sst
T = 365.25 # Annual period on days

'''
err  = ds.variables['analysis_error']  # Error derived from interpolation
time_axis  = ds.variables['time']
ds_lat   = ds.variables['lat']
ds_lon    = ds.variables['lon']
'''

def process_and_save(sst, T):
    """
    Process and save the harmonics and residuals (trend with skills?).
    """
    harmonic_results_path = f"{out_fold}/results/harmonic_results.nc"
    residuals_path = f"{out_fold}/residuals/residuals_chunk.nc"
    # harmonic_stats_results_path = f"{out_fold}/results_skill/harmonic_stats_results_chunk_{chunk_index}.nc"
   
    # Ensure directories exist
    os.makedirs(os.path.dirname(harmonic_results_path), exist_ok=True)
    os.makedirs(os.path.dirname(residuals_path), exist_ok=True)
    # os.makedirs(os.path.dirname(harmonic_stats_results_path), exist_ok=True)
    
    # Fit harmonic model
    results = xr.apply_ufunc(
        fit_harmonic_and_trend, 
        sst, 
        T, 
        kwargs={"n_harmonics": 1},  
        input_core_dims=[["time"], []],  
        output_core_dims=[[], [], [], [], [], [], [], []],  
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32] * 8  
    )
    # Convert tuple to xarray Dataset
    result_names = ["mean", "amplitude", "phase", "harmonic_trend", "slope", "se", "intercept", "p_value"]
    results_ds = xr.Dataset({name: data for name, data in zip(result_names, results)},
        coords={
        'lat': sst['lat'],  # Convert latitudes to numpy array
        'lon': sst['lon']  # Convert longitudes to numpy array
    })

    with ProgressBar():
        results_ds.compute().to_netcdf(harmonic_results_path)
    del results_ds
    print('Main results saved')

    residuals = xr.apply_ufunc(
        compute_residuals,  
        sst,  
        T,  
        kwargs={"n_harmonics": 1},  
        input_core_dims=[["time"], []],  
        output_core_dims=[["time"]],  # Residuals keep time dimension
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32]
        )
    
    residuals_da = xr.DataArray(residuals, name='residuals')

    with ProgressBar():
        residuals_da.compute().to_netcdf(residuals_path)
    del residuals_da
    print('Residuals saved')

    # #Fit harmonic model with stats
    # results_skill = xr.apply_ufunc(
    #     fit_harmonic_with_stats,
    #     sst_chunk,
    #     T,
    #     kwargs={"n_harmonics": 1},
    #     input_core_dims=[["time"], []],
    #     output_core_dims=[[], [], [], [], [], [], []],
    #     vectorize=True,
    #     dask="parallelized",
    #     output_dtypes=[np.float32] * 7
    # )

    # result_skill_names = ["mean", "amplitude", "phase", "trend", "diff_trend", "error_trend", "N_eff"]
    # results_skill_ds = xr.Dataset({name: data for name, data in zip(result_skill_names, results_skill)},
    #                               coords={'lat': sst_chunk['lat'], 'lon': sst_chunk['lon']})

    # with ProgressBar():
    #     results_skill_ds.compute().to_netcdf(f"{out_fold}/results_skill/harmonic_stats_results_chunk_{chunk_index}.nc")
    # del results_skill_ds
    # print('Results with skill saved')
    # print(f"Chunk {chunk_index} saved successfully!")

process_and_save(sst, T)

print("All chunks processed and saved successfully!")

# print('Fitting harmonics and obtaining results...')
# results = xr.apply_ufunc(
#     fit_harmonic_and_trend, 
#     sst, 
#     T, 
#     kwargs={"n_harmonics": 1},  
#     input_core_dims=[["time"], []],  
#     output_core_dims=[[], [], [], [], [], [], []],  
#     vectorize=True,
#     dask="parallelized",
#     output_dtypes=[np.float32] * 7  
# )
# # Convert tuple to xarray Dataset
# result_names = ["mean", "amplitude", "phase", "harmonic_trend", "slope", "se", "intercept"]
# results_ds = xr.Dataset({name: data for name, data in zip(result_names, results)},
#         coords={
#         'lat': sst['lat'],  # Convert latitudes to numpy array
#         'lon': sst['lon']  # Convert longitudes to numpy array
#     })

# with ProgressBar():
#     results_ds.compute().to_netcdf(f"{out_fold}harmonic_results.nc")
# del results_ds
# print("All harmonic components saved.")
# # prueba = compute_residuals(sst.isel(lat=69, lon=78), T)
# # print(prueba)

# residuals = xr.apply_ufunc(
#     compute_residuals,  
#     sst,  
#     T,  
#     kwargs={"n_harmonics": 1},  
#     input_core_dims=[["time"], []],  
#     output_core_dims=[["time"]],  # Residuals keep time dimension
#     vectorize=True,
#     dask="parallelized",
#     output_dtypes=[np.float32]
# )

# residuals_da = xr.DataArray(residuals, name='residuals')

# print("Saving residuals...")
# with ProgressBar():
#     residuals_da.compute().to_netcdf(f"{out_fold}residuals.nc")
# del residuals_da  # Free memory

# print("All results saved successfully!")


# ## Now we do the same for the spatially averaged SST
# sst_mean = btools.weighted_mean(sst)
# mean_results = fit_harmonic_and_trend(sst_mean, T, n_harmonics=1)
# mean_results_ds = xr.Dataset({name: data for name, data in zip(result_names, mean_results)})

# mean_residuals = compute_residuals(sst_mean, T)
# residuals_da = xr.DataArray(mean_residuals, name='residuals')

# mean_ds = xr.merge([mean_results_ds, residuals_da])

# # Save to NetCDF
# mean_ds.load().to_netcdf(out_fold + "harmonic_analysis_mean.nc")
# print('harmonic_analysis_mean saved')

# ### EXTRA 
# results_skill = xr.apply_ufunc(
#     fit_harmonic_with_stats,
#     sst,  
#     T, 
#     kwargs={"n_harmonics": 1},  
#     input_core_dims=[["time"], []],  
#     output_core_dims=[[], [], [], [], [], [], []],  
#     vectorize=True,
#     dask="parallelized",
#     output_dtypes=[np.float32] * 7
# )

# # Compute results if using Dask

# # Convert tuple to xarray Dataset
# result_skill_names = ["mean", "amplitude", "phase", "trend", "diff_trend", "error_trend", "N_eff"]
# results_skill_ds = xr.Dataset({name: data for name, data in zip(result_skill_names, results_skill)},
#         coords={
#         'lat': sst['lat'],  # Convert latitudes to numpy array
#         'lon': sst['lon']  # Convert longitudes to numpy array
#     })

# # Save to NetCDF
# results_skill_ds.load().to_netcdf(out_fold + "harmonic_stats_results.nc")


# %%
