import xarray as xr
import numpy as np
import os
import my_pymannkendall     as  mk # pymannkendall modified by bbarceló to show effective sample size (ESS)
from scipy.stats import chi2, t, f

'''
All Functions to perform statistical analysis

Adapted from R. Escudier (2018), B. Barceló-Llull (2018; 2024) and P. Rosselló (2023)

@author: Blanca-Fdez (2025)

'''

# Directory and File Structure Handling
def dirtodict(dirPath):
    #From a given folder, create a dictionary with a given folder & file tree structure
    d = {}
    for i in [os.path.join(dirPath, i) for i in os.listdir(dirPath)
              if os.path.isdir(os.path.join(dirPath, i))]:
        d[os.path.basename(i)] = dirtodict(i)
    d['.files'] = [os.path.join(dirPath, i) for i in os.listdir(dirPath)
                   if os.path.isfile(os.path.join(dirPath, i))]
    return d

# Spatial Calculations and Averaging 
def area(lat,lon):
    """
    Compute area of a rectilinear grid.
    """
    earth_radius = 6371e3
    omega = 7.2921159e-5

    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    f = 2*omega* np.sin(lat_r)
    grad_lon = lon_r.copy()
    grad_lon.data = np.gradient(lon_r)

    dx=grad_lon*earth_radius*np.cos(lat_r)
    dy=np.gradient(lat_r)*earth_radius
    
    ds_area = xr.DataArray((dx*dy).T,
                      coords = {'lat' : lat,
                                'lon' : lon},
                      dims = ['lat','lon'])

    return ds_area

def weighted_mean(da):
    
    ds_area = area(da.lat, da.lon)
    if "time" in da.dims:
        ds_area = ds_area.where(da.isel(time=0).notnull().compute(), drop=True)

    area_sum = np.nansum(np.array(ds_area))
    weighted = ds_area/area_sum

    result = (da*weighted).sum(('lat', 'lon'))

    # Preserve attributes (like units)
    result.attrs = da.attrs.copy()
    
    return result

def weighted_std(da):
    """
    Adjusting for uneven weighting is important because when a
    few points dominate the data, standard variance calculations 
    become biased, underestimating variability. Correcting for this
    ensures accurate uncertainty estimates by accounting for reduced
    degrees of freedom.

    info about weitghed mean and its uncertanties:
    Bevington, P. R., Data Reduction and Error Analysis for the Physical
    Sciences, 336 pp., McGraw-Hill, 1969
    """

    ds_area = area(da.lat, da.lon)
    if "time" in da.dims:
        ds_area = ds_area.where(da.isel(time=0).notnull().compute(), drop=True)
    
    # Normalize the weights
    area_sum = np.nansum(np.array(ds_area))
    weights = ds_area / area_sum
    
    # Weighted mean at each time step
    weighted_mean = (da * weights).sum(dim=('lat', 'lon'))
    
    # Variance for each grid cell (over lat/lon) and SE for each time step
    # Calculate weighted variance
    weighted_variance = ((da - weighted_mean) ** 2 * weights).sum(dim=('lat', 'lon'))

    # Effective degrees of freedom
    N_eff = 1.0 / (weights ** 2).sum(dim=('lat', 'lon'))

    # Weighted standard deviation
    weighted_std = np.sqrt(weighted_variance * (N_eff / (N_eff - 1)))
    

    weighted_se = weighted_std / np.sqrt(N_eff)


    return weighted_std

# Statistical and Regression Analysis for Time Series
def lin_regression(y_nan, X_nan):
    """
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Linear regression of a time serie 
    % 
    % [B,y_est,S,N] = lin_regression(y_nan, X_nan)
    %
    % IN:
    %       - y_nan: the estimand
    %       - X_nan: matrix with M input variables as columns (size NxM)
    %
    % OUT:
    %       - B     : vector column (size M) with the response coefficients
    %       - y_est : estimation of the estimand with the regression
    %       - S     : hindcast skill of the regression
    %       - N     : number of good values
    %
    % Written by R. Escudier (2018)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    # Check if dimensions are consistent
    N_nan,M = X_nan.shape
    if (N_nan != len(y_nan)):
       print ('Error: The column of X must have the same size as y!')
       return None  
    
    # Ignore NaNs
    id_nonan = np.where(~np.isnan(y_nan))
    y = y_nan[id_nonan]
    X = X_nan[id_nonan,:].squeeze()
    N = len(y)

    # Solve linear regression
    B = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Compute estimate -> BUSCAR LA MANERA DE HACERLO AUTOMATICO
    y_est_nonan = (B[0]*X).sum(axis=1)
    y_est = (B[0]*X_nan).sum(axis=1)
    # Sy_est = xr.dot(X, B, dims="time")  # Ensure dims match
    S = np.var(y_est_nonan)/np.var(y)
    
    return B, y_est, S, N

def coef_dof_skill(y, X):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculate degrees of freedom for the skill of a regression with two methods
    % 
    % [nu_askill,nu_pdf] = coef_dof_skill(y,X)
    %
    % IN:
    %       - y: the estimand
    %       - X: matrix with M input variables as columns (size NxM)
    %
    % OUT:
    %       - nu_askill : Artificial skill estimate
    %       - nu_pdf    : pdf estimate
    %
    % Written by R. Escudier (2018) from D. Chelton method
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    N_tot, M = np.shape(X)
    M = M-1
    
    # Choose lag range (60-80% of dataset size)
    N_min, N_max = int(np.floor(0.6 * N_tot)), int(np.floor(0.8 * N_tot))
    K = N_max - N_min + 1
    
      # Initialization
    S = np.zeros((2*K,))
    N = np.zeros((2*K,))

    print('starting get skill')
    # Get skill values
    for i_cur,k in enumerate(range(N_min,N_max+1)):
       _,_,S[i_cur],N[i_cur]     = lin_regression(y[k:],X[:-k,:])
       _,_,S[i_cur+K],N[i_cur+K] = lin_regression(y[:-k],X[k:,:])

    print('Done get skill')
    # Artificial skill estimate
    NS = S*N
    A = NS.sum() / (2 * K)
    nu_askill = M / A
    
    # pdf estimate
    SCorr = S / (1 - S)
    SNCorr = SCorr * N
    A1 = SCorr.sum() / (2 * K)
    A2 = SNCorr.sum() / (2 * K)
    nu_pdf = (M + (M + 3) * A1) / A2
    
    print('done coef')
    return nu_askill, nu_pdf

def lin_regression_with_skillcrit(y_nan, X_nan, a=0.05):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Linear regression of a time serie 
    % 
    % [B,y_est,S,N,S_crit,dB,N_eff] = lin_regression_with_skillcrit(y_nan,X_nan)
    %
    % IN:
    %       - y_nan: the estimand
    %       - X_nan: matrix with M input variables as columns (size NxM)
    % OPTIONS:
    %       - a  : alpha parameter for the confidence test (at 100(1-a)%)
    %       - nu : coefficient for the dof
    %
    % OUT:
    %       - B     : vector column (size M) with the response coefficients
    %       - y_est : estimation of the estimand with the regression
    %       - S     : hindcast skill of the regression
    %       - N     : number of good values
    %       - S_crit: critical value for the null hypothesis that S=0
    %       - dB    : Intervals of confidence for the coefficients 
    %       - N_eff : Effective number of dof
    %
    % Written by R. Escudier (2018)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    N_nan, M = np.shape(X_nan)
    M = M-1
    print('lin_regresion')
    B, y_est, S, N = lin_regression(y_nan, X_nan)

    print('coef dof skill')
    # Compute the nu coeff (average of both methods)
    nu_askill, nu_pdf = coef_dof_skill(y_nan, X_nan)
    nu = (nu_askill + nu_pdf)/2 # mean of both methods 
    N_eff = N * nu # Neff (effective degrees of freedom)
    
    # Compute Confidence intervals for B
    D = np.atleast_2d(X_nan.T @ X_nan)
    print("Shape of X_nan:", X_nan.shape)
    print("NaN count in X_nan:", np.isnan(X_nan).sum())
    print("Number of non-NaN rows:", np.sum(~np.isnan(X_nan).all(axis=1)))
    print("Rank of X_nan:", np.linalg.matrix_rank(X_nan))
    print("Shape of D:", D.shape)
    # Check if D is singular by trying the inversion or calculating determinant
    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        print("D matrix at failure:", D)
        raise ValueError("Matrix D is singular, cannot invert.")
    
    D_inv = np.linalg.inv(D)
    print("N_eff - M - 1:", N_eff - M - 1) 
    qt = t.ppf(1 - a / 2, float(N_eff - M - 1))
    dB = np.nanstd(y_nan)*np.sqrt(np.diag(D_inv) * (1 - S) / float(N_eff - M - 1)) * qt;

    # Compute S_crit   
    qf = f.ppf(1 - a, M, float(N_eff - M - 1))
    S_crit = M * qf / (float(N_eff - M - 1) + M * qf)
    
    return B, y_est, S, N, S_crit, dB, N_eff


#bbarcelo
def mk_test(ts):
    
    if np.isnan(ts).all():
        return np.nan, np.nan, np.nan, np.nan
    results = mk.yue_wang_modification_test(ts)
    slope, p_value, intercept, n_ns = results.slope, results.p, results.intercept, results.n_ns
    
    # Compute standard error of the slope
    SE_slope = standard_error_of_slope(ts, slope, intercept, n_ns)

    return slope, p_value, intercept, SE_slope

#bbarcelo
def standard_error_of_slope(ts, slope, intercept, n_ns):
    
    '''   
    Compute standard error of the slope using the effective 
    sample size (ESS).

    The standard error of the slope is calculated as the 
    residual standard error divided by the square root of 
    the sum of squared differences in the independent 
    variable (James et al., 2023), considering the effective 
    sample size of the time series (Stan Development 
    Team, 2021; Martínez-Moreno et al., 2021).

    info about standard error of the slope (James et al., 2023): 
    https://www.statlearning.com/ (eq. 3.8)

    info about ESS here (Stan Development Team, 2021): 
    https://mc-stan.org/docs/2_21/reference-manual/effective-sample-size-section.html 

    '''

    ''' 
    Preprocessing the time series to skip nans
    (as in mk.yue_wang_modification_test) 
    '''
    ts_pp, c = mk.__preprocessing(ts)
    ts_p, n  = mk.__missing_values_analysis(ts_pp, method = 'skip')

    ''' 1) Calculate residual standard error '''

    # Create a time array (assuming time points are consecutive integers)
    time = np.arange(len(ts_p))

    # Calculate the predicted values using the Sen's slope
    y_pred = slope * time + intercept 

    # Calculate the residuals
    residuals = ts_p - y_pred

    # Calculate the Sum of Squared Residuals (RSS)
    RSS = np.sum(residuals**2)

    # Compute effective sample size (ESS)
    ESS  = len(ts_p)/n_ns

    # print(' ')
    # print('len(ts)', len(ts))
    # print('len(ts_p)', len(ts_p))
    # print('n_ns', n_ns)
    # print('ESS', ESS)

    # Compute residual standard error

    RSE = np.sqrt(RSS/(ESS-2)) 

    ''' 2) Compute square root of the sum of squared 
    differences in the independent variable '''

    denom = np.sqrt(np.sum((time - np.mean(time))**2))

    ''' 3) Compute standard error '''
    
    SE_slope = RSE / denom

    return SE_slope


# EXTRAS -> P. Rosselló (2023)
import json
import copy
import warnings


def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

def nonzero_and_not_nan(arr):
    # Replace zeros with NaNs temporarily
    arr_with_nans = np.where(arr == 0, np.nan, arr)

    # Find indices of non-NaN elements
    indices = np.argwhere(~np.isnan(arr_with_nans))

    # Extract non-zero, non-NaN elements
    result = [arr_with_nans[idx[0], idx[1], idx[2]] for idx in indices]

    return result

## Utils for MHW detection -> adapted from P.Rosselló (2023)
# Original in DOI: 10.5281/zenodo.7908932 (https://github.com/canagrisa/MHW_moving_ﬁxed)

def run_avg_per(sst, w=11):
    """
    Calculate the periodic moving average of a given sequence.

    This function calculates the periodic moving average of a given input
    array `sst` using a sliding window of size `w`. The function also handles
    periodicity by extending the input array to the left and right, which
    allows for accurate calculations near the edges of the input data.

    Parameters
    ----------
    sst : numpy.ndarray or list
        The input sequence for which the periodic moving average will be calculated.
    w : int, optional, default=11
        The window size for the moving average.

    Returns
    -------
    avg : numpy.ndarray
        The calculated periodic moving average of the input sequence with the same
        shape as the input.
    """

    var = np.array(sst)
    hw = w // 2

    # Pad the input array with periodic boundary conditions
    var_padded = np.pad(var, pad_width=(hw, hw), mode="wrap")

    # Calculate the moving average using convolution
    avg = np.convolve(var_padded, np.ones(w), "valid") / w

    return avg

def get_win_clim(sst, w=11, year_length=365):
    """
    Calculate day-of-year w-day window rolling mean and smooth with
    a 31-day periodic moving average.

    Parameters
    ----------
    sst : numpy.ndarray or list
        The input sequence for which the rolling percentile will be calculated.
    w : int, optional, default=11
        The window size for the rolling mean.

    Returns
    -------
    fin_run : numpy.ndarray
        The calculated day-of-year rolling percentile smoothed with a 31-day
        periodic moving average.
    """

    if np.isnan(sst[0]):
        return np.array([10000.0] * year_length)

    n = len(sst) // year_length
    var = np.reshape(np.array(sst), (n, year_length))
    hw = w // 2

    # Extend the input array with periodic boundary conditions
    var_ext = np.pad(var, pad_width=((0, 0), (hw, hw)), mode="wrap")

    # Calculate the rolling percentile
    clim = [np.mean(var_ext[:, i : i + 2 * hw + 1]) for i in range(year_length)]

    fin = np.array(clim)
    fin_run = run_avg_per(fin, w=31)

    return fin_run

def get_win_pctl(sst, w=11, p=90, year_length=365):
    """
    Calculate day-of-year w-day window rolling p-percentile and smooth with
    a 31-day periodic moving average.

    Parameters
    ----------
    sst : numpy.ndarray or list
        The input sequence for which the rolling percentile will be calculated.
    w : int, optional, default=11
        The window size for the rolling percentile.
    p : int, optional, default=90
        The percentile value to be calculated within the rolling window.

    Returns
    -------
    fin_run : numpy.ndarray
        The calculated day-of-year rolling percentile smoothed with a 31-day
        periodic moving average.
    """

    if np.isnan(sst[0]):
        return np.array([10000.0] * year_length)

    n = len(sst) // year_length
    var = np.reshape(np.array(sst), (n, year_length))
    hw = w // 2

    # Extend the input array with periodic boundary conditions
    var_ext = np.pad(var, pad_width=((0, 0), (hw, hw)), mode="wrap")

    # Calculate the rolling percentile
    tdh = [np.nanpercentile(var_ext[:, i : i + 2 * hw + 1], p) for i in range(year_length)]

    fin = np.array(tdh)
    fin_run = run_avg_per(fin, w=31)

    return fin_run

def compute_thresholds(
    baseline, window=11, q=90, year_length=365, var="tos", lat="lat", lon="lon"
):
    """
    Compute the rolling q-percentile of the baseline sea surface temperature
    dataset with a specified window size and smooth it using a 31-day periodic moving average.

    Parameters
    ----------
    baseline : xarray.Dataset
        The baseline dataset containing sea surface temperature data.
    window : int, optional, default=11
        The window size for the rolling percentile calculation.
    q : int, optional, default=90
        The percentile value to be calculated within the rolling window.

    Returns
    -------
    ds_thres : xarray.Dataset
        The dataset with the calculated rolling q-percentile smoothed with a 31-day
        periodic moving average.
    """
    baseline_arr = baseline[var].values
    thres = np.apply_along_axis(
        get_win_pctl, 0, baseline_arr, window, q, year_length=year_length
    )
    clim = np.apply_along_axis(
        get_win_clim, 0, baseline_arr, window, year_length=year_length
    )

    baseline = baseline.sortby("time")
    ds_thres = baseline.isel(time=slice(0, year_length))
    ds_thres = ds_thres.groupby("time.dayofyear").mean(dim="time")
    ds_thres = ds_thres.drop_vars(var)
    ds_thres["pctl"] = (("dayofyear", lat, lon), thres)
    ds_thres["clim"] = (("dayofyear", lat, lon), clim)

    ds_thres = ds_thres.where(ds_thres.pctl < 9999)

    return ds_thres

#################################################################
## New climatology functions (Blanca Fernández-Álvarez - 2025)

def run_avg(da, w=11, dim="dayofyear", stat="mean", p=90):
    """
    Periodic rolling mean for an xr.DataArray.
    Wraps around at the edges.
    """
    hw = w // 2
    da_ext = xr.concat(
        [da.isel({dim: slice(-hw, None)}), da, da.isel({dim: slice(0, hw)})],
        dim=dim
    )

    if stat == "mean":
        da_roll = da_ext.rolling({dim: w}, center=True, min_periods=1).mean()
    elif stat == "percentile":
        da_roll = da_ext.rolling({dim: w}, center=True, min_periods=1).construct("window")
        da_roll = da_roll.reduce(np.nanpercentile, dim="window", q=p)
    else:
        raise ValueError("stat must be 'mean' or 'percentile'")

    return da_roll.isel({dim: slice(hw, -hw)})

def get_climatology(baseline, w=11, p=90, smooth=31, var="tos", lat="lat", lon="lon"):
    """
    Compute rolling climatology and percentile thresholds

    Parameters
    ----------
    baseline : xr.Dataset
        SST dataset with 'time', 'lat', 'lon'
    w : int, optional, default=11
        Rolling window size (in days).
    p : int, optional, default=90
        Percentile value (used only if stat='percentile').
    smooth : int
        Smoothing window for final climatology
    Returns
    -------
    xr.DataArray
        Day-of-year climatology or percentile (smoothed).
    """
    da = baseline[var].sortby("time")
    # Sort by time just to be safe

    clim = run_avg(da, w=w, dim="time", stat="mean")
    pctl = run_avg(da, w=w, dim="time", stat="percentile", p=p)

    clim = clim.groupby("time.dayofyear").mean("time")
    pctl = pctl.groupby("time.dayofyear").mean("time")

    clim_smooth = run_avg(clim, w=smooth, dim="dayofyear", stat="mean")
    pctl_smooth = run_avg(pctl, w=smooth, dim="dayofyear", stat="mean")

    ds_thres = xr.Dataset(
        {
            "clim": clim_smooth,
            "pctl": pctl_smooth
        },
        coords={
            "dayofyear": clim_smooth["dayofyear"],
            lat: da[lat],
            lon: da[lon],
        },
    )
    return ds_thres
#####################################################################################

def mhs_to_mhw(mhs, min_days=5, gap=2):
    """
    :mhs: should be an array of 1s and 0s, with 1s corresponding to SST above the threshold.
    """

    mhs = np.array(mhs)
    split_indices = np.where(np.diff(mhs) != 0)[0] + 1
    split_bool = np.split(mhs, split_indices)
    split = copy.deepcopy(split_bool)

    num_splits = len(split_bool)

    # Handle the first group
    if split_bool[0][0] == 1 and len(split_bool[0]) < min_days:
        split[0] = [0] * len(split_bool[0])

    for i in range(1, num_splits - 1):
        current_group = split_bool[i]
        previous_group = split_bool[i - 1]
        next_group = split_bool[i + 1]

        if (
            current_group[0] == 0
            and len(current_group) <= gap
            and len(previous_group) >= min_days
            and len(next_group) >= min_days
        ):
            split[i] = [1] * len(current_group)
        elif current_group[0] == 1 and len(current_group) < min_days:
            split[i] = [0] * len(current_group)

    # Handle the last group
    if split_bool[-1][0] == 1 and len(split_bool[-1]) < min_days:
        split[-1] = [0] * len(split_bool[-1])

    mhw = np.concatenate(split)

    return mhw

def mhw_duration_1d(arr_1d):
    mhw_durations = []
    mhw_duration = 0
    for day in arr_1d:
        if not np.isnan(day):
            if day == 1:
                mhw_duration += 1
            elif mhw_duration >= 5:
                mhw_durations.append(mhw_duration)
                mhw_duration = 0
            else:
                mhw_duration = 0
        else:
            nan_arr = np.empty(63)
            nan_arr[:] = np.nan
            return nan_arr

    if mhw_duration >= 5:
        mhw_durations.append(mhw_duration)

    values = np.empty(63)
    values[:] = np.nan
    values[: len(mhw_durations)] = np.array(mhw_durations)

    return np.array(values)

def mhw_mean_anomaly_1d(arr_1d):
    mhw_ans = []
    mhw_an = 0
    i = 0
    for an in arr_1d:
        if not np.isnan(an):
            i += 1
            mhw_an += an
        elif i >= 5:
            mhw_ans.append(mhw_an / i)
            mhw_an = 0
            i = 0
        else:
            mhw_an = 0
            i = 0

    values = np.empty(63)
    values[:] = np.nan
    values[: len(mhw_ans)] = np.array(mhw_ans)

    return np.array(values)

def find_mhw_durations(arr):
    duration_arr = np.apply_along_axis(mhw_duration_1d, 0, arr)
    return duration_arr

def find_mean_anomaly(arr):
    anomaly_arr = np.apply_along_axis(mhw_mean_anomaly_1d, 0, arr)
    return anomaly_arr


def MHW_metrics_satellite(
    ds,
    baseline_years,
    baseline_type,
    out_folder="../../results/MHW/satellite/",
    var="tos",
    distribution=False,
    error=False,
):
    year_length = 365
    if ds.time.dtype == "datetime64[ns]":
        # If the dataset has leapyears create a boolean mask to identify leap year February 29th
        leap_year_feb29 = (ds.time.dt.month == 2) & (ds.time.dt.day == 29)
        # Remove leap year February 29th from the dataset
        ds = ds.where(~leap_year_feb29, drop=True)

    else:
        time_index = ds.indexes["time"]
        if hasattr(time_index, "calendar"):
            calendar = time_index.calendar
            if calendar == "360_day":
                year_length = 360

    y_i = ds.time.dt.year.min().item()
    y_f = ds.time.dt.year.max().item()

    if baseline_type == "fixed_baseline":
        baseline = ds.sel(time=ds.time.dt.year.isin(range(y_i, y_i + baseline_years)))
        print(f"Baseline range: {y_i} to {y_i + baseline_years}")
        print("Baseline mean:", baseline[var].mean().values)
        print("Baseline std:", baseline[var].std().values)
        print("Number of NaNs in baseline:", baseline[var].isnull().sum().values)
        thresholds = compute_thresholds(baseline, year_length=year_length, var=var)

    lat = ds.lat
    lon = ds.lon

    # Initialize the output dataset
    data_vars = {}
    metrics = [
        "MHS",
        "MHW",
        "MHW_cat_2",
        "MHW_cat_3",
        "MHW_cat_4",
        "mean_anomaly",
        "max_anomaly",
        "cumulative_anomaly",
        "mean_duration",
    ]
    for metric in metrics:
        data_vars[metric] = (
            ["time", "lat", "lon"],
            np.zeros((y_f - y_i - baseline_years + 1, len(lat), len(lon))),
        )
        ds_out = xr.Dataset(
            data_vars,
            coords={
                "time": np.arange(y_i + baseline_years, y_f + 1),
                "lat": lat,
                "lon": lon,
            },
        )
    # print(data_vars)
    if error != False:
        suffixes = ["pos", "neg"]
        for metric in metrics:
            if metric not in ["MHW_cat_2", "MHW_cat_3", "MHW_cat_4"]:
                for app in suffixes:
                    ds_out[f"{metric}_{app}"] = xr.DataArray(
                        np.zeros((y_f - y_i - baseline_years + 1, len(lat), len(lon))),
                        dims=["time", "lat", "lon"],
                        coords={
                            "time": np.arange(y_i + baseline_years, y_f + 1),
                            "lat": lat,
                            "lon": lon,
                        },
                    )
    grouped_years = ds.groupby("time.year")
    distribution_metrics = [
        "MHS_days_year",
        "MHW_days_year",
        "MHW_cat_2_days_year",
        "MHW_cat_3_days_year",
        "MHW_cat_4_days_year",
        "MHW_event_duration",
        "MHW_anual_cumulative_anomaly",
        "MHW_event_mean_anomaly",
    ]

    ofo = out_folder + f"{baseline_type}_{baseline_years}_year/"
    if not os.path.exists(ofo):
        os.makedirs(ofo)

    for year, group in grouped_years:
        if year <= y_i + baseline_years - 1:
            continue
        if ds.time.dtype == "datetime64[ns]":
            group = group.where(~(group.time.dt.dayofyear == 366), drop=True)

        distributions = {key: [] for key in distribution_metrics}

        print(year, end=", ")

        if os.path.exists(f"{ofo}/MHW_{year}.nc"):
            continue
        if baseline_type == "moving_baseline":
            baseline = ds.sel(
                time=ds.time.dt.year.isin(range(year - baseline_years, year))
            )
            # print(f"Processing year: {year}, baseline range: {year - baseline_years} to {year}")
            print("Baseline mean:", baseline[var].mean().values)
            print("Baseline std:", baseline[var].std().values)
            print("Number of NaNs in baseline:", baseline[var].isnull().sum().values)
            thresholds = compute_thresholds(baseline, year_length=year_length, var=var)
            print("Number of NaNs in climatology:", thresholds['clim'].isnull().sum().values)
            print("Number of NaNs in thresholds:", thresholds['pctl'].isnull().sum().values)
        # some code to compute the MHSs per gridcell on that year
        year_thresholds = thresholds.sel(dayofyear=group.time.dt.dayofyear)

        sst = group[var]

        def get_metrics(
            sst, year_thresholds, year, ds_out, distribution=distribution, app=""
        ):
            mhs = (
                (sst > year_thresholds["pctl"])
                .where(year_thresholds["pctl"].notnull())
                .values
            )
            mhw = np.apply_along_axis(mhs_to_mhw, 0, mhs)
            ds_out[f"MHS{app}"].loc[{"time": year}] = np.sum(mhs, axis=0)
            ds_out[f"MHW{app}"].loc[{"time": year}] = np.sum(mhw, axis=0)

            dif = year_thresholds["pctl"] - year_thresholds["clim"]

            # Computing MHW categories

            mhw_cat_2 = (
                (sst > (year_thresholds["pctl"] + dif))
                .where(year_thresholds["pctl"].notnull())
                .values
            )
            ds_out[f"MHW_cat_2"].loc[{"time": year}] = np.sum(mhw_cat_2, axis=0)

            mhw_cat_3 = (
                (sst > (year_thresholds["pctl"] + 2 * dif))
                .where(year_thresholds["pctl"].notnull())
                .values
            )
            ds_out[f"MHW_cat_3"].loc[{"time": year}] = np.sum(mhw_cat_3, axis=0)

            mhw_cat_4 = (
                (sst > (year_thresholds["pctl"] + 3 * dif))
                .where(year_thresholds["pctl"].notnull())
                .values
            )
            ds_out[f"MHW_cat_4"].loc[{"time": year}] = np.sum(mhw_cat_4, axis=0)

            # Computing anomalies

            anomaly = (sst - year_thresholds["clim"]).values
            anomaly = np.where(mhw == 0, np.nan, anomaly)

            durs = find_mhw_durations(mhw)
            mean_an_event = find_mean_anomaly(anomaly)
            

            
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_anomaly = np.nanmax(anomaly, axis=0)
            mean_anomaly = np.nanmean(anomaly, axis=0)
            cumulative_anomaly = np.nansum(anomaly, axis=0)
            mean_duration = np.nanmean(durs, axis=0)

            ds_out[f"max_anomaly{app}"].loc[{"time": year}] = max_anomaly
            ds_out[f"mean_anomaly{app}"].loc[{"time": year}] = mean_anomaly
            ds_out[f"cumulative_anomaly{app}"].loc[{"time": year}] = cumulative_anomaly
            ds_out[f"cumulative_anomaly{app}"] = ds_out[
                f"cumulative_anomaly{app}"
            ].where(ds_out[f"MHS{app}"].notnull())
            ds_out[f"mean_duration{app}"].loc[{"time": year}] = mean_duration
            ds_out[f"mean_duration{app}"] = ds_out[f"mean_duration{app}"].where(
                ds_out[f"MHS{app}"].notnull()
            )

            ds_out_year = ds_out.where(ds_out.time == year, drop=True)
            ds_out_year.to_netcdf(f"{ofo}/MHW_{year}.nc")

            if distribution != False:
                distributions["MHW_event_duration"].append(nonzero_and_not_nan(durs))
                distributions["MHW_event_mean_anomaly"].append(
                    nonzero_and_not_nan(mean_an_event)
                )
                distributions["MHS_days_year"].append(
                    nonzero_and_not_nan(ds_out["MHS"].values)
                )
                distributions["MHW_days_year"].append(
                    nonzero_and_not_nan(ds_out["MHW"].values)
                )
                distributions["MHW_cat_2_days_year"].append(
                    nonzero_and_not_nan(ds_out["MHW_cat_2"].values)
                )
                distributions["MHW_cat_3_days_year"].append(
                    nonzero_and_not_nan(ds_out["MHW_cat_3"].values)
                )
                distributions["MHW_cat_4_days_year"].append(
                    nonzero_and_not_nan(ds_out["MHW_cat_4"].values)
                )
                distributions["MHW_anual_cumulative_anomaly"].append(
                    nonzero_and_not_nan(ds_out["cumulative_anomaly"].values)
                )

                histograms = {}
                bin_params = {
                    "MHS_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_cat_2_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_cat_3_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_cat_4_days_year": {"range": (0, 366), "bin_width": 1},
                    "MHW_event_mean_anomaly": {"range": (0, 7.01), "bin_width": 0.01},
                    "MHW_anual_cumulative_anomaly": {
                        "range": (0, 1001),
                        "bin_width": 1,
                    },
                    "MHW_event_duration": {"range": (0, 366), "bin_width": 1},
                }

                for metric in [
                    "MHS_days_year",
                    "MHW_days_year",
                    "MHW_cat_2_days_year",
                    "MHW_cat_3_days_year",
                    "MHW_cat_4_days_year",
                    "MHW_event_duration",
                    "MHW_anual_cumulative_anomaly",
                    "MHW_event_mean_anomaly",
                ]:
                    histograms[metric] = {}
                    distributions[metric] = flatten(distributions[metric])
                    bin_edges = np.arange(
                        bin_params[metric]["range"][0],
                        bin_params[metric]["range"][1],
                        bin_params[metric]["bin_width"],
                    )
                    hist, bin_edges = np.histogram(
                        distributions[metric], bins=bin_edges
                    )
                    histograms[metric]["hist"] = [float(i) for i in hist]
                    histograms[metric]["bin_edges"] = [float(i) for i in bin_edges]

                fold_distr = ofo + "distributions/"
                if not os.path.exists(fold_distr):
                    os.makedirs(fold_distr)
                file_distr = fold_distr + f"distr_{year}.json"
                with open(file_distr, "w") as outfile:
                    json.dump(histograms, outfile)

        print('Calculating the metrics...')
        get_metrics(sst, year_thresholds, year, ds_out, distribution=distribution)

        if error != False:
            sst_pos = sst + group['analysis_error']
            sst_neg = sst - group['analysis_error']
            get_metrics(
                sst_pos, year_thresholds, year, ds_out, distribution=False, app="_pos"
            )
            get_metrics(
                sst_neg, year_thresholds, year, ds_out, distribution=False, app="_neg"
            )

            # Free up memory for this year's data.
        del group, sst, year_thresholds
        if baseline_type == "moving_baseline":
            del baseline, thresholds
        if error != False:
            del sst_pos, sst_neg
    # if distribution == False:
    #     return ds_out, None
    # else:
    #     histograms = {}
    #     bin_params = {'MHS_days_year': {'range': (0, 366), 'bin_width': 1},
    #                   'MHW_days_year': {'range': (0, 366), 'bin_width': 1},
    #                   'MHW_cat_2_days_year': {'range': (0, 366), 'bin_width': 1},
    #                   'MHW_cat_3_days_year': {'range': (0, 366), 'bin_width': 1},
    #                   'MHW_cat_4_days_year': {'range': (0, 366), 'bin_width': 1},
    #                    'MHW_event_mean_anomaly': {'range': (0, 7.01), 'bin_width': 0.01},
    #                    'MHW_anual_cumulative_anomaly': {'range': (0, 1001), 'bin_width': 1},
    #                    'MHW_event_duration': {'range': (0, 366), 'bin_width': 1}}

    #     for metric in ['MHS_days_year', 'MHW_days_year','MHW_cat_2_days_year', 'MHW_cat_3_days_year', 'MHW_cat_4_days_year', 'MHW_event_duration', 'MHW_anual_cumulative_anomaly', 'MHW_event_mean_anomaly']:
    #         histograms[metric] = {}
    #         distributions[metric] = utils.flatten(distributions[metric])
    #         bin_edges = np.arange(bin_params[metric]['range'][0], bin_params[metric]['range'][1],
    #                               bin_params[metric]['bin_width'])
    #         hist, bin_edges = np.histogram(
    #             distributions[metric],
    #             bins=bin_edges)
    #         histograms[metric]['hist'] = [float(i) for i in hist]
    #         histograms[metric]['bin_edges'] =  [float(i) for i in bin_edges]

    #     file_distributions = ofo + 'distributions.json'
    #     with open(file_distributions, 'w') as outfile:
    #         json.dump(histograms, outfile)

    return ds_out

def MHW_metrics_one_point(
    ds,
    baseline_years,
    baseline_type,
    year=2020,
    var="analysed_sst",
    error=False,
    window=11,
    q=90,
):
    year_length = 365
    if ds.time.dtype == "datetime64[ns]":
        # If the dataset has leapyears create a boolean mask to identify leap year February 29th
        leap_year_feb29 = (ds.time.dt.month == 2) & (ds.time.dt.day == 29)
        # Remove leap year February 29th from the dataset
        ds = ds.where(~leap_year_feb29, drop=True)

    y_i = ds.time.dt.year.min().item()
    y_f = ds.time.dt.year.max().item()

    def get_threshold(baseline):
        thres = get_win_pctl(
            baseline[var].values, window, q, year_length=year_length
        )
        clim = get_win_clim(baseline[var].values, window, year_length=year_length)
        baseline = baseline.sortby("time")
        ds_out = baseline.isel(time=slice(0, year_length))
        ds_out = ds_out.groupby("time.dayofyear").mean(dim="time")
        ds_out = ds_out.drop_vars(var)
        if error == True:
            ds_out = ds_out.drop_vars("analysis_error")
        ds_out["pctl"] = (("dayofyear"), thres)
        ds_out["clim"] = (("dayofyear"), clim)
        ds_out = ds_out.where(ds_out.pctl < 9999)
        return ds_out

    if baseline_type == "fixed_baseline":
        baseline = ds.sel(time=ds.time.dt.year.isin(range(y_i, y_i + baseline_years)))
        ds_out = get_threshold(baseline)

    if baseline_type == "moving_baseline":
        baseline = ds.sel(time=ds.time.dt.year.isin(range(year - baseline_years, year)))
        ds_out = get_threshold(baseline)

    ds_y = ds.sel(time=ds.time.dt.year.isin([year]))
    ds_y = ds_y.groupby("time.dayofyear").mean(dim="time")
    ds_y = ds_y.where(~(ds_y.dayofyear == 366), drop=True)

    ds_out["sst"] = (("dayofyear"), ds_y[var].values)
    if error == True:
        ds_out["error"] = (("dayofyear"), ds_y["analysis_error"].values)

        for i, typ in enumerate(['pos', 'neg']):

            sign = (-1)**(i)
            sst = ds_out["sst"].values + sign * ds_out["error"].values
            mhs = (
                (sst > ds_out["pctl"]).where(ds_out["pctl"].notnull()).values
            )
            mhw = mhs_to_mhw(mhs)
            anomaly = (sst - ds_out["clim"]).values

            ds_out['MHS_' + typ] = (('dayofyear'), mhs)
            ds_out['MHW_' + typ] = (('dayofyear'), mhw)
            ds_out['anomaly_' + typ] = (('dayofyear'), anomaly)


    mhs = (ds_out["sst"] > ds_out["pctl"]).where(ds_out["pctl"].notnull()).values
    mhw = mhs_to_mhw(mhs)
    if error == True:
        anomaly = (sst - ds_out["clim"]).values
        anomaly = np.where(mhw == 0, np.nan, anomaly)
        ds_out["anomaly"] = (("dayofyear"), anomaly)

    ds_out["MHS"] = (("dayofyear"), mhs)
    ds_out["MHW"] = (("dayofyear"), mhw)
    

    return ds_out