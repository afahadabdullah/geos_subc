
"""
MJO Wave Spatial Extraction Engine
====================================
This script isolates the Madden-Julian Oscillation (MJO) spatial signal from raw
daily Outgoing Longwave Radiation (OLR) fields using Space-Time (Wavenumber-Frequency) filtering.

The MJO is mathematically defined in atmospheric dynamics as an eastward-propagating
equatorially-trapped wave envelope with:
    - Zonal Wavenumbers: 1 to 5
    - Period: 30 to 90 days

Pipeline:
    1. Load daily NOAA interpolated OLR (olr.day.mean.nc)
    2. Interpolate to target GEOS 1° grid (181x360)
    3. Remove the seasonal cycle (climatology) to get OLR anomalies
    4. Apply 2D FFT (Space-Time) along the longitudes.
    5. Zero out all frequencies and wavenumbers EXCEPT those matching the MJO profile
    6. Apply Inverse 2D FFT to recover the isolated MJO wave in the spatial domain
    7. Save the 1999 MJO wave spatial map to Zarr
    8. Generate a diagnostic plot comparing Raw OLR vs. MJO Wave

Dependencies:
    pip install xarray numpy pandas scipy dask matplotlib cartopy
"""

import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import glob
from scipy.fft import fft2, ifft2, fftfreq, fftshift

# Attempt to load plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg') # Headless backend for TACC
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    PLOT_AVAILABLE = True
except ImportError:
    print("Warning: cartopy or matplotlib not installed. Diagnostic plots will be skipped.")
    PLOT_AVAILABLE = False


DEFAULT_START_YEAR = 1999
DEFAULT_END_YEAR = 2025


def load_olr_dataset(olr_paths):
    existing_paths = [p for p in olr_paths if os.path.exists(p)]
    missing_paths = [p for p in olr_paths if not os.path.exists(p)]

    for path in missing_paths:
        print(f"Warning: OLR file not found and will be skipped: {path}")

    if not existing_paths:
        raise FileNotFoundError("No valid OLR files were found.")

    print("Loading OLR files:")
    for path in existing_paths:
        print(f"  - {path}")

    if len(existing_paths) == 1:
        ds = xr.open_dataset(existing_paths[0], chunks={'time': 365 * 10})
    else:
        ds = xr.open_mfdataset(existing_paths, combine='by_coords', chunks={'time': 365 * 10})

    ds = ds.sortby('time')
    _, unique_index = np.unique(ds.time.values, return_index=True)
    if len(unique_index) != ds.sizes['time']:
        ds = ds.isel(time=np.sort(unique_index))
    return ds


def compute_climatology(da, window=121):
    """
    Compute smoothed daily climatology from the full dataset.
    Follows Wheeler & Hendon (2004) methodology.
    """
    print("  Calculating daily climatology (this takes a moment)...")
    clim = da.groupby('time.dayofyear').mean('time')
    half_window = window // 2

    # Pad cyclically to smooth across the year boundary.
    clim_padded = xr.concat(
        [
            clim.isel(dayofyear=slice(-half_window, None)),
            clim,
            clim.isel(dayofyear=slice(0, half_window)),
        ],
        dim='dayofyear'
    )
    clim_smooth = clim_padded.rolling(dayofyear=window, center=True).mean()
    clim_smooth = clim_smooth.isel(dayofyear=slice(half_window, half_window + clim.sizes['dayofyear']))
    clim_smooth = clim_smooth.assign_coords(dayofyear=clim.dayofyear)
    return clim_smooth


def normalize_olr_grid(da_olr):
    lat_name = 'lat' if 'lat' in da_olr.coords else 'latitude'
    lon_name = 'lon' if 'lon' in da_olr.coords else 'longitude'
    da_olr = da_olr.rename({lat_name: 'latitude', lon_name: 'longitude'})

    # Convert any -180..180 longitudes to 0..360 to match the GEOS convention.
    if float(da_olr.longitude.min()) < 0.0:
        da_olr = da_olr.assign_coords(longitude=(da_olr.longitude % 360)).sortby('longitude')

    # xarray interpolation expects monotonic coordinates.
    da_olr = da_olr.sortby('latitude').sortby('longitude')
    return da_olr


def load_or_build_climatology(da_olr, clim_path):
    rebuild = True
    climatology = None

    if os.path.exists(clim_path):
        try:
            print(f"  Loading cached climatology from {clim_path}...")
            climatology = xr.open_dataarray(clim_path).compute()
            rebuild = False

            expected_sizes = {
                'dayofyear': 366 if 366 in da_olr.time.dt.dayofyear.values else 365,
                'latitude': da_olr.sizes['latitude'],
                'longitude': da_olr.sizes['longitude'],
            }
            for dim, size in expected_sizes.items():
                if climatology.sizes.get(dim) != size:
                    print(f"  Cached climatology dimension mismatch on {dim}: {climatology.sizes.get(dim)} vs {size}")
                    rebuild = True
                    break

            if not rebuild:
                if not np.array_equal(climatology.latitude.values, da_olr.latitude.values):
                    print("  Cached climatology latitude grid mismatch. Rebuilding.")
                    rebuild = True
                elif not np.array_equal(climatology.longitude.values, da_olr.longitude.values):
                    print("  Cached climatology longitude grid mismatch. Rebuilding.")
                    rebuild = True
        except Exception as exc:
            print(f"  Failed to load cached climatology ({exc}). Rebuilding.")
            rebuild = True

    if rebuild:
        climatology = compute_climatology(da_olr).compute()
        print(f"  Saving climatology cache to {clim_path}...")
        climatology.to_netcdf(clim_path, mode='w')

    return climatology


def spacetime_filter(da_anom, time_dim='time', lon_dim='longitude'):
    """
    Applies the classical Wheeler-Kiladis Wavenumber-Frequency filter.
    Extracts the MJO signal:
        Eastward propagating
        Period: 30 to 90 days
        Wavenumber: 1 to 5
    """
    print("  Applying Wavenumber-Frequency (Space-Time) FFT Filter...")
    
    # Extract native numpy array [Time, Lat, Lon]
    arr = da_anom.values
    t_len, lat_len, lon_len = arr.shape
    
    # We must handle NaNs for FFT. OLR shouldn't have any over the ocean, 
    # but NOAA interpolates over land. Let's fill small gaps if any exist.
    if np.isnan(arr).any():
        print("    Warning: NaNs detected. Filling with 0 (anomaly mean) for FFT.")
        arr = np.nan_to_num(arr, nan=0.0)

    # 1. 2D FFT (Time and Longitude)
    # We loop over each latitude independently
    filtered_arr = np.zeros_like(arr, dtype=np.complex128)
    
    # Frequencies
    # Time frequency: cycles per day
    freq_t = fftfreq(t_len, d=1.0) 
    # Spatial frequency: cycles per longitude range (zonal wavenumber)
    freq_s = fftfreq(lon_len, d=1.0/lon_len) 
    
    for y in range(lat_len):
        lat_slice = arr[:, y, :] # [Time, Lon]
        
        # Apply 2D FFT
        F = fft2(lat_slice)
        
        # Create a mask for the MJO
        mask = np.zeros_like(F, dtype=bool)
        
        # Wheeler-Kiladis MJO Definition:
        # Eastward propagation means the sign of spatial and temporal frequencies must be opposite.
        # Here we define the bounds.
        # Periods: 30 to 90 days -> freq_t between 1/90 and 1/30
        min_freq_t = 1.0 / 90.0
        max_freq_t = 1.0 / 30.0
        
        # Wavenumbers: 1 to 5
        min_k = 1
        max_k = 5
        
        # Iterate over 2D frequency space
        for i_t, ft in enumerate(freq_t):
            for i_s, fs in enumerate(freq_s):
                
                # Check Eastward propagation (ft * fs < 0 in standard definition)
                # Equivalently: if time frequency is positive, wavenumber must be negative, and vice versa
                if ft > 0 and fs < 0:
                    if (min_freq_t <= ft <= max_freq_t) and (min_k <= abs(fs) <= max_k):
                        mask[i_t, i_s] = True
                elif ft < 0 and fs > 0:
                    if (min_freq_t <= abs(ft) <= max_freq_t) and (min_k <= fs <= max_k):
                        mask[i_t, i_s] = True
                        
        # Apply mask
        F_filtered = F * mask
        
        # Inverse 2D FFT
        lat_filtered = ifft2(F_filtered)
        filtered_arr[:, y, :] = lat_filtered
        
    # Take the real part (imaginary should be ~0)
    filtered_arr_real = np.real(filtered_arr)
    
    # Reconstruct xarray DataArray
    da_mjo = xr.DataArray(
        filtered_arr_real,
        coords=da_anom.coords,
        dims=da_anom.dims,
        name='mjo_wave'
    )
    
    return da_mjo


def process_mjo_wave(olr_paths, years, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    ds = load_olr_dataset(olr_paths)
    
    # NOAA uses 'olr' variable
    da_olr = ds['olr']
    
    # 0. Slice the global dataset to the relevant training period to drastically speed up Climatology
    print("Slicing base dataset to 1998-01-01 onwards for faster computation...")
    da_olr = da_olr.sel(time=slice('1998-01-01', None))
    
    da_olr = normalize_olr_grid(da_olr)
    olr_start = pd.to_datetime(da_olr.time.values[0])
    olr_end = pd.to_datetime(da_olr.time.values[-1])
    print(f"Available OLR coverage: {olr_start.date()} -> {olr_end.date()}")
    
    # 1. Compute or load climatology on the native OLR grid.
    clim_path = os.path.join(output_dir, "olr_climatology.nc")
    climatology = load_or_build_climatology(da_olr, clim_path)
    
    all_years_mjo = []
    processed_years = []
    
    target_lat = np.linspace(-90, 90, 181)
    target_lon = np.linspace(0, 359, 360)
    
    for target_year in years:
        print(f"\n--- Processing Year: {target_year} ---")
        
        # 2. We need a buffer around the target year for the 30-90 day FFT
        start_time = pd.Timestamp(f"{target_year-1}-09-01")
        end_time = pd.Timestamp(f"{target_year+1}-03-31")
        
        clipped_start = max(start_time, olr_start)
        clipped_end = min(end_time, olr_end)
        print(f"Extracting padded window ({start_time.date()} to {end_time.date()}) for FFT stability...")
        print(f"  Clipped to available OLR coverage: {clipped_start.date()} -> {clipped_end.date()}")
        if clipped_end < clipped_start:
            print(f"  No OLR data available for {target_year}. Skipping.")
            continue

        da_padded = da_olr.sel(time=slice(clipped_start, clipped_end)).compute()
        if da_padded.sizes.get('time', 0) == 0:
            print(f"  Empty padded OLR slice for {target_year}. Skipping.")
            continue
        if da_padded.sizes['time'] < 90:
            print(f"  Only {da_padded.sizes['time']} days available for {target_year}; need a longer OLR record. Skipping.")
            continue
        
        # 3. Compute anomalies on the native OLR grid.
        print("  Calculating OLR anomalies on native grid...")
        day_selector = xr.DataArray(
            da_padded.time.dt.dayofyear.values.astype(int),
            coords={'time': da_padded.time},
            dims='time'
        )
        clim_for_slice = climatology.sel(dayofyear=day_selector)
        da_anom_padded = (da_padded - clim_for_slice).rename('olr_anomaly')
        
        # 4. Apply Space-Time Wavenumber-Frequency Filter
        da_mjo_padded = spacetime_filter(da_anom_padded)
        
        # 5. Slice back to target year only
        target_start = f"{target_year}-01-01"
        target_end = f"{target_year}-12-31"
        da_mjo_year_native = da_mjo_padded.sel(time=slice(target_start, target_end))
        da_raw_year_native = da_olr.sel(time=slice(target_start, target_end)).compute()
        if da_mjo_year_native.sizes.get('time', 0) == 0:
            print(f"  No filtered MJO output available inside target year {target_year}. Skipping.")
            continue
        
        # 6. Interpolate just the 1-target-year maps to GEOS 1-degree grid
        print(f"  Interpolating filtered MJO wave to GEOS 1° grid (181x360)...")
        da_mjo_year = da_mjo_year_native.interp({'latitude': target_lat, 'longitude': target_lon}, method='linear')
        all_years_mjo.append(da_mjo_year)
        processed_years.append(target_year)
        
        # Diagnostic Plot (only for the first year processed to prevent spam)
        if PLOT_AVAILABLE and target_year == years[0]:
            print("  Generating sample diagnostic plot for the first year...")
            plot_target = pd.Timestamp(f"{target_year}-11-15")
            try:
                raw_scale = da_raw_year_native.interp({'latitude': target_lat, 'longitude': target_lon}, method='linear')
                available_times = pd.to_datetime(da_mjo_year.time.values)
                if len(available_times) == 0:
                    raise ValueError("No valid daily MJO times available for plotting.")
                nearest_idx = int(np.argmin(np.abs(available_times - plot_target)))
                plot_time = pd.Timestamp(available_times[nearest_idx])
                plot_label = plot_time.strftime("%Y-%m-%d")
                raw_slice = raw_scale.sel(time=plot_time).squeeze()
                mjo_slice = da_mjo_year.sel(time=plot_time).squeeze()
                
                fig = plt.figure(figsize=(12, 8))
                
                ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
                ax1.set_title(f"Raw NOAA Interpolated OLR ({plot_label})", fontsize=14)
                ax1.coastlines(color='white')
                p1 = ax1.contourf(target_lon, target_lat, raw_slice, transform=ccrs.PlateCarree(),
                                  levels=np.linspace(150, 300, 20), cmap='Blues_r', extend='both')
                fig.colorbar(p1, ax=ax1, orientation='vertical', pad=0.02, label='W/m²')
                
                ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree(central_longitude=180))
                ax2.set_title(f"Isolated MJO Wave (30-90 Day, Wavenumber 1-5) - {plot_label}", fontsize=14)
                ax2.coastlines()
                max_val = max(10, float(np.abs(mjo_slice).max() * 0.8))
                p2 = ax2.contourf(target_lon, target_lat, mjo_slice, transform=ccrs.PlateCarree(),
                                  levels=np.linspace(-max_val, max_val, 21), cmap='RdBu_r', extend='both')
                fig.colorbar(p2, ax=ax2, orientation='vertical', pad=0.02, label='W/m² Anomaly')
                
                plt.tight_layout()
                plot_file = os.path.join(output_dir, f"mjo_wave_diagnostic.png")
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                print(f"  ✅ Diagnostic plot saved: {plot_file}")
                plt.close()
            except Exception as e:
                print(f"  Failed to generate diagnostic plot: {e}")

    # 7. Concatenate and Save all years to a single Zarr
    print("\n==================================")
    print("Concatenating all years...")
    if not all_years_mjo:
        raise RuntimeError(
            "No MJO wave years were generated. The OLR file likely does not cover the requested dates."
        )
    da_mjo_all = xr.concat(all_years_mjo, dim='time')
    
    ds_out = da_mjo_all.to_dataset(name='mjo_wave')
    ds_out['mjo_wave'].attrs = {
        'units': 'W/m2',
        'long_name': 'MJO Wave (Wavenumber-Frequency Filtered OLR)',
        'description': 'Eastward propagating, Wavenumbers 1-5, Periods 30-90 days'
    }
    
    min_year, max_year = min(processed_years), max(processed_years)
    out_file = os.path.join(output_dir, f"mjo_wave_spatial_{min_year}_{max_year}.zarr")
    print(f"Saving merged MJO Wave map to {out_file}...")
    
    ds_out = ds_out.chunk({'time': -1, 'latitude': 181, 'longitude': 360})
    import dask
    with dask.config.set(scheduler='synchronous'):
        ds_out.to_zarr(out_file, mode='w')
    print("✅ Full pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--olr_path", type=str, default="/home1/11353/afahad/geos_subc/dataprocess/olr/olr.day.mean.nc")
    parser.add_argument("--olr_paths", type=str, nargs="+", default=None,
                        help="Optional list of OLR NetCDF files to merge into one continuous record.")
    parser.add_argument("--olr_glob", type=str, default=None,
                        help="Optional glob pattern for OLR NetCDF files, e.g. '/path/OLR-Daily*.nc'")
    parser.add_argument("--years", type=int, nargs='+', default=None)
    parser.add_argument("--start_year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end_year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--output_dir", type=str, default="/home1/11353/afahad/geos_subc/dataprocess/")
    args = parser.parse_args()

    years = args.years if args.years else list(range(args.start_year, args.end_year + 1))
    if args.olr_paths:
        olr_paths = args.olr_paths
    elif args.olr_glob:
        olr_paths = sorted(glob.glob(args.olr_glob))
        if not olr_paths:
            raise FileNotFoundError(f"No OLR files matched glob: {args.olr_glob}")
    else:
        olr_paths = [args.olr_path]

    process_mjo_wave(olr_paths, years, args.output_dir)
