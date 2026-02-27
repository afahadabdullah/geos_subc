
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


def compute_climatology(da, window=121):
    """
    Compute smoothed daily climatology from the full dataset.
    Follows Wheeler & Hendon (2004) methodology.
    """
    print("  Calculating daily climatology (this takes a moment)...")
    # Group by day of year and compute mean
    clim = da.groupby('time.dayofyear').mean('time')
    
    # Pad for rolling mean to prevent edge effects at year boundary
    clim_padded = xr.concat([clim[-window//2:], clim, clim[:window//2]], dim='dayofyear')
    clim_smooth = clim_padded.rolling(dayofyear=window, center=True).mean()[window//2:-window//2]
    
    # Map back to original time coordinates
    # We must explicitly cast dayofyear to int to avoid xarray typing issues
    day_indices = da.time.dt.dayofyear.values.astype(int)
    climatology_timeseries = clim_smooth.sel(dayofyear=day_indices)
    
    # Drop the dayofyear coord
    climatology_timeseries = climatology_timeseries.drop_vars('dayofyear')
    return climatology_timeseries


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


def process_mjo_wave(olr_path, target_year, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading NOAA OLR dataset from {olr_path}...")
    # Chunk by time to prevent memory exhaustion
    ds = xr.open_dataset(olr_path, chunks={'time': 365 * 10})
    
    # NOAA uses 'olr' variable
    da_olr = ds['olr']
    
    # Ensure lat/lon are standard
    lat_name = 'lat' if 'lat' in da_olr.coords else 'latitude'
    lon_name = 'lon' if 'lon' in da_olr.coords else 'longitude'
    da_olr = da_olr.rename({lat_name: 'latitude', lon_name: 'longitude'})
    
    # 1. Compute or Load Climatology using the entire record on native 2.5° grid
    # This is MUCH faster and uses much less memory than interpolating 48 years first
    clim_path = os.path.join(output_dir, "olr_climatology.nc")
    if os.path.exists(clim_path):
        print(f"  Loading cached climatology from {clim_path}...")
        climatology = xr.open_dataarray(clim_path).compute()
    else:
        climatology = compute_climatology(da_olr).compute()
        print(f"  Saving climatology cache to {clim_path}...")
        climatology.to_netcdf(clim_path)
    
    # 2. We need a buffer around the target year for the 30-90 day FFT
    # Standard practice is to pad by at least 90 days on either side
    start_time = f"{target_year-1}-09-01"
    end_time = f"{target_year+1}-03-31"
    
    print(f"Extracting padded window ({start_time} to {end_time}) for FFT stability...")
    da_padded = da_olr.sel(time=slice(start_time, end_time)).compute()
    
    # 3. Compute Anomalies FOR THE SLICE ONLY
    # This avoids a massive global broadcast that triggers Dask MemoryErrors
    print("  Calculating OLR Anomalies on native 2.5° grid (padded slice)...")
    da_anom_padded = da_padded.groupby('time.dayofyear') - climatology
    
    # Drop the added dayofyear coordinate if it exists
    if 'dayofyear' in da_anom_padded.coords:
        da_anom_padded = da_anom_padded.drop_vars('dayofyear')
    
    # 4. Apply Space-Time Wavenumber-Frequency Filter (on native 2.5° grid)
    da_mjo_padded = spacetime_filter(da_anom_padded)
    
    # 5. Slice back to target year only
    target_start = f"{target_year}-01-01"
    target_end = f"{target_year}-12-31"
    da_mjo_year_native = da_mjo_padded.sel(time=slice(target_start, target_end))
    da_raw_year_native = da_olr.sel(time=slice(target_start, target_end)).compute()
    
    # 6. Interpolate just the 1-target-year maps to GEOS 1-degree grid
    print("\nInterpolating filtered MJO wave and raw data to GEOS 1° grid (181x360)...")
    target_lat = np.linspace(-90, 90, 181)
    target_lon = np.linspace(0, 359, 360)
    
    da_mjo_year = da_mjo_year_native.interp({'latitude': target_lat, 'longitude': target_lon}, method='linear')
    da_raw_year = da_raw_year_native.interp({'latitude': target_lat, 'longitude': target_lon}, method='linear')
    
    # 7. Save to Zarr
    out_file = os.path.join(output_dir, f"mjo_wave_spatial_{target_year}.zarr")
    print(f"Saving MJO Wave map to {out_file}...")
    
    ds_out = da_mjo_year.to_dataset(name='mjo_wave')
    ds_out['mjo_wave'].attrs = {
        'units': 'W/m2',
        'long_name': 'MJO Wave (Wavenumber-Frequency Filtered OLR)',
        'description': 'Eastward propagating, Wavenumbers 1-5, Periods 30-90 days'
    }
    
    ds_out = ds_out.chunk({'time': -1, 'latitude': 181, 'longitude': 360})
    import dask
    with dask.config.set(scheduler='synchronous'):
        ds_out.to_zarr(out_file, mode='w')
    print("✅ Zarr save complete.")
    
    # 8. Diagnostic Plot
    if PLOT_AVAILABLE:
        # Pick a random day in the middle of a strong MJO season (e.g., Nov-Dec)
        plot_date = f"{target_year}-11-15"
        try:
            raw_slice = da_raw_year.sel(time=plot_date).squeeze()
            mjo_slice = da_mjo_year.sel(time=plot_date).squeeze()
            
            fig = plt.figure(figsize=(12, 8))
            
            # Raw OLR
            ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
            ax1.set_title(f"Raw NOAA Interpolated OLR ({plot_date})", fontsize=14)
            ax1.coastlines(color='white')
            # OLR is usually plotted reversed (blues for low OLR/high clouds)
            p1 = ax1.contourf(target_lon, target_lat, raw_slice, transform=ccrs.PlateCarree(),
                              levels=np.linspace(150, 300, 20), cmap='Blues_r', extend='both')
            fig.colorbar(p1, ax=ax1, orientation='vertical', pad=0.02, label='W/m²')
            
            # MJO Filtered Wave
            ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree(central_longitude=180))
            ax2.set_title(f"Isolated MJO Wave (30-90 Day, Wavenumber 1-5) - {plot_date}", fontsize=14)
            ax2.coastlines()
            # Anomalies: red positive (suppressed), blue negative (enhanced convection)
            max_val = max(10, float(np.abs(mjo_slice).max() * 0.8)) # Scale nicely
            p2 = ax2.contourf(target_lon, target_lat, mjo_slice, transform=ccrs.PlateCarree(),
                              levels=np.linspace(-max_val, max_val, 21), cmap='RdBu_r', extend='both')
            fig.colorbar(p2, ax=ax2, orientation='vertical', pad=0.02, label='W/m² Anomaly')
            
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f"mjo_wave_diagnostic_{target_year}.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"✅ Diagnostic plot saved: {plot_file}")
            plt.close()
        except Exception as e:
            print(f"Failed to generate diagnostic plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--olr_path", type=str, default="/home1/11353/afahad/geos_subc/dataprocess/olr/olr.day.mean.nc")
    parser.add_argument("--year", type=int, default=1999)
    parser.add_argument("--output_dir", type=str, default="/home1/11353/afahad/geos_subc/dataprocess/")
    args = parser.parse_args()
    
    assert os.path.exists(args.olr_path), f"OLR file not found at {args.olr_path}"
    
    process_mjo_wave(args.olr_path, args.year, args.output_dir)
