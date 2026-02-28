import xarray as xr
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm

def process_year(year, output_dir="dataprocess"):
    out_path = f"{output_dir}/gpcp_weekly_{year}.zarr"
    if os.path.exists(out_path):
        print(f"File {out_path} already exists. Skipping {year}.")
        return

    # 1. Load GEOS Forecast to get Init dates and Grid
    geos_path = f"dataprocess/geos_subc_{year}.zarr"
    if not os.path.exists(geos_path):
        print(f"GEOS file not found: {geos_path}. Skipping {year}.")
        return

    ds_geos = xr.open_zarr(geos_path, consolidated=False)
    
    # Identify Init Dimension
    if 'S' in ds_geos.dims:
        init_dates = pd.to_datetime(ds_geos['S'].values)
    else:
        print(f"Dimension 'S' not found in {geos_path}")
        return

    # Target Grid (Lat/Lon)
    # We want GPCP on this grid
    target_lat = ds_geos.coords['Y'] if 'Y' in ds_geos.coords else ds_geos.coords['lat']
    target_lon = ds_geos.coords['X'] if 'X' in ds_geos.coords else ds_geos.coords['lon']
    
    # 2. Load GPCP Data (Current Year + Next Year for overlap)
    # GPCP files are in dataprocess/gpcp_data/{year}/...
    gpcp_files = sorted(glob.glob(f"dataprocess/gpcp_data/{year}/*.nc"))
    next_year_files = sorted(glob.glob(f"dataprocess/gpcp_data/{year+1}/*.nc"))
    
    if not gpcp_files:
        print(f"No GPCP files found for {year}")
        return

    print(f"Loading GPCP raw data for {year} (and {year+1})...")
    # Load with Xarray
    # We use 'precip' variable usually in GPCP
    try:
        ds_gpcp = xr.open_mfdataset(gpcp_files + next_year_files, combine='by_coords')
    except Exception as e:
        print(f"Error loading GPCP netcdfs: {e}")
        return
        
    # Rename GPCP vars to match if needed (usually 'precip')
    if 'precip' in ds_gpcp:
        da_precip = ds_gpcp['precip']
    elif 'p' in ds_gpcp:
        da_precip = ds_gpcp['p']
    else:
        print("Precipitation variable not found in GPCP (checked 'precip', 'p')")
        return

    # 3. Regrid GPCP to GEOS Grid
    # Using simple interpolation
    print(f"Regridding GPCP to GEOS grid...")
    # Ensure coords are named consistently for interpolation
    # GPCP likely has 'latitude', 'longitude' vs GEOS 'Y', 'X'
    # We need to rename GPCP coords to match target or vice versa
    # Let's assume standard lat/lon mapping
    
    # Check GPCP coords
    gpcp_lat_name = 'latitude' if 'latitude' in da_precip.coords else 'lat'
    gpcp_lon_name = 'longitude' if 'longitude' in da_precip.coords else 'lon'
    
    # Rename for interp if keys don't match
    rename_dict = {}
    if gpcp_lat_name != target_lat.name: rename_dict[gpcp_lat_name] = target_lat.name
    if gpcp_lon_name != target_lon.name: rename_dict[gpcp_lon_name] = target_lon.name
    
    if rename_dict:
        da_precip = da_precip.rename(rename_dict)

    # Interpolate
    da_precip_interp = da_precip.interp({target_lat.name: target_lat, target_lon.name: target_lon}, method='linear')
    
    # 4. Compute Weekly Means for each Init Date
    print(f"Computing 4-weekly means for {len(init_dates)} init dates...")
    
    processed_data = []
    
    for init_date in tqdm(init_dates):
        # We want 4 weeks starting from init_date
        # W1: [S, S+6], W2: [S+7, S+13], ...
        # Total 28 days
        start_ts = init_date
        end_ts = init_date + pd.Timedelta(days=27)
        
        # Select time slice
        # Note: GPCP time coord name must match, usually 'time'
        try:
             # Slice 28 days
             # We assume da_precip_interp has 'time' dimension
             chunk = da_precip_interp.sel(time=slice(start_ts, end_ts))
             
             if len(chunk.time) < 28:
                 # Check if missing data (e.g. end of 2016 without 2017 data)
                 # Pad or fill NaN?
                 # If slightly less, maybe ok? But strict 28 is better.
                 # Let's fill with NaNs if missing
                 pass
             
             # Resample to weekly or just manual aggregation
             # Manual is safer for exact offsets
             weeks = []
             for w in range(4):
                 w_start = start_ts + pd.Timedelta(days=w*7)
                 w_end = w_start + pd.Timedelta(days=6)
                 w_mean = chunk.sel(time=slice(w_start, w_end)).mean(dim='time')
                 weeks.append(w_mean)
             
             # Stack 4 weeks: (4, Lat, Lon)
             sample = xr.concat(weeks, dim='L') # New dimension 'L' for Lead
             sample = sample.assign_coords(L=np.arange(4))
             processed_data.append(sample)
             
        except KeyError:
            print(f"Time slice missing given indices: {start_ts} to {end_ts}")
            # Append NaN sample?
            nan_sample = xr.full_like(da_precip_interp.isel(time=slice(0, 4)), np.nan)
            processed_data.append(nan_sample)

    # 5. Stack all Initializations: (S, L, Lat, Lon)
    ds_out = xr.concat(processed_data, dim='S')
    ds_out = ds_out.assign_coords(S=init_dates)
    
    # Save
    out_path = f"{output_dir}/gpcp_weekly_{year}.zarr"
    print(f"Saving to {out_path}...")
    import dask
    with dask.config.set(scheduler='synchronous'):
        ds_out.to_dataset(name='precip').to_zarr(out_path, mode='w', zarr_format=3)
    print(f"Finished {year}.")

if __name__ == "__main__":
    for year in range(1999, 2026):
        process_year(year)
