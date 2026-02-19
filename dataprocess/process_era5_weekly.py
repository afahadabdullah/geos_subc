"""
ERA5 Weekly Processing Script
=============================
Processes daily ERA5 (z500, u250) Zarr files into weekly-mean Zarr files
aligned with GEOS S2S3 initialization dates.

Input:  Daily Zarr files (era5_z500_u250_{year}.zarr) created by extract_era5.py
Output: era5_weekly_{year}.zarr with dims (S, L, Y, X) at GEOS 1° grid

For each GEOS init date (S dimension), we compute 4 weekly means of
OBSERVED ERA5 leading up to the forecast start:
    L=0 → Week -4: [S-28, S-22]
    L=1 → Week -3: [S-21, S-15]
    L=2 → Week -2: [S-14, S-8]
    L=3 → Week -1: [S-7,  S-1]

Usage:
    python dataprocess/process_era5_weekly.py --years 2020
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm

# --- Configuration ---
DAILY_ERA5_DIR = "/home1/11353/afahad/geos_subc/dataprocess/era5_z500_u250"
GEOS_DIR = "dataprocess"
OUTPUT_DIR = "dataprocess"

def process_year(year, daily_dir=DAILY_ERA5_DIR, output_dir=OUTPUT_DIR):
    """
    Process ERA5 data for one year:
    1. Load GEOS Zarr to get init dates and target grid
    2. Load Daily ERA5 files (current + previous year if needed)
    3. Compute 4 weekly means before each init date
    4. Save as Zarr
    """
    # 1. Load GEOS to get init dates and grid
    geos_path = os.path.join(GEOS_DIR, f"geos_subc_{year}.zarr")
    if not os.path.exists(geos_path):
        print(f"GEOS file not found: {geos_path}. Skipping {year}.")
        return
    
    ds_geos = xr.open_zarr(geos_path, consolidated=False)
    if 'S' not in ds_geos.dims:
        print(f"Dimension 'S' not found in {geos_path}. Skipping.")
        ds_geos.close()
        return
    
    init_dates = pd.to_datetime(ds_geos['S'].values)
    target_lat = ds_geos.coords['Y'] if 'Y' in ds_geos.coords else ds_geos.coords['latitude']
    target_lon = ds_geos.coords['X'] if 'X' in ds_geos.coords else ds_geos.coords['longitude']
    
    print(f"\n{'='*60}")
    print(f"Processing ERA5 Weekly for {year}")
    print(f"  GEOS init dates: {len(init_dates)}")
    
    # 2. Load Daily ERA5 files
    # We might need the previous year for early January initializations
    daily_files = [
        os.path.join(daily_dir, f"era5_z500_u250_{year}.zarr")
    ]
    prev_year_file = os.path.join(daily_dir, f"era5_z500_u250_{year-1}.zarr")
    if os.path.exists(prev_year_file):
        daily_files.insert(0, prev_year_file)
        
    print(f"  Loading daily files: {[os.path.basename(f) for f in daily_files]}")
    try:
        ds_daily = xr.open_mfdataset(daily_files, engine='zarr', combine='by_coords')
    except Exception as e:
        print(f"  Error loading daily ERA5 files: {e}")
        ds_geos.close()
        return

    # Variables should already be interpolated to 1-degree grid by extract_era5.py
    # and named z500 and u250 (as per the extraction script we built earlier)
    
    # 3. Compute 4 weekly means BEFORE each init date
    print(f"  Computing 4-weekly observed means...")
    
    processed_samples = []
    skipped = 0
    
    for init_date in tqdm(init_dates, desc=f"  ERA5 Weekly {year}"):
        weeks = []
        valid = True
        
        for w in range(4):
            # Week offset from init date (going backwards)
            w_end = init_date - pd.Timedelta(days=(3 - w) * 7 + 1)
            w_start = w_end - pd.Timedelta(days=6)
            
            try:
                # Select time slice
                chunk = ds_daily.sel(time=slice(w_start, w_end))
                if len(chunk.time) < 7: # Expect 7 days of daily data
                    valid = False
                    break
                
                # Compute and compute() immediately to reduce dask graph complexity
                w_mean = chunk.mean(dim='time').compute()
                weeks.append(w_mean)
            except Exception:
                valid = False
                break
        
        if valid and len(weeks) == 4:
            # Concat weeks along Lead dimension
            sample = xr.concat(weeks, dim='L')
            processed_samples.append(sample)
        else:
            skipped += 1
            # Fill with NaN if missing
            nan_ds = ds_daily.isel(time=0, drop=True).expand_dims(L=4).copy(deep=True)
            for var in nan_ds.data_vars:
                nan_ds[var].values[:] = np.nan
            processed_samples.append(nan_ds)
            
    if skipped > 0:
        print(f"  Warning: {skipped}/{len(init_dates)} dates had missing data (filled NaN)")

    # 4. Finalize and Save
    ds_yearly = xr.concat(processed_samples, dim='S')
    ds_yearly = ds_yearly.assign_coords(S=init_dates, L=np.arange(4))
    
    # Rename coords to match GEOS convention (Y, X) if they are (latitude, longitude)
    rename_dict = {}
    if 'latitude' in ds_yearly.coords and 'Y' not in ds_yearly.coords:
        rename_dict['latitude'] = 'Y'
    if 'longitude' in ds_yearly.coords and 'X' not in ds_yearly.coords:
        rename_dict['longitude'] = 'X'
    if rename_dict:
        ds_yearly = ds_yearly.rename(rename_dict)

    out_path = os.path.join(output_dir, f"era5_weekly_{year}.zarr")
    print(f"  Saving to {out_path}...")
    
    import dask
    with dask.config.set(scheduler='synchronous'):
        ds_yearly.to_zarr(out_path, mode='w', zarr_format=3)
        
    print(f"  ✓ Finished {year}: {out_path}")
    
    ds_geos.close()
    ds_daily.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Daily ERA5 Zarr → Weekly Mean Zarr")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Specific years to process.")
    parser.add_argument("--daily_dir", type=str, default=DAILY_ERA5_DIR,
                        help="Directory containing daily ERA5 Zarr files.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory.")
    args = parser.parse_args()
    
    years = args.years if args.years else list(range(1999, 2023))
    
    for year in years:
        process_year(year, daily_dir=args.daily_dir, output_dir=args.output_dir)
    
    print("\nAll processing complete.")
