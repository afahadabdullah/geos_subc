"""
IVT Weekly Processing Script
=============================
Processes daily ERA5 IVT Zarr files into weekly-mean Zarr files
aligned with GEOS S2S3 initialization dates.

Input:  Daily Zarr files (era5_ivt_{year}.zarr) created by extract_era5_ivt.py in dataprocess/era5_ivt/
Output: ivt_weekly_{year}.zarr with dims (S, L, Y, X) at GEOS 1° grid

For each GEOS init date (S dimension), we compute 4 weekly means of
OBSERVED IVT leading up to the forecast start:
    L=0 → Week -4: [S-28, S-22]
    L=1 → Week -3: [S-21, S-15]
    L=2 → Week -2: [S-14, S-8]
    L=3 → Week -1: [S-7,  S-1]

Usage:
    python dataprocess/process_ivt.py --years 2020
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import warnings

# --- Configuration ---
# Default TACC Input Path from extract_era5_ivt.py
DAILY_IVT_DIR = "/home1/11353/afahad/geos_subc/dataprocess/era5_ivt"
GEOS_DIR = "dataprocess"
OUTPUT_DIR = "dataprocess"

def process_year(year, daily_dir=DAILY_IVT_DIR, output_dir=OUTPUT_DIR):
    """
    Process ERA5 IVT for one year:
    1. Load GEOS Zarr to get init dates and target grid
    2. Load Daily IVT files (current + previous year if needed)
    3. Compute 4 weekly means before each init date
    4. Save as Zarr
    """
    out_path = os.path.join(output_dir, f"ivt_weekly_{year}.zarr")
    if os.path.exists(out_path):
        print(f"File {out_path} already exists. Skipping {year}.")
        return

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
    
    print(f"\n{'='*60}")
    print(f"Processing IVT Weekly for {year}")
    print(f"  GEOS init dates: {len(init_dates)}")
    
    # 2. Load Daily IVT files
    # We might need the previous year for early January initializations
    daily_files = [
        os.path.join(daily_dir, f"era5_ivt_{year}.zarr")
    ]
    prev_year_file = os.path.join(daily_dir, f"era5_ivt_{year-1}.zarr")
    if os.path.exists(prev_year_file):
        daily_files.insert(0, prev_year_file)
    
    valid_files = [f for f in daily_files if os.path.exists(f)]
    
    if not valid_files:
        print(f"  No daily IVT files found for {year} in {daily_dir}. run extract_era5_ivt.py first.")
        ds_geos.close()
        return
        
    print(f"  Loading daily files: {[os.path.basename(f) for f in valid_files]}")
    try:
        # Suppress warnings about large chunks or time encoding
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # daily IVT files have 'ivt' variable (magnitude)
            ds_daily = xr.open_mfdataset(valid_files, engine='zarr', combine='by_coords')
    except Exception as e:
        print(f"  Error loading daily IVT files: {e}")
        ds_geos.close()
        return

    # Check for 'ivt' variable
    var_name = 'ivt'
    if 'ivt' not in ds_daily:
        # Fallback search
        for v in ds_daily.data_vars:
            if 'ivt' in v.lower() or 'flux' in v.lower():
                var_name = v
                break
    
    if var_name not in ds_daily:
        print(f"  Variable 'ivt' not found in daily files. Found: {list(ds_daily.data_vars)}")
        ds_daily.close()
        ds_geos.close()
        return
        
    print(f"  Using variable: {var_name}")

    # 3. Compute 4 weekly means BEFORE each init date
    print(f"  Computing 4-weekly observed means...")
    
    processed_samples = []
    skipped = 0
    
    for init_date in tqdm(init_dates, desc=f"  IVT Weekly {year}"):
        weeks = []
        valid = True
        
        for w in range(4):
            # Week offset from init date (going backwards)
            # L=0 (Week -4): init - 29 to init - 22 ? No.
            # L=0 is Week -1 ? Or Week -4?
            # Standard conventions:
            # L=0 -> Week 1 Forecast
            # Here we are processing PAST observations (Lags)
            # Dataset convention:
            # L=0 -> Week -4 (Oldest)
            # L=3 -> Week -1 (Most Recent)
            
            # Week -1 ends on (init_date - 1 day)
            # Week -1 starts on (init_date - 7 days)
            
            # L=3 (Week -1): [init-7, init-1]
            # L=2 (Week -2): [init-14, init-8]
            # L=1 (Week -3): [init-21, init-15]
            # L=0 (Week -4): [init-28, init-22]
            
            # If w iterating 0..3:
            # Let's map w to correct lag index.
            # w=0 -> L=0 (Week -4)
            # w=3 -> L=3 (Week -1)
            
            # Days back for END of week:
            # w=3 -> Week -1 End: init - 1
            # w=0 -> Week -4 End: init - 1 - (3*7) = init - 22
            
            days_back_end = (3 - w) * 7 + 1
            w_end = init_date - pd.Timedelta(days=days_back_end)
            w_start = w_end - pd.Timedelta(days=6)
            
            try:
                # Select time slice
                chunk = ds_daily[var_name].sel(time=slice(w_start, w_end))
                
                # Check coverage
                if len(chunk.time) < 1: 
                    # Try broadening slice slightly if timestamps are misaligned
                    # Actually standard pandas slice is inclusive.
                    valid = False
                    break
                
                if len(chunk.time) < 7:
                    # Allowing minor missing days? Maybe 5/7 is okay?
                    # Strict: < 7 -> invalid
                    # Relaxed: < 4 -> invalid
                    if len(chunk.time) < 4:
                        valid = False
                        break
                
                # Compute mean
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
            # Fill with NaN (L, Y, X)
            # Get shape from specific humidity shape
            # Assuming (Y, X) = (181, 360) usually
            dummy_shape = ds_daily[var_name].shape[1:] # (Y, X) typically
            nan_arr = np.full((4,) + dummy_shape, np.nan, dtype=np.float32)
            
            # Create DataArray
            # Need coords from somewhere? No, we will concat later.
            # But xarray concat needs aligned coords or it fills nan.
            # Let's use first valid sample or just create dummy
            processed_samples.append(None) # Mark as None for now
            
    # Handle Nones
    # If all valid, concat. If some None, we need a template.
    if len(processed_samples) > skipped:
        # Find first valid sample
        template = next(s for s in processed_samples if s is not None)
    else:
        print(f"  No valid samples found for {year}. Skipping save.")
        return

    final_samples = []
    for s in processed_samples:
        if s is None:
            # Create NaN array with template coords
            nan_da = xr.full_like(template, np.nan)
            final_samples.append(nan_da)
        else:
            final_samples.append(s)

    if skipped > 0:
        print(f"  Warning: {skipped}/{len(init_dates)} dates had missing data (filled NaN)")

    # 4. Finalize and Save
    ds_yearly = xr.concat(final_samples, dim='S')
    ds_yearly = ds_yearly.assign_coords(S=init_dates, L=np.arange(4))
    
    # Assign name
    ds_yearly.name = 'ivt'
    ds_yearly = ds_yearly.to_dataset()
    
    # Rename coords to match GEOS convention (Y, X)
    rename_dict = {}
    if 'latitude' in ds_yearly.coords and 'Y' not in ds_yearly.coords:
        rename_dict['latitude'] = 'Y'
    if 'longitude' in ds_yearly.coords and 'X' not in ds_yearly.coords:
        rename_dict['longitude'] = 'X'
    if rename_dict:
        ds_yearly = ds_yearly.rename(rename_dict)

    out_path = os.path.join(output_dir, f"ivt_weekly_{year}.zarr")
    print(f"  Saving to {out_path}...")
    
    import dask
    with dask.config.set(scheduler='synchronous'):
        ds_yearly.to_zarr(out_path, mode='w') # standard zarr
        
    print(f"  ✓ Finished {year}: {out_path}")
    
    ds_geos.close()
    ds_daily.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Daily ERA5 IVT Zarr → Weekly Mean Zarr")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Specific years to process.")
    parser.add_argument("--daily_dir", type=str, default=DAILY_IVT_DIR,
                        help="Directory containing daily ERA5 IVT Zarr files.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory.")
    args = parser.parse_args()
    
    years = args.years if args.years else list(range(1999, 2026))
    
    for year in years:
        process_year(year, daily_dir=args.daily_dir, output_dir=args.output_dir)
    
    print("\nAll processing complete.")
