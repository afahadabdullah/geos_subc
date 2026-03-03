import xarray as xr
import numpy as np
import os
import shutil
import pandas as pd
import dask
from tqdm import tqdm
from dask.diagnostics import ProgressBar

def clean_year(year, data_dir="dataprocess"):
    """
    Scans the GEOS Zarr for a given year to find valid (non-NaN) initialization dates.
    Then subsets ALL corresponding weekly Zarr files to only include those valid dates,
    drastically shrinking the dataset size and stripping empty placeholders.
    """
    print(f"\n======================================")
    print(f"Cleaning Year {year}")
    print(f"======================================")
    
    geos_path = os.path.join(data_dir, f"geos_subc_{year}.zarr")
    if not os.path.exists(geos_path):
        print(f"GEOS file not found: {geos_path}. Skipping year.")
        return

    # 1. Open GEOS logic to find valid dates
    try:
        ds_geos = xr.open_zarr(geos_path, consolidated=False)
        pr_var = next((v for v in ['pr', 'precip', 'PRECTOT', 'flux_precip'] if v in ds_geos), None)
        
        if not pr_var:
            print(f"Error: Could not find precipitation variable in {geos_path}")
            return
            
        print(f"Scanning {ds_geos.sizes['S']} initialization dates in {geos_path}...")
        
        valid_dates = []
        # Get dimensions excluding S to select a single pixel for the NaN probe
        dims_to_select = {d: 0 for d in ds_geos[pr_var].dims if d != 'S'}
        
        for s_idx in range(ds_geos.sizes['S']):
            probe_val = ds_geos[pr_var].isel(S=s_idx, **dims_to_select).values
            if not np.isnan(probe_val):
                valid_dates.append(ds_geos['S'].values[s_idx])
                
        ds_geos.close()
    except Exception as e:
        print(f"Error scanning {geos_path}: {e}")
        return
        
    if not valid_dates:
        print(f"CRITICAL: No valid dates found in {year}! Skipping.")
        return
        
    print(f"Found {len(valid_dates)} valid initialization dates out of {ds_geos.sizes['S']}.")
    
    # 2. List of all Zarr files to filter for this year
    zarr_files = [
        f"geos_subc_{year}.zarr",
        f"gpcp_weekly_{year}.zarr",
        f"sst_weekly_{year}.zarr",
        f"sss_weekly_{year}.zarr",
        f"soilw_weekly_{year}.zarr",
        f"ivt_weekly_{year}.zarr",
        f"mjowave_weekly_{year}.zarr",
        f"z500_u250_weekly_{year}.zarr"
    ]
    
    # 3. Filter and overwrite each dataset
    for zarr_name in zarr_files:
        path = os.path.join(data_dir, zarr_name)
        if not os.path.exists(path):
            continue
            
        temp_path = f"{path}_temp"
        print(f"\nFiltering: {zarr_name}")
        
        try:
            ds = xr.open_zarr(path, consolidated=False)
            
            # Check if dataset has 'S' dimension
            if 'S' not in ds.dims:
                print(f"  Warning: No 'S' dimension in {zarr_name}. Skipping filtering.")
                ds.close()
                continue
                
            # Check if already filtered (sizes match)
            if ds.sizes['S'] == len(valid_dates):
                print(f"  Already filtered ({ds.sizes['S']} dates). Skipping.")
                ds.close()
                continue
                
            # Filter the dataset using the valid dates
            print(f"  Subsetting from {ds.sizes['S']} to {len(valid_dates)} dates...")
            ds_filtered = ds.sel(S=valid_dates)
            
            # Reset encoding to prevent metadata conflicts during write
            ds_filtered.encoding = {}
            for var in ds_filtered.variables:
                ds_filtered[var].encoding = {}
                
            # Save to temporary path
            with ProgressBar(), dask.config.set(scheduler='synchronous'):
                ds_filtered.to_zarr(temp_path, mode='w')
                
            ds.close()
            
            # Replace original with filtered version
            shutil.rmtree(path)
            os.rename(temp_path, path)
            print(f"  Success: Replaced {zarr_name} with filtered subset.")
            
        except Exception as e:
            print(f"  Error filtering {zarr_name}: {e}")
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean FIMR Zarr stores of NaN dates")
    parser.add_argument("--start", type=int, default=2017)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--dir", type=str, default="dataprocess")
    args = parser.parse_args()
    
    print(f"Starting cleanup of {args.dir} for years {args.start}-{args.end}")
    for y in range(args.start, args.end + 1):
        clean_year(y, data_dir=args.dir)
    print("\nAll target years cleaned!")
