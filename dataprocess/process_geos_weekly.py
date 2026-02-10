import xarray as xr
import numpy as np
import os
import shutil
import dask
from tqdm import tqdm

def process_geos_weekly(start_year=1999, end_year=2016, data_dir="dataprocess"):
    """
    Processes GEOS SubC Zarr files to 4-weekly means.
    - Input: Daily data (32 leads)
    - Output: Weekly means (4 leads: Weeks 1-4)
    - Overwrites the original file.
    """
    print(f"Processing GEOS files from {start_year} to {end_year}...")

    for year in range(start_year, end_year + 1):
        file_path = f"{data_dir}/geos_subc_{year}.zarr"
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping.")
            continue
            
        print(f"Processing {year}: {file_path}")
        
        try:
            # Load Data
            ds = xr.open_zarr(file_path, consolidated=False)
            
            # Check dimensions
            # Expecting (S, L, Y, X) or similar
            # L should be lead time in days (0..31)
            
            if 'L' not in ds.dims:
                print(f"Error: 'L' dimension not found in {file_path}. Dims: {ds.dims}")
                continue
                
            n_leads = ds.sizes['L']
            
            # CASE 1: Already processed (L=4)
            if n_leads == 4:
                print("File seems to be already processed to weekly means (L=4). Checking units...")
                ds_weekly = ds
                # Proceed to unit conversion block
                
            # CASE 2: Raw Daily Data (L >= 28)
            elif n_leads >= 28:
                # Select first 28 days (Lead 0 to 27)
                ds_28 = ds.isel(L=slice(0, 28))
                
                # Compute Weekly Means
                ds_weekly = ds_28.coarsen(L=7, boundary='exact').mean()
            else:
                 print(f"Error: Unexpected lead dimension size ({n_leads}). Skipping.")
                 continue
            
            # Unit Conversion: kg/m2/s -> mm/day
            
            # Unit Conversion: kg/m2/s -> mm/day
            # Check if mean > 1e-3 (already mm/day?) or < 1e-3 (flux)
            # Safe assumption: GEOS raw is flux.
            # But let's check a sample value to be safe, or just force if we know source.
            # Given previous check, it is definitely flux.
            print("Converting GEOS precipitation from kg/m2/s to mm/day (* 86400)...")
            if 'pr' in ds_weekly:
                ds_weekly['pr'] = ds_weekly['pr'] * 86400
            elif 'precip' in ds_weekly:
                ds_weekly['precip'] = ds_weekly['precip'] * 86400
            
            # Identify output path (Temp first)
            temp_path = f"{data_dir}/geos_subc_{year}_weekly_temp.zarr"
            
            print(f"Saving weekly means to temporary file {temp_path}...")
            
            # Use synchronous scheduler
            with dask.config.set(scheduler='synchronous'):
                ds_weekly.to_zarr(temp_path, mode='w', zarr_format=3)
            
            # Close dataset to verify release of file handles
            ds.close()
            
            # Replace original file
            print(f"Overwriting original file {file_path}...")
            shutil.rmtree(file_path)
            shutil.move(temp_path, file_path)
            
            print(f"Successfully processed {year}.")
            
        except Exception as e:
            print(f"Failed to process {year}: {e}")
            # clean up temp if exists
            if os.path.exists(f"{data_dir}/geos_subc_{year}_weekly_temp.zarr"):
                 shutil.rmtree(f"{data_dir}/geos_subc_{year}_weekly_temp.zarr")

if __name__ == "__main__":
    process_geos_weekly()
