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
            if n_leads < 28:
                print(f"Error: Not enough lead times ({n_leads}) for 4 weeks processing.")
                continue
                
            # Select first 28 days (Lead 0 to 27)
            # Assuming L=0 is Day 1, L=1 is Day 2, etc. (or Day 0..27)
            ds_28 = ds.isel(L=slice(0, 28))
            
            # Compute Weekly Means
            # We want to group L into 4 chunks of 7
            # Coarsen / Resample
            # ds.coarsen(L=7).mean() is perfect for this
            
            ds_weekly = ds_28.coarsen(L=7, boundary='exact').mean()
            
            # Rename L to simple 0..3 index? It will be handled by coarsen automatically?
            # Coarsen reduces the dimension size.
            # L will become size 4.
            
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
