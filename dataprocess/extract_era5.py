import xarray as xr
import gcsfs
import os
import pandas as pd
import numpy as np
import dask
from tqdm import tqdm
import argparse

def process_era5_yearly(start_year=1999, end_year=2022, output_base_dir="/home1/11353/afahad/geos_subc/dataprocess/era5_z500_u250"):
    # Define the GCS path to the Zarr store
    zarr_path = 'gs://gcp-public-data-arco-era5/ar/1959-2022-6h-512x256_equiangular_conservative.zarr'
    
    print(f"Connecting to {zarr_path}...")
    
    # Open the dataset lazily with Dask
    try:
        ds = xr.open_zarr(zarr_path, chunks={'time': 240, 'longitude': 256, 'latitude': 256}, consolidated=True)
        print("Dataset opened successfully.")
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return

    # Define variable names
    z_var = 'geopotential'
    u_var = 'u_component_of_wind'
    level_coord = 'level'
    
    # Define GEOS 1-degree target grid
    # GEOS grid is typically 1 degree: Lat 181 (-90 to 90), Lon 360 (0 to 359)
    target_lat = np.linspace(-90, 90, 181)
    target_lon = np.linspace(0, 359, 360)
    
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    for year in range(start_year, end_year + 1):
        output_path = os.path.join(output_base_dir, f"era5_z500_u250_{year}.zarr")
        
        if os.path.exists(output_path):
            print(f"Skipping {year}, file already exists at {output_path}")
            continue
            
        print(f"\n--- Processing Year: {year} ---")
        
        try:
            # 1. Select year and variables/levels
            # Slicing time for the year
            ds_year = ds.sel(time=str(year))
            
            # Select specific variables and levels
            # Geopotential at 500 hPa and U at 250 hPa
            z500 = ds_year[z_var].sel({level_coord: 500}).rename('z500')
            u250 = ds_year[u_var].sel({level_coord: 250}).rename('u250')
            
            # Combine into a temporary dataset for processing
            ds_subset = xr.merge([z500.drop_vars(level_coord), u250.drop_vars(level_coord)])
            
            # 2. Daily Mean Calculation
            print(f"Calculating daily means for {year}...")
            # We resample by 'D' (day)
            ds_daily = ds_subset.resample(time='1D').mean()
            
            # 3. Interpolation to GEOS 1-degree grid
            print(f"Interpolating to GEOS 1-degree grid (181x360)...")
            # Rename coordinates to match target if needed (ERA5: latitude/longitude)
            # ds_daily already has latitude/longitude from ARCO-ERA5
            ds_interp = ds_daily.interp(latitude=target_lat, longitude=target_lon, method='linear')
            
            # 4. Save to Zarr
            print(f"Saving to {output_path}...")
            # Using synchronous scheduler for stability during save
            with dask.config.set(scheduler='synchronous'):
                ds_interp.to_zarr(output_path, mode='w')
            
            print(f"Successfully saved {year}.")
            
        except Exception as e:
            print(f"Error processing {year}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ARCO-ERA5 6-hourly to daily 1-degree GEOS grid.")
    parser.add_argument("--start_year", type=int, default=1999)
    parser.add_argument("--end_year", type=int, default=2022)
    parser.add_argument("--output_dir", type=str, default="/home1/11353/afahad/geos_subc/dataprocess/era5_z500_u250")
    args = parser.parse_args()
    
    process_era5_yearly(start_year=args.start_year, end_year=args.end_year, output_base_dir=args.output_dir)
