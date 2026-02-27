
import xarray as xr
import gcsfs
import os
import pandas as pd
import numpy as np
import dask
import dask.array as da
from tqdm import tqdm
import argparse

def process_era5_olr(start_year=1999, end_year=2022, output_base_dir="/home1/11353/afahad/geos_subc/dataprocess/era5_olr"):
    """
    Extract Outgoing Longwave Radiation (OLR) from ARCO-ERA5.
    
    ERA5 stores 'top_net_thermal_radiation' (ttr) as an accumulation variable (J/m²)
    at 6-hourly intervals. OLR = -ttr (sign convention: OLR is positive outward).
    
    To convert from accumulated J/m² to instantaneous W/m² (flux):
       OLR (W/m²) = -ttr (J/m²) / accumulation_period (seconds)
       For 6-hourly data: accumulation_period = 6 * 3600 = 21600 seconds
    
    Pipeline:
       1. Load 6-hourly ttr from ARCO-ERA5 (GCS Zarr)
       2. Convert to OLR flux (W/m²)
       3. Compute daily mean
       4. Interpolate to GEOS 1-degree grid (181x360)
       5. Save as yearly Zarr files
    """
    
    # Define the GCS path to the ARCO-ERA5 Zarr store (single-level / surface variables)
    zarr_path = 'gs://gcp-public-data-arco-era5/ar/1959-2022-6h-512x256_equiangular_conservative.zarr'
    
    print(f"Connecting to {zarr_path}...")
    
    # Open the dataset lazily with Dask
    try:
        ds = xr.open_zarr(zarr_path, chunks={'time': 48, 'longitude': 256, 'latitude': 256}, consolidated=True)
        print("Dataset opened successfully.")
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return

    # Check if ttr is available
    olr_var = 'top_net_thermal_radiation'
    if olr_var not in ds:
        print(f"ERROR: Variable '{olr_var}' not found in dataset.")
        print(f"Available variables: {list(ds.data_vars)}")
        return
    
    print(f"Found '{olr_var}' in dataset.")
    print(f"  Shape: {ds[olr_var].dims}")
    print(f"  Coords: {list(ds[olr_var].coords)}")
    
    # Define GEOS 1-degree target grid
    target_lat = np.linspace(-90, 90, 181)
    target_lon = np.linspace(0, 359, 360)
    
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # ERA5 accumulation period for 6-hourly data (seconds)
    accum_period_seconds = 6 * 3600  # 21600 seconds
    
    for year in range(start_year, end_year + 1):
        output_path = os.path.join(output_base_dir, f"era5_olr_{year}.zarr")
        
        if os.path.exists(output_path):
            print(f"Skipping {year}, file already exists at {output_path}")
            continue
            
        print(f"\n--- Processing Year: {year} ---")
        
        try:
            # 1. Select year and variable
            ttr = ds[olr_var].sel(time=str(year))
            
            # 2. Convert accumulated ttr (J/m²) to OLR flux (W/m²)
            # OLR = -ttr / accumulation_period
            # ttr is negative in ERA5 convention (energy leaving the system)
            # OLR is positive (outgoing radiation)
            olr = -1.0 * ttr / accum_period_seconds
            olr.name = 'olr'
            
            ds_olr = olr.to_dataset(name='olr')
            ds_olr['olr'].attrs = {
                'units': 'W/m²',
                'long_name': 'Outgoing Longwave Radiation',
                'description': 'Derived from ERA5 top_net_thermal_radiation: OLR = -ttr / 21600'
            }
            
            # 3. Daily Mean Calculation
            print(f"Calculating daily means for {year}...")
            ds_daily = ds_olr.resample(time='1D').mean()
            
            # 4. Interpolation to GEOS 1-degree grid
            print(f"Interpolating to GEOS 1-degree grid (181x360)...")
            ds_interp = ds_daily.interp(latitude=target_lat, longitude=target_lon, method='linear')
            
            # 5. Save to Zarr
            print(f"Saving to {output_path}...")
            # Rechunk for efficient access
            ds_interp = ds_interp.chunk({'time': -1, 'latitude': 181, 'longitude': 360})
            
            with dask.config.set(scheduler='synchronous'):
                ds_interp.to_zarr(output_path, mode='w')
            
            print(f"✅ Successfully saved OLR for {year}.")
            
        except Exception as e:
            print(f"Error processing {year}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ERA5 OLR (Outgoing Longwave Radiation) to daily 1-degree GEOS grid.")
    parser.add_argument("--start_year", type=int, default=1999)
    parser.add_argument("--end_year", type=int, default=2022)
    parser.add_argument("--output_dir", type=str, default="/home1/11353/afahad/geos_subc/dataprocess/era5_olr")
    args = parser.parse_args()
    
    process_era5_olr(start_year=args.start_year, end_year=args.end_year, output_base_dir=args.output_dir)
