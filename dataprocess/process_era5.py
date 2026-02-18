"""
ERA5 Processing Script
======================
Processes ARCO-ERA5 data into weekly-mean Zarr files aligned with 
GEOS S2S3 initialization dates.

Variables:
    - z500: Geopotential at 500 hPa
    - u200: U-component of wind at 200 hPa

Output: era5_weekly_{year}.zarr with dims (S, L, Y, X) at GEOS 1° grid

For each GEOS init date (S dimension), we compute 4 weekly means of
OBSERVED ERA5 leading up to the forecast start:
    L=0 → Week -4: [S-28, S-22]
    L=1 → Week -3: [S-21, S-15]
    L=2 → Week -2: [S-14, S-8]
    L=3 → Week -1: [S-7,  S-1]

Usage:
    python dataprocess/process_era5.py --years 1999 2000
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
import dask
from tqdm import tqdm
import argparse

# --- Configuration ---
GEOS_DIR = "dataprocess"
OUTPUT_DIR = "dataprocess/era5_weekly"
ZARR_PATH = 'gs://gcp-public-data-arco-era5/ar/1959-2022-6h-512x256_equiangular_conservative.zarr'

def process_year(year, ds_era5, output_dir=OUTPUT_DIR):
    """
    Process ERA5 data for one year:
    1. Load GEOS Zarr to get init dates and target grid
    2. Compute 4 weekly means before each init date
    3. Regrid ERA5 to GEOS 1° grid
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
    target_lat = ds_geos.coords['Y'] if 'Y' in ds_geos.coords else ds_geos.coords['lat']
    target_lon = ds_geos.coords['X'] if 'X' in ds_geos.coords else ds_geos.coords['lon']
    
    print(f"\n{'='*60}")
    print(f"Processing ERA5 for {year}")
    print(f"  GEOS init dates: {len(init_dates)}")
    
    # 2. Extract Z500 and U200
    print(f"  Selecting levels (z500, u200)...")
    z500 = ds_era5['geopotential'].sel(level=500).drop_vars('level').rename('z500')
    u200 = ds_era5['u_component_of_wind'].sel(level=200).drop_vars('level').rename('u200')
    
    ds_subset = xr.merge([z500, u200])
    
    # 3. Compute 4 weekly means BEFORE each init date
    print(f"  Computing 4-weekly observed means for {len(init_dates)} init dates...")
    
    processed_samples = []
    
    for init_date in tqdm(init_dates, desc=f"  ERA5 {year}"):
        weeks = []
        for w in range(4):
            # Week offset from init date (going backwards)
            w_end = init_date - pd.Timedelta(days=(3 - w) * 7 + 1)
            w_start = w_end - pd.Timedelta(days=6)
            
            # ERA5 is 6-hourly, so slice will contain multiple points per day
            try:
                # Use sel with slice and then mean across time
                chunk = ds_subset.sel(time=slice(w_start, w_end))
                # Compute mean lazily
                w_mean = chunk.mean(dim='time')
                weeks.append(w_mean)
            except Exception as e:
                print(f"  Error selecting {w_start} to {w_end}: {e}")
                # Fill with NaN if missing (shouldn't happen for ERA5 usually)
                weeks.append(ds_subset.isel(time=0).where(False))
        
        # Concat along L dimension
        sample = xr.concat(weeks, dim='L')
        processed_samples.append(sample)
    
    # 4. Concatenate all samples into S dimension
    ds_yearly = xr.concat(processed_samples, dim='S')
    ds_yearly = ds_yearly.assign_coords(S=init_dates, L=np.arange(4))
    
    # 5. Regrid to GEOS 1° grid
    print(f"  Interpolating to GEOS 1° grid...")
    # ERA5 coordinates: latitude, longitude
    ds_interp = ds_yearly.interp(
        latitude=target_lat, 
        longitude=target_lon, 
        method='linear'
    )
    
    # 6. Save as Zarr
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"era5_weekly_{year}.zarr")
    print(f"  Saving to {out_path}...")
    
    # Rechunk for Zarr stability
    ds_interp = ds_interp.chunk({'S': 1, 'L': 4, 'latitude': 181, 'longitude': 360})
    
    with dask.config.set(scheduler='synchronous'):
        ds_interp.to_zarr(out_path, mode='w')
    
    print(f"  ✓ Finished {year}: {out_path}")
    
    ds_geos.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ARCO-ERA5 to weekly Zarr aligned with GEOS.")
    parser.add_argument("--years", type=int, nargs="+", default=[1999])
    parser.add_argument("--output_dir", type=str, default="/home1/11353/afahad/geos_subc/dataprocess/era5_weekly")
    args = parser.parse_args()
    
    print(f"Connecting to {ZARR_PATH}...")
    # Open dataset once
    # ERA5 ARCO is large, use appropriate chunks
    ds_era5 = xr.open_zarr(ZARR_PATH, chunks={'time': 240, 'longitude': 256, 'latitude': 256}, consolidated=True)
    
    for year in args.years:
        process_year(year, ds_era5, output_dir=args.output_dir)
    
    print("\nAll processing complete.")
