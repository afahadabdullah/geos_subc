"""
Extract daily ERA5 Z500/U250 on a GEOS-like 1 degree grid.

This is the first stage of the legacy two-step Z500/U250 input pipeline:
1. ``extract_era5.py`` writes daily ``era5_z500_u250_{year}.zarr``
2. ``process_z500_u250.py`` converts those daily files into the 4 trailing
   observed weekly means before each GEOS init date.

By default this script uses the newer ARCO ERA5 analysis-ready v3 store:
``gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3``
"""

import xarray as xr
import os
import numpy as np
import dask
import argparse
import traceback

DEFAULT_ZARR_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

def process_era5_yearly(
    start_year=2023,
    end_year=2025,
    output_base_dir="/home1/11353/afahad/geos_subc/dataprocess/era5_z500_u250",
    overwrite=False,
    zarr_path=DEFAULT_ZARR_PATH,
):
    
    print(f"Connecting to {zarr_path}...")
    
    # Follow the official ARCO access pattern for the v3 store.
    try:
        ds = xr.open_zarr(
            zarr_path,
            chunks=None,
            storage_options={"token": "anon"},
        )
        if "valid_time_start" in ds.attrs and "valid_time_stop" in ds.attrs and "time" in ds.coords:
            ds = ds.sel(time=slice(ds.attrs["valid_time_start"], ds.attrs["valid_time_stop"]))
        print("Dataset opened successfully.")
    except Exception as e:
        print(f"Error opening dataset: {type(e).__name__}: {e!r}")
        traceback.print_exc()
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
            if not overwrite:
                print(f"Skipping {year}, file already exists at {output_path}")
                continue
            print(f"Overwriting existing daily file: {output_path}")
            import shutil
            shutil.rmtree(output_path)
            
        print(f"\n--- Processing Year: {year} ---")
        
        try:
            # 1. Select year and variables/levels
            # Slicing time for the year
            ds_year = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
            if len(ds_year.time) == 0:
                print(f"Warning: No data found for {year} in ARCO ERA5. Skipping.")
                continue
            
            # Select specific variables and levels
            # Geopotential at 500 hPa and U at 250 hPa
            z500 = ds_year[z_var].sel({level_coord: 500}).rename('z500')
            u250 = ds_year[u_var].sel({level_coord: 250}).rename('u250')
            
            # Combine into a temporary dataset for processing
            ds_subset = xr.merge([z500.drop_vars(level_coord), u250.drop_vars(level_coord)])
            ds_subset = ds_subset.chunk({'time': 48, 'latitude': 721, 'longitude': 1440})
            
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
            # Rechunk to ensure uniform sizes for Zarr
            ds_interp = ds_interp.chunk({'time': -1, 'latitude': 181, 'longitude': 360})
            
            # Using synchronous scheduler for stability during save
            with dask.config.set(scheduler='synchronous'):
                ds_interp.to_zarr(output_path, mode='w')
            
            print(f"Successfully saved {year}.")
            
        except Exception as e:
            print(f"Error processing {year}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ARCO-ERA5 6-hourly to daily 1-degree GEOS grid.")
    parser.add_argument("--start_year", type=int, default=2023)
    parser.add_argument("--end_year", type=int, default=2025)
    parser.add_argument("--output_dir", type=str, default="/home1/11353/afahad/geos_subc/dataprocess/era5_z500_u250")
    parser.add_argument("--zarr_path", type=str, default=DEFAULT_ZARR_PATH,
                        help="ARCO ERA5 Zarr source. Defaults to the public v3 full 37-variable hourly store.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing era5_z500_u250_<year>.zarr files.")
    args = parser.parse_args()
    
    process_era5_yearly(
        start_year=args.start_year,
        end_year=args.end_year,
        output_base_dir=args.output_dir,
        overwrite=args.overwrite,
        zarr_path=args.zarr_path,
    )
