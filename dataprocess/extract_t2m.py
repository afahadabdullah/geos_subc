"""
Extract daily ERA5 T2M on a GEOS-like 1 degree grid.

This is the first stage of the legacy two-step T2M target pipeline:
1. ``extract_t2m.py`` writes daily ``era5_t2m_{year}.zarr``
2. ``process_t2m_weekly.py`` converts those daily files into weekly lead targets
   aligned with GEOS init dates.

By default this script uses the newer ARCO ERA5 analysis-ready v3 store:
``gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3``

The script clips to the stable ERA5 range advertised by the store metadata
(`valid_time_start`, `valid_time_stop`) when those attributes are present.
"""

import xarray as xr
import os
import numpy as np
import dask
import argparse

DEFAULT_ZARR_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def process_era5_t2m_yearly(
    start_year=2022,
    end_year=2025,
    output_base_dir="/home1/11353/afahad/geos_subc/dataprocess/era5_t2m",
    overwrite=False,
    zarr_path=DEFAULT_ZARR_PATH,
):
    
    print(f"Connecting to {zarr_path}...")
    
    # Open the dataset lazily with Dask (ignoring chunks warnings if any)
    try:
        ds = xr.open_zarr(
            zarr_path,
            chunks={'time': 48, 'longitude': 256, 'latitude': 256},
            storage_options={'token': 'anon'},
        )
        if "valid_time_start" in ds.attrs and "valid_time_stop" in ds.attrs and "time" in ds.coords:
            ds = ds.sel(time=slice(ds.attrs["valid_time_start"], ds.attrs["valid_time_stop"]))
        print("Dataset opened successfully.")
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return

    # Define variable names (ERA5 2-meter temperature)
    var_name = '2m_temperature'
    
    if var_name not in ds:
        print(f"Error: {var_name} not found in ARCO dataset. Available vars: {list(ds.data_vars)}")
        return
        
    # Define GEOS 1-degree target grid
    target_lat = np.linspace(-90, 90, 181)
    target_lon = np.linspace(0, 359, 360)
    
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    for year in range(start_year, end_year + 1):
        output_path = os.path.join(output_base_dir, f"era5_t2m_{year}.zarr")
        
        if os.path.exists(output_path):
            if not overwrite:
                print(f"Skipping {year}, file already exists at {output_path}")
                continue
            print(f"Overwriting existing daily file: {output_path}")
            import shutil
            shutil.rmtree(output_path)
            
        print(f"\n--- Processing Year: {year} ---")
        
        try:
            # Slicing time for the year using a robust datetime slice
            ds_year = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
            
            if len(ds_year.time) == 0:
                print(f"Warning: No data found for {year} in ARCO-ERA5 bucket. Skipping.")
                continue
            
            # Extract T2M (no level dimension needed for 2m_temperature)
            t2m = ds_year[var_name].rename('t2m')
            
            # Combine into a temporary dataset (just a single variable here)
            ds_subset = t2m.to_dataset()
            
            # 2. Daily Mean Calculation
            print(f"Calculating daily means for {year}...")
            # We resample by '1D' (day)
            ds_daily = ds_subset.resample(time='1D').mean()
            
            # 3. Interpolation to GEOS 1-degree grid
            print(f"Interpolating to GEOS 1-degree (181x360)...")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ARCO-ERA5 2m_temperature to daily 1-degree GEOS grid.")
    parser.add_argument("--start_year", type=int, default=2022)
    parser.add_argument("--end_year", type=int, default=2025)
    parser.add_argument("--output_dir", type=str, default="/home1/11353/afahad/geos_subc/dataprocess/era5_t2m")
    parser.add_argument("--zarr_path", type=str, default=DEFAULT_ZARR_PATH,
                        help="ARCO ERA5 Zarr source. Defaults to the public v3 full 37-variable hourly store.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing era5_t2m_<year>.zarr files.")
    args = parser.parse_args()
    
    process_era5_t2m_yearly(
        start_year=args.start_year,
        end_year=args.end_year,
        output_base_dir=args.output_dir,
        overwrite=args.overwrite,
        zarr_path=args.zarr_path,
    )
