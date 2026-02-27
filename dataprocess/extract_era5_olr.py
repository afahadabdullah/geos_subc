
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
    
    IMPORTANT: OLR (top_net_thermal_radiation / ttr) is a FORECAST ACCUMULATION variable,
    NOT an analysis variable. It lives in the single-level-forecast Zarr store,
    NOT the analysis (ar) store used for IVT/temperature/winds.
    
    ERA5 forecast accumulation convention:
       ttr is accumulated J/m² over the forecast step (typically 1 hour for ERA5).
       OLR = -ttr / forecast_step_seconds (sign: OLR positive outward)
    
    Pipeline:
       1. Load ttr from ARCO-ERA5 forecast store (GCS Zarr)
       2. Convert to OLR flux (W/m²)
       3. Compute daily mean
       4. Interpolate to GEOS 1-degree grid (181x360)
       5. Save as yearly Zarr files
    """
    
    # ARCO-ERA5 forecast accumulation variables (radiation, precipitation, etc.)
    zarr_path = 'gs://gcp-public-data-arco-era5/co/single-level-forecast.zarr'
    
    print(f"Connecting to {zarr_path}...")
    print("(This is the FORECAST store, not the analysis store)")
    
    # Open the dataset lazily with Dask
    try:
        ds = xr.open_zarr(zarr_path, chunks='auto', consolidated=True)
        print("Dataset opened successfully.")
        print(f"Available variables: {list(ds.data_vars)}")
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return

    # Try common variable names for OLR in ARCO-ERA5
    olr_candidates = ['top_net_thermal_radiation', 'ttr']
    olr_var = None
    for candidate in olr_candidates:
        if candidate in ds:
            olr_var = candidate
            print(f"Found OLR variable: '{olr_var}'")
            break
    
    if olr_var is None:
        print(f"ERROR: None of {olr_candidates} found in dataset.")
        print(f"Available variables: {list(ds.data_vars)}")
        print("Try inspecting the dataset manually to find the correct variable name.")
        return
    
    # Print variable metadata
    print(f"  Dims: {ds[olr_var].dims}")
    print(f"  Shape hint: {dict(zip(ds[olr_var].dims, ds[olr_var].shape))}")
    if hasattr(ds[olr_var], 'attrs'):
        print(f"  Attrs: {ds[olr_var].attrs}")
    
    # Define GEOS 1-degree target grid
    target_lat = np.linspace(-90, 90, 181)
    target_lon = np.linspace(0, 359, 360)
    
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    for year in range(start_year, end_year + 1):
        output_path = os.path.join(output_base_dir, f"era5_olr_{year}.zarr")
        
        if os.path.exists(output_path):
            print(f"Skipping {year}, file already exists at {output_path}")
            continue
            
        print(f"\n--- Processing Year: {year} ---")
        
        try:
            # 1. Select year and variable
            ttr = ds[olr_var].sel(time=str(year))
            
            # 2. Convert to OLR flux (W/m²)
            # ERA5 forecast accumulations: ttr is accumulated J/m² over the forecast step.
            # For hourly data (ERA5 default): step = 3600 seconds
            # For 6-hourly: step = 21600 seconds
            # The ARCO-ERA5 co store typically has hourly forecast data.
            # We detect the step from the time coordinate spacing.
            times = ttr.time.values
            if len(times) > 1:
                dt_hours = (pd.Timestamp(times[1]) - pd.Timestamp(times[0])).total_seconds() / 3600
                accum_seconds = dt_hours * 3600
                print(f"  Detected time step: {dt_hours:.1f} hours ({accum_seconds:.0f} seconds)")
            else:
                accum_seconds = 3600  # Default to hourly
                print(f"  Defaulting to hourly accumulation (3600 seconds)")
            
            # OLR = -ttr / accumulation_period
            # ttr is negative in ERA5 (energy leaving system), OLR is positive outward
            olr = -1.0 * ttr / accum_seconds
            olr.name = 'olr'
            
            ds_olr = olr.to_dataset(name='olr')
            ds_olr['olr'].attrs = {
                'units': 'W/m2',
                'long_name': 'Outgoing Longwave Radiation',
                'description': f'Derived from ERA5 {olr_var}: OLR = -{olr_var} / {accum_seconds:.0f}'
            }
            
            # 3. Daily Mean Calculation
            print(f"  Calculating daily means for {year}...")
            ds_daily = ds_olr.resample(time='1D').mean()
            
            # 4. Interpolation to GEOS 1-degree grid
            print(f"  Interpolating to GEOS 1-degree grid (181x360)...")
            ds_interp = ds_daily.interp(latitude=target_lat, longitude=target_lon, method='linear')
            
            # 5. Save to Zarr
            print(f"  Saving to {output_path}...")
            ds_interp = ds_interp.chunk({'time': -1, 'latitude': 181, 'longitude': 360})
            
            with dask.config.set(scheduler='synchronous'):
                ds_interp.to_zarr(output_path, mode='w')
            
            print(f"  ✅ Successfully saved OLR for {year}.")
            
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
