import xarray as xr
import numpy as np
import os
import shutil
import dask
from tqdm import tqdm
from dask.diagnostics import ProgressBar

def process_fimr_weekly(start_year=2017, end_year=2025, data_dir="dataprocess"):
    """
    Processes FIMR Forecast Zarr files to 4-weekly means.
    - Input: Daily data (lead days L)
    - Output: Weekly means (4 leads: Weeks 1-4) as geos_subc_{year}.zarr
    """
    print(f"Processing FIMR files from {start_year} to {end_year}...")

    for year in range(start_year, end_year + 1):
        input_path = f"{data_dir}/fimr_forecast_{year}.zarr"
        output_path = f"{data_dir}/geos_subc_{year}.zarr"
        
        if not os.path.exists(input_path):
            print(f"Input file not found: {input_path}. Skipping.")
            continue
            
        print(f"\n--- Processing {year} ---")
        if os.path.exists(output_path):
            print(f"Output file {output_path} already exists. Skipping.")
            continue
        
        try:
            # Load Data
            ds = xr.open_zarr(input_path, consolidated=False)
            
            # 1. Identify Lead Dimension
            lead_dim = None
            for dim in ['L', 'lead', 'lead_time']:
                if dim in ds.dims:
                    lead_dim = dim
                    break
            
            if not lead_dim:
                print(f"Error: Lead dimension not found. Dims: {ds.dims}")
                continue
                
            n_leads = ds.sizes[lead_dim]
            print(f"Found lead dimension '{lead_dim}' with {n_leads} steps.")

            # 2. Extract first 28 days for Weeks 1-4
            if n_leads < 28:
                print(f"Error: Not enough lead steps ({n_leads} < 28). Skipping.")
                continue
                
            ds_28 = ds.isel({lead_dim: slice(0, 28)})
            
            # 3. Compute Weekly Means
            print(f"Computing weekly means (coarsen {lead_dim}=7)...")
            ds_weekly = ds_28.coarsen({lead_dim: 7}, boundary='exact').mean()
            
            # Standardize lead dimension name to 'L' for training pipeline
            if lead_dim != 'L':
                ds_weekly = ds_weekly.rename({lead_dim: 'L'})
            
            # 4. Unit Conversion for Precipitation: kg/m2/s -> mm/day
            # FIMR 'pr' is typically flux in kg/m2/s
            print("Converting precipitation to mm/day (* 86400)...")
            if 'pr' in ds_weekly:
                ds_weekly['pr'] = ds_weekly['pr'] * 86400
            
            # 5. Save to Final Path
            temp_path = f"{output_path}_temp"
            print(f"Saving to {output_path}...")
            
            with ProgressBar(), dask.config.set(scheduler='synchronous'):
                # Reset encoding to avoid metadata conflicts
                ds_weekly.encoding = {}
                for var in ds_weekly.variables:
                    ds_weekly[var].encoding = {}
                
                ds_weekly.to_zarr(temp_path, mode='w')
            
            # Cleanup and Move
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.rename(temp_path, output_path)
            
            ds.close()
            print(f"Successfully processed {year} -> {output_path}")
            
        except Exception as e:
            print(f"Failed to process {year}: {e}")
            if os.path.exists(f"{output_path}_temp"):
                 shutil.rmtree(f"{output_path}_temp")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process FIMR daily to weekly")
    parser.add_argument("--start", type=int, default=2017)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()
    
    process_fimr_weekly(start_year=args.start, end_year=args.end)
