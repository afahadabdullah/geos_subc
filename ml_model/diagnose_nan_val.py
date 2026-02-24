import xarray as xr
import torch
import numpy as np
import os
import pandas as pd
from dataset_hybrid import S2SHybridDataset

def check_val_nans(data_root="dataprocess"):
    years = [2015, 2016]
    print(f"--- LIGHTWEIGHT VALIDATION CHECK ({years}) ---")
    print(f"Data Root: {data_root}")
    
    for year in years:
        gpcp_path = f"{data_root}/gpcp_weekly_{year}.zarr"
        geos_path = f"{data_root}/geos_subc_{year}.zarr"
        
        print(f"\nYEAR {year}:")
        
        if os.path.exists(geos_path):
            try:
                ds_geos = xr.open_zarr(geos_path, consolidated=False)
                geos_var = next((v for v in ['pr', 'precip', 'PRECTOT', 'flux_precip'] if v in ds_geos), 'pr')
                # Check only the first sample mid-point to save memory/threads
                mid_idx = ds_geos.sizes['S'] // 2
                slice_val = ds_geos[geos_var].isel(S=mid_idx).values
                print(f"  GEOS ({geos_var}): Mid-point sample (S={mid_idx}) Mean: {np.nanmean(slice_val):.4f}")
                ds_geos.close()
            except Exception as e:
                print(f"  GEOS Error: {e}")
        else:
            print(f"  GEOS: MISSING {geos_path}")

        if os.path.exists(gpcp_path):
            try:
                ds_gpcp = xr.open_zarr(gpcp_path, consolidated=False)
                gpcp_var = next((v for v in ['precip', 'target', 'total_precipitation'] if v in ds_gpcp), list(ds_gpcp.data_vars)[0])
                # Check first and last sample
                s_size = ds_gpcp.sizes['S']
                first_slice = ds_gpcp[gpcp_var].isel(S=0).values
                last_slice = ds_gpcp[gpcp_var].isel(S=s_size-1).values
                
                print(f"  GPCP ({gpcp_var}): First Sample (S=0) NaNs: {np.isnan(first_slice).sum()} / {first_slice.size}")
                print(f"  GPCP ({gpcp_var}): Last Sample (S={s_size-1}) NaNs: {np.isnan(last_slice).sum()} / {last_slice.size}")
                ds_gpcp.close()
            except Exception as e:
                print(f"  GPCP Error: {e}")
        else:
            print(f"  GPCP: MISSING {gpcp_path}")

    # Dataset check without preloading or loading full sample if possible
    print("\n--- DATASET INDEX CHECK ---")
    try:
        dataset = S2SHybridDataset(data_root=data_root, start_year=2015, end_year=2016, normalize=True, preload=False)
        print(f"Dataset indexed {len(dataset)} items for 2015-2016.")
        print("Success: Data is findable and indexable.")
    except Exception as e:
        print(f"  Index test failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataprocess")
    args = parser.parse_args()
    check_val_nans(args.data_root)
