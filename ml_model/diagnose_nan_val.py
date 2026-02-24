import xarray as xr
import torch
import numpy as np
import os
import pandas as pd
from dataset_hybrid import S2SHybridDataset

def check_val_nans(data_root="dataprocess"):
    years = [2015, 2016]
    print(f"--- CHECKING VALIDATION DATA FOR NaNs ({years}) ---")
    
    for year in years:
        gpcp_path = f"{data_root}/gpcp_weekly_{year}.zarr"
        geos_path = f"{data_root}/geos_subc_{year}.zarr"
        
        print(f"\nYEAR {year}:")
        
        if os.path.exists(geos_path):
            ds_geos = xr.open_zarr(geos_path, consolidated=False)
            precip_geos = ds_geos['precip'].values
            nan_count = np.isnan(precip_geos).sum()
            total = precip_geos.size
            print(f"  GEOS: NaNs = {nan_count} / {total} ({100*nan_count/total:.2f}%)")
            ds_geos.close()
        else:
            print(f"  GEOS: MISSING {geos_path}")

        if os.path.exists(gpcp_path):
            ds_gpcp = xr.open_zarr(gpcp_path, consolidated=False)
            # Use data variable 'precip'
            var_name = 'precip' if 'precip' in ds_gpcp else list(ds_gpcp.data_vars)[0]
            precip_gpcp = ds_gpcp[var_name].values
            nan_count = np.isnan(precip_gpcp).sum()
            total = precip_gpcp.size
            print(f"  GPCP: NaNs = {nan_count} / {total} ({100*nan_count/total:.2f}%)")
            if nan_count > 0:
                # Check lead weeks
                for l in range(4):
                    l_nans = np.isnan(precip_gpcp[:, l]).sum()
                    print(f"    - Lead {l}: NaNs = {l_nans}")
            ds_gpcp.close()
        else:
            print(f"  GPCP: MISSING {gpcp_path}")

    # Dataset check
    print("\n--- DATASET RETRIEVAL CHECK ---")
    try:
        dataset = S2SHybridDataset(data_root=data_root, start_year=2015, end_year=2015, normalize=True)
        sample = dataset[0]
        print(f"  Sample 0 keys: {sample.keys()}")
        print(f"  y_target bounds: {sample['y_target'].min().item()} to {sample['y_target'].max().item()}")
        print(f"  target_raw_full NaNs: {torch.isnan(sample['target_raw_full']).sum().item()}")
    except Exception as e:
        print(f"  Dataset failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataprocess")
    args = parser.parse_args()
    check_val_nans(args.data_root)
