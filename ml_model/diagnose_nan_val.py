import xarray as xr
import torch
import numpy as np
import os
import pandas as pd
from dataset_hybrid import S2SHybridDataset

def check_val_nans(data_root="dataprocess"):
    years = [2015, 2016]
    print(f"--- CHECKING VALIDATION DATA FOR NaNs ({years}) ---")
    print(f"Data Root: {data_root}")
    
    for year in years:
        gpcp_path = f"{data_root}/gpcp_weekly_{year}.zarr"
        geos_path = f"{data_root}/geos_subc_{year}.zarr"
        
        print(f"\nYEAR {year}:")
        
        if os.path.exists(geos_path):
            try:
                ds_geos = xr.open_zarr(geos_path, consolidated=False)
                print(f"  GEOS File Info: {ds_geos.dims}")
                print(f"  GEOS Coords: {list(ds_geos.coords)}")
                print(f"  GEOS Vars: {list(ds_geos.data_vars)}")
                
                # Match robust detection in dataset_hybrid.py
                geos_var = next((v for v in ['pr', 'precip', 'PRECTOT', 'flux_precip'] if v in ds_geos), 'pr')
                if geos_var in ds_geos:
                    precip_geos = ds_geos[geos_var].values
                    nan_count = np.isnan(precip_geos).sum()
                    total = precip_geos.size
                    print(f"  GEOS ({geos_var}): NaNs = {nan_count} / {total} ({100*nan_count/total:.2f}%)")
                    if total > 0:
                        print(f"  GEOS Mean/Max: {np.nanmean(precip_geos):.4f} / {np.nanmax(precip_geos):.4f}")
                else:
                    print(f"  GEOS: Variable {geos_var} NOT found!")
                ds_geos.close()
            except Exception as e:
                print(f"  GEOS Error: {e}")
        else:
            print(f"  GEOS: MISSING {geos_path}")

        if os.path.exists(gpcp_path):
            try:
                ds_gpcp = xr.open_zarr(gpcp_path, consolidated=False)
                print(f"  GPCP File Info: {ds_gpcp.dims}")
                print(f"  GPCP Coords: {list(ds_gpcp.coords)}")
                print(f"  GPCP Vars: {list(ds_gpcp.data_vars)}")
                
                # Robust GPCP detection
                gpcp_var = next((v for v in ['precip', 'target', 'total_precipitation'] if v in ds_gpcp), list(ds_gpcp.data_vars)[0])
                if gpcp_var in ds_gpcp:
                    precip_gpcp = ds_gpcp[gpcp_var].values
                    nan_count = np.isnan(precip_gpcp).sum()
                    total = precip_gpcp.size
                    print(f"  GPCP ({gpcp_var}): NaNs = {nan_count} / {total} ({100*nan_count/total:.2f}%)")
                    if nan_count > 0:
                        # Check lead weeks
                        L_dim = ds_gpcp.dims.get('L', 4)
                        for l in range(min(4, L_dim)):
                            # Handle potential dim order differences
                            try:
                                # We assume (S, L, H, W)
                                l_slice = precip_gpcp[:, l, :, :] if precip_gpcp.ndim == 4 else (precip_gpcp[l] if precip_gpcp.ndim == 3 else precip_gpcp)
                                l_nans = np.isnan(l_slice).sum()
                                print(f"    - Lead {l}: NaNs = {l_nans}")
                            except:
                                pass
                    if total > 0:
                        print(f"  GPCP Mean/Max: {np.nanmean(precip_gpcp):.4f} / {np.nanmax(precip_gpcp):.4f}")
                else:
                    print(f"  GPCP: Variable {gpcp_var} NOT found!")
                ds_gpcp.close()
            except Exception as e:
                print(f"  GPCP Error: {e}")
        else:
            print(f"  GPCP: MISSING {gpcp_path}")

    # Dataset check
    print("\n--- DATASET RETRIEVAL CHECK ---")
    try:
        print(f"Initializing Dataset with years {years[0]}-{years[-1]} using Root: {data_root}")
        dataset = S2SHybridDataset(data_root=data_root, start_year=years[0], end_year=years[-1], normalize=True)
        print(f"Dataset Length: {len(dataset)}")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample 0 keys: {sample.keys()}")
            print(f"  y_target bounds: {sample['y_target'].min().item():.4f} to {sample['y_target'].max().item():.4f}")
            print(f"  target_raw_full NaNs: {torch.isnan(sample['target_raw_full']).sum().item()}")
        else:
            print("  WARNING: Dataset is empty for these years!")
    except Exception as e:
        print(f"  Dataset test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataprocess")
    args = parser.parse_args()
    check_val_nans(args.data_root)
