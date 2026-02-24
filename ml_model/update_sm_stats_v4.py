"""
Update SM Stats in v4_global_stats.pt
=====================================
Targeted script to recalculate Soil Moisture (SM) min/max bounds
and update the v4_global_stats.pt file without touching other variables.

Usage:
    python ml_model/update_sm_stats_v4.py
"""
import torch
import xarray as xr
import numpy as np
import os
import glob

STATS_PATH = "ml_model/v4_global_stats.pt"
DATA_ROOT = "dataprocess"

def update_sm_only():
    if not os.path.exists(STATS_PATH):
        print(f"Error: {STATS_PATH} not found.")
        return

    # Load existing bounds
    try:
        bounds = torch.load(STATS_PATH, weights_only=True)
    except Exception as e:
        # Fallback for older torch versions
        bounds = torch.load(STATS_PATH)
        
    print(f"Current SM bounds in file: {bounds.get('sm')}")

    sm_min = float('inf')
    sm_max = float('-inf')

    # Find all SoilW Zarr files
    files = sorted(glob.glob(os.path.join(DATA_ROOT, "soilw_weekly_*.zarr")))
    if not files:
        print(f"No Soil Moisture Zarr files found in {DATA_ROOT}")
        return

    print(f"Scanning {len(files)} files for Soil Moisture bounds...")
    
    any_data = False
    for f in files:
        try:
            ds = xr.open_zarr(f, consolidated=False)
            # Detect variable
            var = next((v for v in ['soilw', 'sm', 'soil_moisture'] if v in ds), None)
            if var:
                data = ds[var].values
                # Filter out NaNs to get physical land range
                valid = data[~np.isnan(data)]
                if valid.size > 0:
                    any_data = True
                    f_min = np.min(valid)
                    f_max = np.max(valid)
                    sm_min = min(sm_min, f_min)
                    sm_max = max(sm_max, f_max)
                    # print(f"  {os.path.basename(f)}: {f_min:.4f} to {f_max:.4f}")
            ds.close()
        except Exception as e:
            print(f"  Error processing {f}: {e}")

    if not any_data:
        print("Could not find any valid SM data.")
        return

    print(f"\nRecalculated SM Bounds (Land Only):")
    print(f"  Min: {sm_min:.6f}")
    print(f"  Max: {sm_max:.6f}")

    # Update only the SM entry
    bounds['sm'] = {'min': float(sm_min), 'max': float(sm_max)}
    
    # Save back to file
    torch.save(bounds, STATS_PATH)
    print(f"\nSuccessfully updated 'sm' in {STATS_PATH}")
    print("Other variables (SST, SSS, IVT, Z500, U250, etc.) were NOT touched.")

if __name__ == "__main__":
    update_sm_only()
