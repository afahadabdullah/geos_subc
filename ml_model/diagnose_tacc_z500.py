import xarray as xr
import numpy as np
import os
import torch

def diagnose_zarr(path):
    if not os.path.exists(path):
        print(f"❌ Path not found: {path}")
        return
    
    print(f"🔍 Analyzing Zarr: {path}")
    ds = xr.open_zarr(path, consolidated=False)
    print(ds)
    
    # Check variables
    for var in ds.data_vars:
        data = ds[var].isel(S=0).values
        print(f"  Variable: {var:10s} | Shape: {data.shape} | Min: {np.nanmin(data):10.2f} | Max: {np.nanmax(data):10.2f} | Mean: {np.nanmean(data):10.2f}")
    
    ds.close()

if __name__ == "__main__":
    # Check a few years
    base = "/scratch/11353/afahad/geossub/geos_subc/dataprocess"
    for year in [2000, 2015, 2020]:
        path = os.path.join(base, f"z500_u250_weekly_{year}.zarr")
        if os.path.exists(path):
            diagnose_zarr(path)
            
    # Check stats file if it exists
    stats_path = "/scratch/11353/afahad/geossub/geos_subc/ml_model/v5_global_stats.pt"
    if os.path.exists(stats_path):
        print(f"\n📊 Checking stats file: {stats_path}")
        stats = torch.load(stats_path, map_location='cpu', weights_only=True)
        for k, v in stats.items():
            if isinstance(v, dict):
                print(f"  {k:15s}: {v}")
