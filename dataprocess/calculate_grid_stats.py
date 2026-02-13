import xarray as xr
import numpy as np
import os
from tqdm import tqdm

def calculate_stats():
    data_root = "dataprocess"
    years = range(1999, 2015) # Training set range
    
    # Initialize accumulators
    # We don't know shape yet, will init on first file
    geos_sum = None
    geos_sq_sum = None
    geos_count = 0
    
    gpcp_sum = None
    gpcp_sq_sum = None
    gpcp_count = 0
    
    print("Calculating per-grid statistics for 1999-2014...")
    
    for year in tqdm(years):
        geos_path = os.path.join(data_root, f"geos_subc_{year}.zarr")
        gpcp_path = os.path.join(data_root, f"gpcp_weekly_{year}.zarr")
        
        if not os.path.exists(geos_path) or not os.path.exists(gpcp_path):
            print(f"Skipping {year} (missing files)")
            continue
            
        ds_geos = xr.open_zarr(geos_path, consolidated=False)
        ds_gpcp = xr.open_zarr(gpcp_path, consolidated=False)
        
        # Load data (assuming it fits in RAM or doing chunked)
        # GEOS: (S, M, L, Y, X) or (S, L, Y, X)
        # We want stats over (S, M, L) -> Result (Y, X)
        # Flatten all dimensions except Y, X
        
        # GPCP
        gpcp_data = ds_gpcp['precip'].values # (S, Y, X)
        # Handle NaNs: replace with 0 or exclude? Rainfall shouldn't be NaN usually.
        # But if masked, we should use nanmean. 
        # For simplicity, assuming no NaNs for now or replace with 0.
        gpcp_data = np.nan_to_num(gpcp_data, nan=0.0)
        
        if gpcp_sum is None:
            H, W = gpcp_data.shape[-2:]
            gpcp_sum = np.zeros((H, W), dtype=np.float64)
            gpcp_sq_sum = np.zeros((H, W), dtype=np.float64)
            
        gpcp_count += gpcp_data.shape[0] # N samples
        gpcp_sum += np.sum(gpcp_data, axis=0)
        gpcp_sq_sum += np.sum(gpcp_data**2, axis=0)
        
        # GEOS
        geos_val = ds_geos['pr'].values # (S, M, L, Y, X) or similar
        # Collapse S, M, L
        # Make one big array (N, Y, X)
        if 'M' in ds_geos.dims:
             # (S, M, L, Y, X) -> reshape to (-1, Y, X)
             geos_reshaped = geos_val.reshape(-1, H, W)
        else:
             # (S, L, Y, X) -> reshape
             geos_reshaped = geos_val.reshape(-1, H, W)
             
        geos_reshaped = np.nan_to_num(geos_reshaped, nan=0.0)
             
        if geos_sum is None:
            geos_sum = np.zeros((H, W), dtype=np.float64)
            geos_sq_sum = np.zeros((H, W), dtype=np.float64)
            
        geos_count += geos_reshaped.shape[0]
        geos_sum += np.sum(geos_reshaped, axis=0)
        geos_sq_sum += np.sum(geos_reshaped**2, axis=0)
        
    # Final Compute
    gpcp_mean = gpcp_sum / gpcp_count
    gpcp_var = (gpcp_sq_sum / gpcp_count) - (gpcp_mean ** 2)
    gpcp_std = np.sqrt(np.maximum(gpcp_var, 1e-6)) # Avoid sqrt(negative)
    
    geos_mean = geos_sum / geos_count
    geos_var = (geos_sq_sum / geos_count) - (geos_mean ** 2)
    geos_std = np.sqrt(np.maximum(geos_var, 1e-6))
    
    # Clip very small std to avoid division by zero (e.g. deserts)
    gpcp_std = np.maximum(gpcp_std, 1e-2)
    geos_std = np.maximum(geos_std, 1e-2)
    
    # Save
    ds_out = xr.Dataset({
        'geos_mean': (('lat', 'lon'), geos_mean),
        'geos_std': (('lat', 'lon'), geos_std),
        'gpcp_mean': (('lat', 'lon'), gpcp_mean),
        'gpcp_std': (('lat', 'lon'), gpcp_std),
    })
    
    out_path = "ml_model/grid_stats.nc"
    ds_out.to_netcdf(out_path)
    print(f"Stats saved to {out_path}")

if __name__ == "__main__":
    calculate_stats()
