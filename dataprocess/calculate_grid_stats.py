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
            # Check shape of sum reduction
            sample_sum = np.sum(gpcp_data, axis=0)
            gpcp_sum = np.zeros_like(sample_sum, dtype=np.float64)
            gpcp_sq_sum = np.zeros_like(sample_sum, dtype=np.float64)
            
        gpcp_count += gpcp_data.shape[0] # N samples
        gpcp_sum += np.sum(gpcp_data, axis=0)
        gpcp_sq_sum += np.sum(gpcp_data**2, axis=0)
        
        if year == years[0]:
            print(f"  Sample shape (GPCP): {gpcp_data.shape}")
            if gpcp_data.ndim == 4:
                print(f"    (S, L, Y, X) -> Per-Lead Z-Score")
            else:
                print(f"    (S, Y, X) -> Per-Grid (All Weeks Same) Z-Score")
                
        # GEOS
        geos_val = ds_geos['pr'].values 
        
        if year == years[0]:
             print(f"  Sample shape (GEOS): {geos_val.shape}")
        
        if geos_sum is None:
            # Do a trial sum to get shape
            # We want to sum over axes corresponding to S and M
            # Assuming 'S' is axis 0. 
            # If 'M' exists?
            pass

        # We need generic reduction. 
        # Let's count dimensions.
        # Last 2 are Y, X.
        # Middle is L?
        # If ndim=5 (S, M, L, Y, X): reshape to (-1, L, Y, X)? 
        # But M is ensemble members. They are independent samples of the distribution.
        # So we treat (S, M) as N samples.
        
        if geos_val.ndim == 5: # (S, M, L, Y, X)
            S, M, L, Y, X = geos_val.shape
            geos_reshaped = geos_val.reshape(S*M, L, Y, X)
        elif geos_val.ndim == 4: # (S, L, Y, X)
            geos_reshaped = geos_val
        elif geos_val.ndim == 3: # (S, Y, X) - unlikely if GPCP is 4D
            geos_reshaped = geos_val
            
        geos_reshaped = np.nan_to_num(geos_reshaped, nan=0.0)
            
        if geos_sum is None:
            sample_sum_g = np.sum(geos_reshaped, axis=0)
            geos_sum = np.zeros_like(sample_sum_g, dtype=np.float64)
            geos_sq_sum = np.zeros_like(sample_sum_g, dtype=np.float64)
            
        geos_count += geos_reshaped.shape[0]
        geos_sum += np.sum(geos_reshaped, axis=0)
        geos_sq_sum += np.sum(geos_reshaped**2, axis=0)
        
    # Final Compute
    gpcp_mean = gpcp_sum / gpcp_count
    gpcp_var = (gpcp_sq_sum / gpcp_count) - (gpcp_mean ** 2)
    gpcp_std = np.sqrt(np.maximum(gpcp_var, 1e-6))
    
    geos_mean = geos_sum / geos_count
    geos_var = (geos_sq_sum / geos_count) - (geos_mean ** 2)
    geos_std = np.sqrt(np.maximum(geos_var, 1e-6))
    
    # Clip
    gpcp_std = np.maximum(gpcp_std, 1e-2)
    geos_std = np.maximum(geos_std, 1e-2)
    
    # Save
    # Determine dims based on shape
    if gpcp_mean.ndim == 3:
        gpcp_dims = ('lead', 'lat', 'lon')
    else:
        gpcp_dims = ('lat', 'lon')
        
    if geos_mean.ndim == 3:
        geos_dims = ('lead', 'lat', 'lon')
    else:
        geos_dims = ('lat', 'lon')
        
    ds_out = xr.Dataset({
        'geos_mean': (geos_dims, geos_mean),
        'geos_std': (geos_dims, geos_std),
        'gpcp_mean': (gpcp_dims, gpcp_mean),
        'gpcp_std': (gpcp_dims, gpcp_std),
    })
    
    out_path = "ml_model/grid_stats.nc"
    ds_out.to_netcdf(out_path)
    print(f"Stats saved to {out_path}")

if __name__ == "__main__":
    calculate_stats()
