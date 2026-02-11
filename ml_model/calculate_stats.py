import xarray as xr
import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm

def calculate_stats(data_root="dataprocess", start_year=1999, end_year=2014, output_file="ml_model/norm_stats.json"):
    """
    Calculate global min and max for log1p-transformed precipitation data across training years.
    Used for min-max normalization: (log1p(x) - min) / (max - min) -> [0, 1]
    """
    global_min = float('inf')
    global_max = float('-inf')
    residual_min = float('inf')
    residual_max = float('-inf')
    
    print(f"Calculating min-max stats for years {start_year} to {end_year}...")
    
    for year in range(start_year, end_year + 1):
        geos_path = os.path.join(data_root, f"geos_subc_{year}.zarr")
        gpcp_path = os.path.join(data_root, f"gpcp_weekly_{year}.zarr")
        
        if not os.path.exists(geos_path) or not os.path.exists(gpcp_path):
            print(f"  WARNING: Skipping {year} — missing GEOS or GPCP zarr file")
            continue
            
        print(f"  Processing {year}...")
        try:
            ds_geos = xr.open_zarr(geos_path, consolidated=False)
            ds_gpcp = xr.open_zarr(gpcp_path, consolidated=False)
            
            geos_vals = ds_geos['pr'].values
            gpcp_vals = ds_gpcp['precip'].values
            
            # Log1p transform (individual)
            geos_log = np.log1p(np.maximum(np.nan_to_num(geos_vals, nan=0.0), 0.0))
            gpcp_log = np.log1p(np.maximum(np.nan_to_num(gpcp_vals, nan=0.0), 0.0))
            
            # Global min/max for individual fields
            global_min = min(global_min, float(np.min(geos_log)), float(np.min(gpcp_log)))
            global_max = max(global_max, float(np.max(geos_log)), float(np.max(gpcp_log)))
            
            # Residual stats: log1p(GPCP) - log1p(GEOS) per matching init date
            # We need to align by S dimension
            geos_s = pd.to_datetime(ds_geos['S'].values)
            gpcp_s = pd.to_datetime(ds_gpcp['S'].values)
            common_dates = set(geos_s) & set(gpcp_s)
            
            for s_date in common_dates:
                g_idx = list(geos_s).index(s_date)
                p_idx = list(gpcp_s).index(s_date)
                
                geos_sample = geos_log[g_idx]  # May have M dim
                gpcp_sample = gpcp_log[p_idx]   # No M dim
                
                # If GEOS has ensemble members, compute residual for each
                if geos_sample.ndim == 4:  # (M, weeks, lat, lon)
                    for m in range(geos_sample.shape[0]):
                        res = gpcp_sample - geos_sample[m]
                        residual_min = min(residual_min, float(np.min(res)))
                        residual_max = max(residual_max, float(np.max(res)))
                else:
                    res = gpcp_sample - geos_sample
                    residual_min = min(residual_min, float(np.min(res)))
                    residual_max = max(residual_max, float(np.max(res)))
            
            ds_geos.close()
            ds_gpcp.close()
            
        except Exception as e:
            print(f"Error processing {year}: {e}")

    if global_min == float('inf'):
        print("No data found to calculate stats.")
        return

    print(f"Computing global min-max...")
    
    stats = {
        "log1p_min": global_min,
        "log1p_max": global_max,
        "residual_min": residual_min,
        "residual_max": residual_max,
        "train_years": [start_year, end_year]
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Stats saved to {output_file}")
    print(f"Min: {global_min:.4f}, Max: {global_max:.4f}")

if __name__ == "__main__":
    calculate_stats()
