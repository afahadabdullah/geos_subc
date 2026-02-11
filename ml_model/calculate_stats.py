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
            geos_vals = ds_geos['pr'].values.flatten()
            geos_log = np.log1p(np.maximum(np.nan_to_num(geos_vals, nan=0.0), 0.0))
            global_min = min(global_min, float(np.min(geos_log)))
            global_max = max(global_max, float(np.max(geos_log)))
            ds_geos.close()
            
            ds_gpcp = xr.open_zarr(gpcp_path, consolidated=False)
            gpcp_vals = ds_gpcp['precip'].values.flatten()
            gpcp_log = np.log1p(np.maximum(np.nan_to_num(gpcp_vals, nan=0.0), 0.0))
            global_min = min(global_min, float(np.min(gpcp_log)))
            global_max = max(global_max, float(np.max(gpcp_log)))
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
        "train_years": [start_year, end_year]
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Stats saved to {output_file}")
    print(f"Min: {global_min:.4f}, Max: {global_max:.4f}")

if __name__ == "__main__":
    calculate_stats()
