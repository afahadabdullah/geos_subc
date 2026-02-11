import xarray as xr
import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm

def calculate_stats(data_root="dataprocess", start_year=1999, end_year=2014, output_file="ml_model/norm_stats.json"):
    """
    Calculate mean and std for log1p-transformed precipitation data across training years.
    """
    all_log_vals = []
    
    print(f"Calculating stats for years {start_year} to {end_year}...")
    
    for year in range(start_year, end_year + 1):
        geos_path = os.path.join(data_root, f"geos_subc_{year}.zarr")
        gpcp_path = os.path.join(data_root, f"gpcp_weekly_{year}.zarr")
        
        if not os.path.exists(geos_path) or not os.path.exists(gpcp_path):
            print(f"  WARNING: Skipping {year} — missing GEOS or GPCP zarr file")
            continue
            
        print(f"  Processing {year}...")
        try:
            ds_geos = xr.open_zarr(geos_path, consolidated=False)
            # Process GEOS (Forecast)
            geos_vals = ds_geos['pr'].values.flatten()
            geos_log = np.log1p(np.maximum(np.nan_to_num(geos_vals, nan=0.0), 0.0))
            all_log_vals.append(geos_log)
            ds_geos.close()
            
            # Process GPCP (Truth)
            ds_gpcp = xr.open_zarr(gpcp_path, consolidated=False)
            gpcp_vals = ds_gpcp['precip'].values.flatten()
            gpcp_log = np.log1p(np.maximum(np.nan_to_num(gpcp_vals, nan=0.0), 0.0))
            all_log_vals.append(gpcp_log)
            ds_gpcp.close()
            
        except Exception as e:
            print(f"Error processing {year}: {e}")

    if not all_log_vals:
        print("No data found to calculate stats.")
        return

    print("Computing global mean and std...")
    # Concatenate all to compute global stats
    combined = np.concatenate(all_log_vals)
    mean = float(np.mean(combined))
    std = float(np.std(combined))
    
    stats = {
        "log1p_mean": mean,
        "log1p_std": std,
        "train_years": [start_year, end_year]
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Stats saved to {output_file}")
    print(f"Mean: {mean:.4f}, Std: {std:.4f}")

if __name__ == "__main__":
    calculate_stats()
