"""
Calculate Soil Moisture Statistics (Mean, Std)
==============================================
Iterates through Soil Moisture Zarr files to compute global mean and std
using Welford's online algorithm (or simple accumulation).

Usage:
    python ml_model/calculate_stats_sm.py
"""
import xarray as xr
import numpy as np
import os
import json

# Remove torch import
# import torch 

DATA_ROOT = "dataprocess"
START_YEAR = 1999
END_YEAR = 2016
OUTPUT_FILE = "ml_model/sm_stats.json"

def calculate_stats():
    # Accumulators for Mean and Std
    # We want a single global scalar for SM (or per-pixel if desired, but code uses scalars usually)
    # The existing code uses (16, 1, 1) for Obs Mean/Std if loading from global_stats.pt
    # But for now let's just compute the scalar mean/std for SM specifically.
    
    # We will use Welford's algorithm for numerical stability
    count = 0
    mean = 0.0
    m2 = 0.0
    
    print(f"Calculating Soil Moisture Stats for {START_YEAR}-{END_YEAR}...")
    
    for year in range(START_YEAR, END_YEAR): # range is exclusive at end? typical python is. 
                             # strict check: dataset uses range(start, end)
        sm_path = os.path.join(DATA_ROOT, f"soilw_weekly_{year}.zarr")
        
        if not os.path.exists(sm_path):
            print(f"Skipping {year}: {sm_path} not found")
            continue
            
        try:
            ds = xr.open_zarr(sm_path, consolidated=False)
            # Variable name check
            candidates = ['sm', 'soil_moisture', 'soilw', 'swvl1', 'var40', 'mtpr']
            var_name = None
            for c in candidates:
                if c in ds:
                    var_name = c
                    break
            
            if var_name is None:
                # print(f"Skipping {year}: Variable '{var_name}' not found")
                # List available
                print(f"Skipping {year}: Available variables: {list(ds.data_vars)}")
                continue
                
            data = ds[var_name].values # (S, L, H, W)
            # Flatten
            data_flat = data.flatten()
            # Remove NaNs
            data_flat = data_flat[~np.isnan(data_flat)]
            
            if len(data_flat) == 0:
                continue
                
            # Update Welford
            # Batch update is better for speed
            n = len(data_flat)
            new_data = data_flat.astype(np.float64) # Precision
            delta = new_data - mean
            mean += delta.sum() / (count + n)
            m2 += (delta * (new_data - mean)).sum()
            count += n
            
            ds.close()
            print(f"Processed {year}: {n} samples. Current Mean: {mean:.5f}")
            
        except Exception as e:
            print(f"Error processing {year}: {e}")
            
    if count < 2:
        print("Insufficient data to compute stats.")
        return
        
    final_mean = mean
    final_std = np.sqrt(m2 / (count - 1))
    
    print(f"\nFinal Stats (Soil Moisture):")
    print(f"Mean: {final_mean}")
    print(f"Std:  {final_std}")
    
    # Save as JSON
    stats = {
        'sm_mean': float(final_mean),
        'sm_std': float(final_std)
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(stats, f)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    calculate_stats()
