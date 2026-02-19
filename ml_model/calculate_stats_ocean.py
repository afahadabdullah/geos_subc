"""
Calculate Normalization Stats (with Ocean Variables)
=====================================================
Extends the original calculate_stats.py to also compute min/max
for SST and SSS fields. Outputs norm_stats_ocean.json.

Usage:
    python ml_model/calculate_stats_ocean.py
"""
import xarray as xr
import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm


def calculate_stats(data_root="dataprocess", start_year=1999, end_year=2014,
                    output_file="ml_model/norm_stats_ocean.json"):
    """
    Calculate global min and max for:
    - log1p-transformed precipitation (GEOS + GPCP)
    - log1p residual (GPCP - GEOS)
    - SST (raw, not log-transformed — can be negative in some regions)
    - SSS (raw, not log-transformed)
    """
    # Precipitation stats (same as original)
    pr_min = float('inf')
    pr_max = float('-inf')
    res_min = float('inf')
    res_max = float('-inf')
    
    # Ocean variable stats
    sst_min = float('inf')
    sst_max = float('-inf')
    sss_min = float('inf')
    sss_max = float('-inf')
    
    print(f"Calculating stats for years {start_year} to {end_year}...")
    
    for year in range(start_year, end_year + 1):
        geos_path = os.path.join(data_root, f"geos_subc_{year}.zarr")
        gpcp_path = os.path.join(data_root, f"gpcp_weekly_{year}.zarr")
        sst_path = os.path.join(data_root, f"sst_weekly_{year}.zarr")
        sss_path = os.path.join(data_root, f"sss_weekly_{year}.zarr")
        
        # --- Precipitation ---
        if os.path.exists(geos_path) and os.path.exists(gpcp_path):
            print(f"  [{year}] Processing precipitation...")
            try:
                ds_geos = xr.open_zarr(geos_path, consolidated=False)
                ds_gpcp = xr.open_zarr(gpcp_path, consolidated=False)
                
                geos_vals = ds_geos['pr'].values
                gpcp_vals = ds_gpcp['precip'].values
                
                geos_log = np.log1p(np.maximum(np.nan_to_num(geos_vals, nan=0.0), 0.0))
                gpcp_log = np.log1p(np.maximum(np.nan_to_num(gpcp_vals, nan=0.0), 0.0))
                
                pr_min = min(pr_min, float(np.min(geos_log)), float(np.min(gpcp_log)))
                pr_max = max(pr_max, float(np.max(geos_log)), float(np.max(gpcp_log)))
                
                # Residual stats
                geos_s = pd.to_datetime(ds_geos['S'].values)
                gpcp_s = pd.to_datetime(ds_gpcp['S'].values)
                common_dates = set(geos_s) & set(gpcp_s)
                
                for s_date in common_dates:
                    g_idx = list(geos_s).index(s_date)
                    p_idx = list(gpcp_s).index(s_date)
                    geos_sample = geos_log[g_idx]
                    gpcp_sample = gpcp_log[p_idx]
                    
                    if geos_sample.ndim == 4:  # (M, weeks, lat, lon)
                        for m in range(geos_sample.shape[0]):
                            res = gpcp_sample - geos_sample[m]
                            res_min = min(res_min, float(np.min(res)))
                            res_max = max(res_max, float(np.max(res)))
                    else:
                        res = gpcp_sample - geos_sample
                        res_min = min(res_min, float(np.min(res)))
                        res_max = max(res_max, float(np.max(res)))
                
                ds_geos.close()
                ds_gpcp.close()
            except Exception as e:
                print(f"    Error processing precipitation {year}: {e}")
        else:
            print(f"  [{year}] WARNING: Missing GEOS or GPCP zarr")
        
        # --- SST ---
        if os.path.exists(sst_path):
            print(f"  [{year}] Processing SST...")
            try:
                ds_sst = xr.open_zarr(sst_path, consolidated=False)
                sst_vals = np.nan_to_num(ds_sst['sst'].values, nan=0.0).astype(np.float32)
                sst_min = min(sst_min, float(np.nanmin(sst_vals)))
                sst_max = max(sst_max, float(np.nanmax(sst_vals)))
                ds_sst.close()
            except Exception as e:
                print(f"    Error processing SST {year}: {e}")
        else:
            print(f"  [{year}] WARNING: SST zarr not found at {sst_path}")
        
        # --- SSS ---
        if os.path.exists(sss_path):
            print(f"  [{year}] Processing SSS...")
            try:
                ds_sss = xr.open_zarr(sss_path, consolidated=False)
                sss_vals = np.nan_to_num(ds_sss['sss'].values, nan=0.0).astype(np.float32)
                sss_min = min(sss_min, float(np.nanmin(sss_vals)))
                sss_max = max(sss_max, float(np.nanmax(sss_vals)))
                ds_sss.close()
            except Exception as e:
                print(f"    Error processing SSS {year}: {e}")
        else:
            print(f"  [{year}] WARNING: SSS zarr not found at {sss_path}")
    
    # Build stats dict
    stats = {
        "log1p_min": pr_min,
        "log1p_max": pr_max,
        "residual_min": res_min,
        "residual_max": res_max,
        "sst_min": sst_min if sst_min != float('inf') else 0.0,
        "sst_max": sst_max if sst_max != float('-inf') else 1.0,
        "sss_min": sss_min if sss_min != float('inf') else 0.0,
        "sss_max": sss_max if sss_max != float('-inf') else 1.0,
        "train_years": [start_year, end_year]
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\nStats saved to {output_file}")
    print(f"  Precip log1p: [{pr_min:.4f}, {pr_max:.4f}]")
    print(f"  Residual:     [{res_min:.4f}, {res_max:.4f}]")
    print(f"  SST:          [{stats['sst_min']:.4f}, {stats['sst_max']:.4f}]")
    print(f"  SSS:          [{stats['sss_min']:.4f}, {stats['sss_max']:.4f}]")


if __name__ == "__main__":
    calculate_stats()
