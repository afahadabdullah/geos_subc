import xarray as xr
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def calculate_stats_z(data_root="dataprocess", start_year=1999, end_year=2014, output_file="ml_model/stats_z.nc"):
    """
    Calculate per-grid Mean and Std Dev for:
    1. log1p(GEOS Forecast)
    2. log1p(GPCP Truth)
    3. log1p(Residual) = log1p(GPCP) - log1p(GEOS)
    
    Used for Z-Score normalization: (x - mean) / std -> N(0, 1)
    """
    print(f"Calculating Per-Grid Z-Score Stats for years {start_year} to {end_year}...")
    
    # Initialize accumulators (shape 181, 360)
    # We don't know shape yet, will init on first batch
    sums = {
        "geos": None, "geos_sq": None, "geos_count": 0,
        "gpcp": None, "gpcp_sq": None, "gpcp_count": 0,
        "resid": None, "resid_sq": None, "resid_count": 0
    }
    
    shape = None
    
    for year in range(start_year, end_year + 1):
        geos_path = os.path.join(data_root, f"geos_subc_{year}.zarr")
        gpcp_path = os.path.join(data_root, f"gpcp_weekly_{year}.zarr")
        
        if not os.path.exists(geos_path) or not os.path.exists(gpcp_path):
            print(f"  WARNING: Skipping {year} — missing GEOS or GPCP zarr")
            continue
            
        print(f"  Processing {year}...")
        try:
            ds_geos = xr.open_zarr(geos_path, consolidated=False)
            ds_gpcp = xr.open_zarr(gpcp_path, consolidated=False)
            
            # --- GEOS Stats ---
            geos_vals = ds_geos['pr'].values
            # log1p
            geos_log = np.log1p(np.maximum(np.nan_to_num(geos_vals, nan=0.0), 0.0))
            
            # Flatten time/member dims to (N, H, W)
            # geos_vals: (S, lead, lat, lon) or (S, M, lead, lat, lon)?
            # Usually (S, lead, lat, lon). If lead is condensed or M is present.
            # calculate_stats.py line 58 check for dim=4 implies (S, lead, lat, lon) or (M, ...)?
            # dataset.py says 'pr' dim is (S, lead, lat, lon) but if 'M' exists it's (S, M, lead, lat, lon)?
            # Actually dataset.py: `ds_geos['pr']` has dims.
            # If M exists, dataset selects it.
            # We want stats across ALL samples (S) and ALL members (M) and ALL leads?
            # Or is 'lead' a separate channel?
            # Model takes 4 channels (weeks 1-4).
            # If we normalize per-channel (per-week), we need (4, H, W) stats.
            # If we normalize globally across weeks, we need (H, W).
            # The user asked "per grid per variable".
            # Usually weather models normalize per-variable. Week 1 is same variable as Week 2 (Precip).
            # So (H, W) is sufficient?
            # BUT: Week 1 physics might differ from Week 4.
            # Let's compute (4, H, W) to be safe? Or just (H, W).
            # V1 used global min/max across all weeks.
            # Let's stick to (H, W) to keep it simple and robust (more samples).
            # If we want (C, H, W), we can change later.
            # For now, treat all weeks as samples of "Precipitation".
            
            # Flatten to (-1, H, W)
            if geos_log.ndim == 4: # (S, Lead, H, W) or (S, M, H, W)?
                # Dataset.py treats Lead as Channels.
                # calculate_stats.py treated it as flats.
                pass
            
            # We need to handle dimensions carefully.
            # Let's reshape to (-1, H, W).
            geos_flat = geos_log.reshape(-1, geos_log.shape[-2], geos_log.shape[-1])
            
            if shape is None:
                shape = geos_flat.shape[1:]  # (H, W)
                for key in sums:
                    if "count" not in key:
                        sums[key] = np.zeros(shape, dtype=np.float64)
            
            sums["geos"] += np.sum(geos_flat, axis=0)
            sums["geos_sq"] += np.sum(geos_flat**2, axis=0)
            sums["geos_count"] += geos_flat.shape[0]
            
            # --- GPCP Stats ---
            gpcp_vals = ds_gpcp['precip'].values
            gpcp_log = np.log1p(np.maximum(np.nan_to_num(gpcp_vals, nan=0.0), 0.0))
            gpcp_flat = gpcp_log.reshape(-1, gpcp_log.shape[-2], gpcp_log.shape[-1])
            
            sums["gpcp"] += np.sum(gpcp_flat, axis=0)
            sums["gpcp_sq"] += np.sum(gpcp_flat**2, axis=0)
            sums["gpcp_count"] += gpcp_flat.shape[0]
            
            # --- Residual Stats ---
            # Match dates like calculate_stats.py
            geos_s = pd.to_datetime(ds_geos['S'].values)
            gpcp_s = pd.to_datetime(ds_gpcp['S'].values)
            common_dates = set(geos_s) & set(gpcp_s)
            
            res_list = []
            
            for s_date in common_dates:
                g_idx = list(geos_s).index(s_date)
                p_idx = list(gpcp_s).index(s_date)
                
                # (Lead, Lat, Lon) or (M, Lead, Lat, Lon)
                geos_sample = geos_log[g_idx] 
                gpcp_sample = gpcp_log[p_idx] # (Lead, Lat, Lon)
                
                # Check for M dim in geos
                # ds_geos.dims check?
                # Assume if ndim > gpcp.ndim, it has M?
                # GPCP is (Lead, H, W) = 3 dims.
                # GEOS might be (M, Lead, H, W) = 4 dims.
                
                if geos_sample.ndim == 4: # (M, Lead, H, W)
                    for m in range(geos_sample.shape[0]):
                         # res = log(GPCP) - log(GEOS_m)
                         res = gpcp_sample - geos_sample[m]
                         res_list.append(res)
                else:
                    # (Lead, H, W)
                    res = gpcp_sample - geos_sample
                    res_list.append(res)
            
            if res_list:
                res_stack = np.stack(res_list) # (N_samples, Lead, H, W)
                res_flat = res_stack.reshape(-1, shape[0], shape[1])
                
                sums["resid"] += np.sum(res_flat, axis=0)
                sums["resid_sq"] += np.sum(res_flat**2, axis=0)
                sums["resid_count"] += res_flat.shape[0]
            
            ds_geos.close()
            ds_gpcp.close()
            
        except Exception as e:
            print(f"Error processing {year}: {e}")
            import traceback
            traceback.print_exc()

    if sums["geos_count"] == 0:
        print("No data found.")
        return

    # Compute Means and Stds
    results = {}
    for key in ["geos", "gpcp", "resid"]:
        count = sums[f"{key}_count"]
        mean = sums[key] / count
        # Var = E[x^2] - (E[x])^2
        var = (sums[f"{key}_sq"] / count) - (mean ** 2)
        var = np.maximum(var, 0) # Clip negative due to float precision
        std = np.sqrt(var)
        
        # Avoid zero division in normalization
        std[std < 1e-6] = 1.0 
        
        results[f"{key}_mean"] = mean
        results[f"{key}_std"] = std
        
    # Save to NetCDF
    print(f"Saving statistics to {output_file}...")
    
    # Create dataset
    # Lat/Lon not strictly needed if we just treat as image, but good to have
    # We define dims y, x
    ds_out = xr.Dataset(
        {
            "geos_mean": (("y", "x"), results["geos_mean"].astype(np.float32)),
            "geos_std": (("y", "x"), results["geos_std"].astype(np.float32)),
            "gpcp_mean": (("y", "x"), results["gpcp_mean"].astype(np.float32)),
            "gpcp_std": (("y", "x"), results["gpcp_std"].astype(np.float32)),
            "resid_mean": (("y", "x"), results["resid_mean"].astype(np.float32)),
            "resid_std": (("y", "x"), results["resid_std"].astype(np.float32)),
        },
        coords={
            # We can try to load real coords if needed, but index is fine for now
            # since data is consistent shape.
            # Future improvement: copy coords from sample ds_geos
        }
    )
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ds_out.to_netcdf(output_file)
    print("Done.")

if __name__ == "__main__":
    calculate_stats_z()
