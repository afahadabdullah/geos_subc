import torch
"""
GEOS S2S3 Dataset Loader

Handles loading of:
1. GEOS Subseasonal Forecasts (Input) - Format: Zarr
2. GPCP Precipitation (Target/Truth) - Format: Zarr
3. MJO Indices (Conditioning) - Format: CSV

Features:
- Zarr-based lazy loading or RAM preloading.
- Log-transform normalization (log1p) using pre-calculated stats.
- Sequential or Random sampling.
"""
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import pandas as pd
import os

class GeosSubCDataset(Dataset):
    def __init__(self, data_root="dataprocess", start_year=1999, end_year=2016, mjo_file="mjo_processed.csv", transform=None, preload=True, normalization="minmax"):
        """
        Args:
            data_root (str): Root directory.
            start_year (int): Start year.
            end_year (int): End year.
            mjo_file (str): MJO CSV.
            transform (callable): Optional transform.
            preload (bool): Load to RAM.
            normalization (str): "minmax" (global [-1,1]) or "zscore" (per-grid standard).
        """
        self.data_root = data_root
        self.years = list(range(start_year, end_year + 1))
        self.transform = transform
        self.preload = preload
        self.normalization = normalization
        
        # 0. Load Normalization Stats (MUST exist — run calculate_stats.py first)
        if self.normalization == "minmax":
            # Load Min-Max Stats
            stats_path = os.path.join(os.path.dirname(__file__), "norm_stats.json")
            if not os.path.exists(stats_path):
                raise FileNotFoundError(
                    f"norm_stats.json not found. Run `python ml_model/calculate_stats.py`."
                )
            import json
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            self.norm_min = stats["log1p_min"]
            self.norm_max = stats["log1p_max"]
            self.res_min = stats.get("residual_min", -5.0)
            self.res_max = stats.get("residual_max", 5.0)
            print(f"MinMax stats loaded: min={self.norm_min:.4f}, max={self.norm_max:.4f}")
        
        elif self.normalization == "zscore":
            # Load Z-Score Maps (stats_z.nc)
            stats_path = os.path.join(os.path.dirname(__file__), "stats_z.nc")
            if not os.path.exists(stats_path):
                 raise FileNotFoundError(
                    f"stats_z.nc not found. Run `python ml_model/calculate_stats_z.py`."
                )
            ds_stats = xr.open_dataset(stats_path)
            # Load maps into RAM as float32
            self.maps = {
                "geos_mean": ds_stats["geos_mean"].values.astype(np.float32),
                "geos_std": ds_stats["geos_std"].values.astype(np.float32),
                "gpcp_mean": ds_stats["gpcp_mean"].values.astype(np.float32),
                "gpcp_std": ds_stats["gpcp_std"].values.astype(np.float32),
                "resid_mean": ds_stats["resid_mean"].values.astype(np.float32),
                "resid_std": ds_stats["resid_std"].values.astype(np.float32),
            }
            ds_stats.close()
            print("Z-Score stats maps loaded from stats_z.nc")
        
        # 1. Load MJO Data
        mjo_path = os.path.join(data_root, mjo_file)
        if os.path.exists(mjo_path):
            print(f"Loading MJO features from {mjo_path}...")
            self.df_mjo = pd.read_csv(mjo_path)
            self.df_mjo['S'] = pd.to_datetime(self.df_mjo['S'])
            self.df_mjo = self.df_mjo.set_index('S')
        else:
            print(f"Warning: MJO file not found at {mjo_path}. MJO features will be zeros.")
            self.df_mjo = None

        # 2. Index all available samples (Year, InitDate)
        # We need to scan files to build a global index map: idx -> (year, init_date, file_paths)
        # This might be slow if we open every Zarr.
        # Faster: Assume file existence and standard structure?
        # Safer: Open each year once to get 'S' coords.
        
        self.samples = []
        self.preloaded_geos = {} # (S, M) -> ndarray
        self.preloaded_gpcp = {} # S -> ndarray
        self.preloaded_gpcp_obs = {} # S -> ndarray (observed state from prev init)
        
        # Collect ALL init dates across years for prev-init lookup
        all_init_dates = []
        
        print(f"Indexing samples from {start_year} to {end_year}...")
        for year in self.years:
            geos_path = os.path.join(data_root, f"geos_subc_{year}.zarr")
            gpcp_path = os.path.join(data_root, f"gpcp_weekly_{year}.zarr")
            
            if not os.path.exists(geos_path) or not os.path.exists(gpcp_path):
                continue
                
            try:
                ds_geos = xr.open_zarr(geos_path, consolidated=False)
                ds_gpcp = xr.open_zarr(gpcp_path, consolidated=False)
                
                if 'S' not in ds_geos.dims:
                    continue
                
                init_dates = pd.to_datetime(ds_geos['S'].values)
                
                # Check for M dimension
                members = [0]
                if 'M' in ds_geos.dims:
                    members = list(range(ds_geos.sizes['M']))
                
                if self.preload:
                    print(f"  Preloading year {year} into RAM...")
                    geos_vals = ds_geos['pr'].compute()
                    gpcp_vals = ds_gpcp['precip'].compute()
                
                for s_idx, s_date in enumerate(init_dates):
                    s_key = str(s_date)
                    all_init_dates.append(s_date)
                    
                    if self.preload:
                        self.preloaded_gpcp[s_key] = gpcp_vals.isel(S=s_idx).values.astype(np.float32)
                    
                    for m_idx in members:
                        self.samples.append({
                            'year': year,
                            'S': s_date,
                            'S_idx': s_idx,
                            'S_key': s_key,
                            'M': m_idx,
                            'geos_path': geos_path,
                            'gpcp_path': gpcp_path
                        })
                        
                        if self.preload:
                            f_val = geos_vals.isel(S=s_idx)
                            if 'M' in f_val.dims:
                                f_val = f_val.isel(M=m_idx)
                            self.preloaded_geos[(s_key, m_idx)] = f_val.values.astype(np.float32)
                
                ds_geos.close()
                ds_gpcp.close()
                
            except Exception as e:
                print(f"Error indexing/preloading {year}: {e}")
        
        # 3. Build prev-init lookup: for each init S, find the init ~28 days earlier
        all_init_dates_sorted = sorted(set(all_init_dates))
        self.prev_init_map = {}  # s_key -> prev_s_key (or None)
        
        for i, s_date in enumerate(all_init_dates_sorted):
            s_key = str(s_date)
            # Find the closest init date ~28 days before
            target = s_date - pd.Timedelta(days=28)
            best_prev = None
            best_diff = pd.Timedelta(days=999)
            for j in range(i - 1, max(i - 8, -1), -1):  # Check up to 8 prior inits
                if j < 0:
                    break
                diff = abs(all_init_dates_sorted[j] - target)
                if diff < best_diff:
                    best_diff = diff
                    best_prev = all_init_dates_sorted[j]
            
            # Accept if within 14-day tolerance
            if best_prev is not None and best_diff <= pd.Timedelta(days=14):
                self.prev_init_map[s_key] = str(best_prev)
            else:
                self.prev_init_map[s_key] = None
        
        # 4. Cache observed state for preloaded mode
        if self.preload:
            n_obs_found = 0
            for s_key, prev_key in self.prev_init_map.items():
                if prev_key is not None and prev_key in self.preloaded_gpcp:
                    self.preloaded_gpcp_obs[s_key] = self.preloaded_gpcp[prev_key]
                    n_obs_found += 1
                else:
                    self.preloaded_gpcp_obs[s_key] = None  # Will use zeros
            print(f"  Observed state: {n_obs_found}/{len(self.prev_init_map)} samples have prev-init GPCP")
                
        print(f"Found {len(self.samples)} samples (including ensemble members).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        s_date = meta['S']
        s_key = meta['S_key']
        m_idx = meta['M']
        
        try:
            if self.preload:
                x_forecast = self.preloaded_geos[(s_key, m_idx)]
                y_truth = self.preloaded_gpcp[s_key]
                # Observed state (GPCP from prev init, or zeros)
                obs_state = self.preloaded_gpcp_obs.get(s_key)
                if obs_state is None:
                    obs_state = np.zeros_like(y_truth)
            else:
                ds_geos = xr.open_zarr(meta['geos_path'], consolidated=False)
                ds_gpcp = xr.open_zarr(meta['gpcp_path'], consolidated=False)
                
                # Forecast (Input)
                f_data = ds_geos['pr'].isel(S=meta['S_idx'])
                if 'M' in f_data.dims:
                    f_data = f_data.isel(M=m_idx)
                
                # Ground Truth (Target)
                t_data = ds_gpcp['precip'].isel(S=meta['S_idx'])
                
                x_forecast = f_data.values.astype(np.float32)
                y_truth = t_data.values.astype(np.float32)
                
                # Observed state: load prev init's GPCP
                prev_key = self.prev_init_map.get(s_key)
                if prev_key is not None:
                    # Find which zarr file contains the prev init
                    prev_date = pd.Timestamp(prev_key)
                    prev_gpcp_path = os.path.join(self.data_root, f"gpcp_weekly_{prev_date.year}.zarr")
                    if os.path.exists(prev_gpcp_path):
                        ds_prev = xr.open_zarr(prev_gpcp_path, consolidated=False)
                        prev_dates = pd.to_datetime(ds_prev['S'].values)
                        if prev_date in prev_dates:
                            prev_idx = list(prev_dates).index(prev_date)
                            obs_state = ds_prev['precip'].isel(S=prev_idx).values.astype(np.float32)
                        else:
                            obs_state = np.zeros_like(y_truth)
                        ds_prev.close()
                    else:
                        obs_state = np.zeros_like(y_truth)
                else:
                    obs_state = np.zeros_like(y_truth)
                
                ds_geos.close()
                ds_gpcp.close()
            
            # MJO Features (Conditioning)
            if self.df_mjo is not None and s_date in self.df_mjo.index:
                mjo = self.df_mjo.loc[s_date]
                rmm_vals = np.array([mjo['RMM1_lagged'], mjo['RMM2_lagged']], dtype=np.float32)
            else:
                rmm_vals = np.zeros(2, dtype=np.float32)
            
            # Month (1-12)
            month = s_date.month
            month_onehot = np.zeros(12, dtype=np.float32)
            month_onehot[month - 1] = 1.0
            
            if self.normalization == "minmax":
                # Min-Max Normalization to [-1, 1]
                vmin, vmax = self.norm_min, self.norm_max
                denom = vmax - vmin if vmax != vmin else 1.0
                
                x_forecast_norm = 2 * ((x_forecast_log - vmin) / denom) - 1.0
                y_truth_norm = 2 * ((y_truth_log - vmin) / denom) - 1.0
                obs_state_norm = 2 * ((obs_state_log - vmin) / denom) - 1.0
                
                res_denom = self.res_max - self.res_min if self.res_max != self.res_min else 1.0
                residual_norm = 2 * ((residual_log - self.res_min) / res_denom) - 1.0
                residual_norm = np.clip(residual_norm, -1.0, 1.0)
                
                stats_out = {"mode": "minmax", "min": vmin, "max": vmax, "res_min": self.res_min, "res_max": self.res_max}
                
            elif self.normalization == "zscore":
                # Z-Score Normalization: (x - mu) / sigma
                # Maps are (H, W). Input is (C, H, W). Broadcasting works.
                
                x_forecast_norm = (x_forecast_log - self.maps["geos_mean"]) / self.maps["geos_std"]
                # For truth/obs, use GPCP stats
                y_truth_norm = (y_truth_log - self.maps["gpcp_mean"]) / self.maps["gpcp_std"]
                obs_state_norm = (obs_state_log - self.maps["gpcp_mean"]) / self.maps["gpcp_std"]
                
                residual_norm = (residual_log - self.maps["resid_mean"]) / self.maps["resid_std"]
                # Clip extreme outliers? Maybe [-5, 5]
                # Standard practice for diff models is often [-1, 1] if minmax, or approx N(0,1) if Gaussian.
                # Let's clip to [-5, 5] to avoid instability? Or check trainv2 logic.
                # Let's not clip here, trainv2 might handle it, or robust model.
                
                stats_out = {"mode": "zscore"} # Maps are in dataset, can't pass them per sample easily (too big)

            
            return {
                "input_forecast": torch.tensor(x_forecast_norm, dtype=torch.float32), 
                "target_truth": torch.tensor(y_truth_norm, dtype=torch.float32),
                "target_residual": torch.tensor(residual_norm, dtype=torch.float32),
                "observed_state": torch.tensor(obs_state_norm, dtype=torch.float32),
                "mjo_conditioning": torch.tensor(rmm_vals, dtype=torch.float32), 
                "month": torch.tensor(month, dtype=torch.long),
                "month_onehot": torch.tensor(month_onehot, dtype=torch.float32),
                "S": str(s_date),
                "M": m_idx,
                "norm_stats": stats_out
            }
        except Exception as e:
            # Better to show the error and fail than recurse infinitely
            print(f"Error loading sample {idx} (Date {s_date}, Member {m_idx}): {e}")
            raise e

if __name__ == "__main__":
    # verification
    ds = GeosSubCDataset(data_root="dataprocess", start_year=2000, end_year=2001)
    print(f"Sample 0: {ds[0]['S']}")
    print(f"Forecast S: {ds[0]['input_forecast'].shape}")
    print(f"Truth S: {ds[0]['target_truth'].shape}")
    print(f"MJO: {ds[0]['mjo_conditioning']}")
