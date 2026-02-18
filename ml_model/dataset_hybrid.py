import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import os
import glob
import pandas as pd
from datetime import datetime, timedelta

class S2SHybridDataset(Dataset):
    """
    Hybrid S2S Dataset for Committee Model.
    
    Inputs:
        - X_obs: (C_obs, H, W) - Stacked Obs (SST, SSS, SM, Prev_GPCP)
                 Each component is (4, H, W) -> Total C_obs = 16
        - X_geos: (4, C_geos, L, H, W) - GEOS Forecast (Precip)
    
    Targets:
        - Y_target: (L, H, W) - GPCP Precip (Weekly Mean)
        
    """
    def __init__(self, data_root="dataprocess", start_year=1999, end_year=2016, 
                 transform=None, preload=False, normalize=True):
        self.data_root = data_root
        self.years = range(start_year, end_year)
        self.transform = transform
        self.preload = preload
        self.normalize = normalize
        
        # Load Stats
        if self.normalize:
            self.load_stats()
        
        # Index samples
        self.prepare_samples()
        
        if self.preload:
            self._preload_data()
        
    def load_stats(self):
        # Load Z-Score stats for Precip
        stats_path = os.path.join(os.path.dirname(__file__), "global_stats.pt")
        # Global stats (Precip)
        if os.path.exists(stats_path):
            print(f"Loading global stats from {stats_path}")
            stats = torch.load(stats_path)
            self.geos_mean = stats['geos_mean']
            self.geos_std = stats['geos_std']
            self.obs_mean = stats['obs_mean'] # (16, 1, 1)
            self.obs_std = stats['obs_std']   # (16, 1, 1)
        else:
            print("Warning: global_stats.pt not found. Using default/fallback normalization.")
            self.obs_mean = None
            self.obs_std = None
            
            stats_path_old = os.path.join(os.path.dirname(__file__), "grid_stats.nc")
            if os.path.exists(stats_path_old):
                ds = xr.open_dataset(stats_path_old)
                self.geos_mean = torch.from_numpy(ds['geos_mean'].values).float()
                self.geos_std = torch.from_numpy(ds['geos_std'].values).float()
                ds.close()
                if self.geos_mean.ndim == 2:
                    self.geos_mean = self.geos_mean.unsqueeze(0)
                    self.geos_std = self.geos_std.unsqueeze(0)
            else:
                 self.geos_mean = None
                 
        # Load Soil Moisture Stats (JSON)
        import json
        sm_stats_path = os.path.join(os.path.dirname(__file__), "sm_stats.json")
        if os.path.exists(sm_stats_path):
            with open(sm_stats_path, 'r') as f:
                sm_stats = json.load(f)
            self.sm_mean = float(sm_stats['sm_mean'])
            self.sm_std = float(sm_stats['sm_std'])
            print(f"Loaded SM Stats: Mean={self.sm_mean:.4f}, Std={self.sm_std:.4f}")
        else:
            self.sm_mean = None
            self.sm_std = None

        # Load IVT Stats (JSON)
        ivt_stats_path = os.path.join(os.path.dirname(__file__), "ivt_stats.json")
        if os.path.exists(ivt_stats_path):
            with open(ivt_stats_path, 'r') as f:
                ivt_stats = json.load(f)
            self.ivt_mean = float(ivt_stats['ivt_mean'])
            self.ivt_std = float(ivt_stats['ivt_std'])
            print(f"Loaded IVT Stats: Mean={self.ivt_mean:.2f}, Std={self.ivt_std:.2f} kg/m/s")
        else:
            print("Warning: ivt_stats.json not found. Run ml_model/calculate_stats_ivt.py first. Using fallback /1000.")
            self.ivt_mean = None
            self.ivt_std = None

    def prepare_samples(self):
        """Indexing all available samples (aggregated by Init Date)."""
        print(f"Indexing samples from {self.years[0]} to {self.years[-1]}...")
        
        all_init_dates = []
        samples_tmp = []
        
        for year in self.years:
            # File paths
            geos_path = os.path.join(self.data_root, f"geos_subc_{year}.zarr")
            gpcp_path = os.path.join(self.data_root, f"gpcp_weekly_{year}.zarr")
            sst_path = os.path.join(self.data_root, f"sst_weekly_{year}.zarr")
            sss_path = os.path.join(self.data_root, f"sss_weekly_{year}.zarr")
            sm_path = os.path.join(self.data_root, f"soilw_weekly_{year}.zarr")
            ivt_path = os.path.join(self.data_root, f"ivt_weekly_{year}.zarr")
            
            # Check existence of core files
            if not os.path.exists(geos_path) or not os.path.exists(gpcp_path):
                continue
                
            # Open Datasets to get dates
            try:
                ds_geos = xr.open_zarr(geos_path, consolidated=False)
                if 'S' not in ds_geos.coords:
                    continue
                
                has_sst = os.path.exists(sst_path)
                has_sss = os.path.exists(sss_path)
                has_sm = os.path.exists(sm_path)
                has_ivt = os.path.exists(ivt_path)
                
                n_samples = ds_geos.sizes['S']
                init_dates = pd.to_datetime(ds_geos['S'].values)
                
                for s_idx, s_date in enumerate(init_dates):
                    all_init_dates.append(s_date)
                    samples_tmp.append({
                        "year": year,
                        "s_idx": s_idx,
                        "date": s_date,
                        "s_key": str(s_date),
                        "geos_path": geos_path,
                        "gpcp_path": gpcp_path,
                        "sst_path": sst_path if has_sst else None,
                        "sss_path": sss_path if has_sss else None,
                        "sm_path": sm_path if has_sm else None,
                        "ivt_path": ivt_path if has_ivt else None
                    })
                
                ds_geos.close()
                
            except Exception as e:
                print(f"Error indexing {year}: {e}")
        
        # Build Prev-Init Map (Logic from dataset.py)
        all_dates_sorted = sorted(set(all_init_dates))
        self.prev_init_map = {}
        for i, s_date in enumerate(all_dates_sorted):
             target = s_date - pd.Timedelta(days=28)
             best_prev = None
             best_diff = pd.Timedelta(days=999)
             for j in range(i - 1, max(i - 8, -1), -1):
                 diff = abs(all_dates_sorted[j] - target)
                 if diff < best_diff:
                     best_diff = diff
                     best_prev = all_dates_sorted[j]
             if best_prev and best_diff <= pd.Timedelta(days=14):
                 self.prev_init_map[str(s_date)] = str(best_prev)
             else:
                 self.prev_init_map[str(s_date)] = None
                 
        self.samples = samples_tmp
        print(f"Found {len(self.samples)} samples. Prev-Init map built.")

    def _preload_data(self):
        print(f"Preloading {len(self.samples)} samples into RAM...")
        self.data_cache = []
        for i in range(len(self.samples)):
            self.data_cache.append(self._load_sample(i))
            if (i+1) % 100 == 0:
                print(f"Loaded {i+1}/{len(self.samples)}")
        print("Preloading complete.")

    def __len__(self):
        return len(self.samples)
        
    def _load_prev_gpcp(self, meta):
        """Load GPCP from previous initialization (Observed State)."""
        s_key = meta['s_key']
        prev_key = self.prev_init_map.get(s_key)
        
        if prev_key:
             prev_date = pd.Timestamp(prev_key)
             # Find file for prev_date year
             path = os.path.join(self.data_root, f"gpcp_weekly_{prev_date.year}.zarr")
             if os.path.exists(path):
                 ds = xr.open_zarr(path, consolidated=False)
                 if 'S' in ds.coords:
                     dates = pd.to_datetime(ds['S'].values)
                     if prev_date in dates:
                         idx = list(dates).index(prev_date)
                         # Load 4 weeks of precip
                         val = ds['precip'].isel(S=idx).values # (L, H, W)
                         ds.close()
                         return val
                 ds.close()
        
        # Fallback: Zeros (L=4, H, W)
        return np.zeros((4, 181, 360), dtype=np.float32)

    def __getitem__(self, idx):
        if self.preload:
            return self.data_cache[idx]
        else:
            return self._load_sample(idx)

    def _load_sample(self, idx):
        meta = self.samples[idx]
        
        # 1. Load GEOS (Dynamic) -> (M, L, H, W)
        ds_geos = xr.open_zarr(meta["geos_path"], consolidated=False)
        geos_data = ds_geos['pr'].isel(S=meta['s_idx']).values 
        ds_geos.close()
        
        geos_tensor = torch.from_numpy(geos_data).float()
        if geos_tensor.ndim == 3: 
             geos_tensor = geos_tensor.unsqueeze(0)
             
        # Add Channel Dim: (M, L, H, W) -> (M, 1, L, H, W)
        geos_tensor = geos_tensor.unsqueeze(1)
        
        # Broadcast normalization
        if self.normalize and self.geos_mean is not None:
             # Ensure stats are broadcastable
             gm = self.geos_mean
             gs = self.geos_std
             
             # If using global scalar stats, gm/gs are scalars or (1,)
             if gm.numel() == 1:
                 geos_tensor = (geos_tensor - gm) / (gs * 3.0)
             else:
                 # Grid Stats (1, L, H, W)
                 if gm.ndim == 3: gm = gm.unsqueeze(0) # (1, L, H, W)
                 if gs.ndim == 3: gs = gs.unsqueeze(0)
                 
                 gm = gm.unsqueeze(1) # (1, 1, L, H, W)
                 gs = gs.unsqueeze(1)
                 
                 geos_tensor = (geos_tensor - gm) / (gs * 3.0)

        # 2. Load Obs (Static/State)
        # SST (4, H, W)
        sst_val = np.zeros((4, 181, 360), dtype=np.float32)
        if meta["sst_path"]:
            ds_sst = xr.open_zarr(meta["sst_path"], consolidated=False)
            v = ds_sst['sst'].isel(S=meta['s_idx']).values
            if v.ndim == 3: sst_val = v
            elif v.ndim == 2: sst_val[:] = v 
            ds_sst.close()
            
        # SSS (4, H, W)
        sss_val = np.zeros((4, 181, 360), dtype=np.float32)
        if meta["sss_path"]:
            ds_sss = xr.open_zarr(meta["sss_path"], consolidated=False)
            v = ds_sss['sss'].isel(S=meta['s_idx']).values
            if v.ndim == 3: sss_val = v
            elif v.ndim == 2: sss_val[:] = v
            ds_sss.close()
            
        # Soil Moisture (4, H, W)
        sm_val = np.zeros((4, 181, 360), dtype=np.float32)
        if meta["sm_path"]:
            ds_sm = xr.open_zarr(meta["sm_path"], consolidated=False)
            var_name = None
            for c in ['sm', 'soil_moisture', 'soilw', 'swvl1', 'var40']:
                if c in ds_sm:
                    var_name = c
                    break
                    
            if var_name:
                v = ds_sm[var_name].isel(S=meta['s_idx']).values
                if v.ndim == 3: sm_val = v
                elif v.ndim == 2: sm_val[:] = v
            ds_sm.close()
            
        # Previous GPCP (4, H, W)
        prev_gpcp_val = self._load_prev_gpcp(meta)
        
        # IVT (4, H, W)
        ivt_val = np.zeros((4, 181, 360), dtype=np.float32)
        if meta.get("ivt_path"):
            ds_ivt = xr.open_zarr(meta["ivt_path"], consolidated=False)
            v = ds_ivt['ivt'].isel(S=meta['s_idx']).values
            if v.ndim == 3: ivt_val = v
            elif v.ndim == 2: ivt_val[:] = v
            ds_ivt.close()

        # Stack Obs along Channel dimension
        # Obs: [SST(4), SSS(4), SM(4), Prev(4), IVT(4)] -> 20 channels
        obs_stack = np.concatenate([sst_val, sss_val, sm_val, prev_gpcp_val, ivt_val], axis=0) 
        
        # Normalize Obs
        # SST (K) ~ 270-310 -> (val - 270) / 40
        obs_stack[0:4] = (obs_stack[0:4] - 273.15) / 30.0 
        # SSS (psu) ~ 30-40 -> (val - 30) / 10
        obs_stack[4:8] = (obs_stack[4:8] - 30.0) / 10.0
        # SM (m3/m3) ~ 0-0.5 -> val / 0.5
        obs_stack[8:12] = obs_stack[8:12] / 0.5
        # Prev GPCP (mm/day) ~ 0-20 -> val / 10
        obs_stack[12:16] = obs_stack[12:16] / 10.0
        # IVT (kg/m/s) — z-score normalization using computed stats
        if self.ivt_mean is not None:
            obs_stack[16:20] = (obs_stack[16:20] - self.ivt_mean) / (self.ivt_std * 3.0 + 1e-6)
        else:
            obs_stack[16:20] = obs_stack[16:20] / 1000.0  # Fallback
        
        obs_tensor = torch.from_numpy(obs_stack).float()
        
        # Normalize Obs
        if self.normalize:
            if hasattr(self, 'obs_mean') and self.obs_mean is not None:
                # Global Z-Score Normalization
                # obs_mean is (16,) -> broadcast to (16, H, W)
                om = self.obs_mean.view(16, 1, 1)
                os_ = self.obs_std.view(16, 1, 1)
                obs_tensor = (obs_tensor - om) / (os_ * 3.0)
            else:
                # Fallback Min-Max scaling
                # SST (K) ~ 270-310 -> (val - 270) / 40
                obs_tensor[0:4] = (obs_tensor[0:4] - 273.15) / 30.0 
                # SSS (psu) ~ 30-40 -> (val - 30) / 10
                obs_tensor[4:8] = (obs_tensor[4:8] - 30.0) / 10.0
                # SM (m3/m3)
                if hasattr(self, 'sm_mean') and self.sm_mean is not None:
                     # Z-Score using computed stats
                     obs_tensor[8:12] = (obs_tensor[8:12] - self.sm_mean) / (self.sm_std * 3.0)
                else:
                     # Fallback Min-Max scaling ~ 0-0.5 -> val / 0.5
                     obs_tensor[8:12] = obs_tensor[8:12] / 0.5
                
                # Prev GPCP (mm/day) ~ 0-20 -> val / 10
                obs_tensor[12:16] = obs_tensor[12:16] / 10.0
        
        # 3. Load Target (GPCP)
        ds_gpcp = xr.open_zarr(meta["gpcp_path"], consolidated=False)
        target_val = ds_gpcp['precip'].isel(S=meta['s_idx']).values 
        ds_gpcp.close()
            
        # Sanitize Inputs (Handle NaNs/Infs in Obs/GEOS)
        # Use torch.nan_to_num on tensors directly
        if torch.isnan(geos_tensor).any() or torch.isinf(geos_tensor).any():
            geos_tensor = torch.nan_to_num(geos_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
            
        if torch.isnan(obs_tensor).any() or torch.isinf(obs_tensor).any():
            obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
            
        target_tensor = torch.from_numpy(target_val).float()
        
        # Handle NaNs and Fill Values in Target
        if torch.isnan(target_tensor).any() or torch.isinf(target_tensor).any():
             target_tensor = torch.nan_to_num(target_tensor, nan=0.0, posinf=100.0, neginf=0.0)
             
        # Clamp negative values to 0
        target_tensor = torch.clamp(target_tensor, min=0.0)
        
        # FINAL SAFETY CLAMP (Prevent severe outliers in inputs)
        # Z-Scores > 20 are likely data errors or land masks
        geos_tensor = torch.clamp(geos_tensor, min=-20.0, max=20.0)
        obs_tensor = torch.clamp(obs_tensor, min=-20.0, max=20.0)
        
        return {
            "x_geos": geos_tensor, # (M, L, H, W)
            "x_obs": obs_tensor,   # (16, H, W)
            "y_target": target_tensor, # (L, H, W)
            "month": meta['date'].month # Int 1-12
        }
