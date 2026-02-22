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
        self.years = range(start_year, end_year + 1)
        self.transform = transform
        self.preload = preload
        self.normalize = normalize
        
        # Load Stats
        if self.normalize:
            self.load_stats()
        
        # MJO is deliberately removed for V4 architecture as we are isolating atmospheric spatial mechanics.
        self.df_mjo = None
        
        # Index samples
        self.prepare_samples()
        
        if self.preload:
            self._preload_data()
        
    def load_stats(self):
        # Load Z-Score stats for Precip
        self.bounds = None
        if self.normalize:
            stats_file = os.path.join(os.path.dirname(__file__), "v4_global_stats.pt")
            if os.path.exists(stats_file):
                self.bounds = torch.load(stats_file)
                print(f"Loaded strict global bounds from {stats_file}")
            else:
                print(f"CRITICAL WARNING: {stats_file} not found. Normalization enabled but no bounds available!")
                self.bounds = None
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
            z500u250_path = os.path.join(self.data_root, f"z500_u250_weekly_{year}.zarr")
            
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
                has_z500u250 = os.path.exists(z500u250_path)
                
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
                        "ivt_path": ivt_path if has_ivt else None,
                        "z500u250_path": z500u250_path if has_z500u250 else None
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
        if torch.isnan(geos_tensor).any() or torch.isinf(geos_tensor).any():
            geos_tensor = torch.nan_to_num(geos_tensor, nan=0.0, posinf=10.0, neginf=0.0)
            
        if geos_tensor.ndim == 3: 
             geos_tensor = geos_tensor.unsqueeze(0)
             
        # Take the ensemble mean across the 4 members
        geos_tensor = geos_tensor.mean(dim=0, keepdim=True) # [1, L, H, W]
             
        # Add Channel Dim: (M, L, H, W) -> (M, 1, L, H, W)
        geos_tensor = geos_tensor.unsqueeze(1) # [1, 1, L, H, W]
        
        # Enforce physical precipitation minimums (0 mm/day) on incoming raw GEOS fields
        geos_tensor = torch.clamp(geos_tensor, min=0.0)
        
        # Save a pure copy of the raw GEOS field to calculate the exact residual with the GPCP target later
        pure_geos_raw = geos_tensor.clone().squeeze(0).squeeze(0) # [L, H, W]
        
        if self.normalize and self.bounds is not None:
            g_min = self.bounds["geos_raw"]["min"]
            g_max = self.bounds["geos_raw"]["max"]
            geos_tensor = 2.0 * (torch.clamp(geos_tensor, g_min, g_max) - g_min) / (g_max - g_min + 1e-6) - 1.0

        # 2. Load Obs (Static/State)
        # SST (4, H, W)
        sst_val = np.zeros((4, 181, 360), dtype=np.float32)
        if meta["sst_path"]:
            ds_sst = xr.open_zarr(meta["sst_path"], consolidated=False)
            
            # Auto-detect target variable name
            var_name = None
            for c in ['sst', 'SST', 'analysed_sst', 'sea_surface_temperature', 'var']:
                if c in ds_sst:
                    var_name = c
                    break
                    
            if var_name:
                v = ds_sst[var_name].isel(S=meta['s_idx']).values
                v = np.squeeze(v)
                if v.ndim == 3: sst_val = v
                elif v.ndim == 2: sst_val[:] = v 
            ds_sst.close()
            
        # SSS (4, H, W)
        sss_val = np.zeros((4, 181, 360), dtype=np.float32)
        if meta["sss_path"]:
            ds_sss = xr.open_zarr(meta["sss_path"], consolidated=False)
            
            # Auto-detect target variable name 
            var_name = None
            for c in ['sss', 'SSS', 'sos', 'SOS', 'sea_surface_salinity', 's_surface', 'var']:
                if c in ds_sss:
                    var_name = c
                    break
                    
            if var_name:
                v = ds_sss[var_name].isel(S=meta['s_idx']).values
                v = np.squeeze(v)
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
                v = np.squeeze(v)
                if v.ndim == 3: sm_val = v
                elif v.ndim == 2: sm_val[:] = v
            ds_sm.close()
            
        
        # IVT (4, H, W)  — H=181(lat), W=360(lon)
        ivt_val = np.zeros((4, 181, 360), dtype=np.float32)
        if meta.get("ivt_path"):
            ds_ivt = xr.open_zarr(meta["ivt_path"], consolidated=False)
            v = ds_ivt['ivt'].isel(S=meta['s_idx']).values
            if v.ndim == 3:
                if v.shape[1] == 360 and v.shape[2] == 181:
                    v = np.transpose(v, (0, 2, 1))
                ivt_val = v
            elif v.ndim == 2:
                if v.shape[0] == 360 and v.shape[1] == 181:
                    v = v.T
                ivt_val[:] = v
            ds_ivt.close()

        # Z500 (4, H, W) & U250 (4, H, W)
        z500_val = np.zeros((4, 181, 360), dtype=np.float32)
        u250_val = np.zeros((4, 181, 360), dtype=np.float32)
        if meta.get("z500u250_path"):
            ds_zu = xr.open_zarr(meta["z500u250_path"], consolidated=False)
            # Z500
            z_var = next((c for c in ['z500', 'z', 'geopotential'] if c in ds_zu), None)
            if z_var:
                v = ds_zu[z_var].isel(S=meta['s_idx']).values
                if v.ndim == 3:
                    if v.shape[1] == 360 and v.shape[2] == 181:
                        v = np.transpose(v, (0, 2, 1))
                    z500_val = v
                elif v.ndim == 2:
                    if v.shape[0] == 360 and v.shape[1] == 181:
                        v = v.T
                    z500_val[:] = v
            z500_val = np.clip(z500_val, a_min=30000.0, a_max=None)
            # U250
            u_var = next((c for c in ['u250', 'u', 'u_component_of_wind'] if c in ds_zu), None)
            if u_var:
                v = ds_zu[u_var].isel(S=meta['s_idx']).values
                if v.ndim == 3:
                    if v.shape[1] == 360 and v.shape[2] == 181:
                        v = np.transpose(v, (0, 2, 1))
                    u250_val = v
                elif v.ndim == 2:
                    if v.shape[0] == 360 and v.shape[1] == 181:
                        v = v.T
                    u250_val[:] = v
            ds_zu.close()

        # Stack Obs along Channel dimension
        # Obs: [SST(4), SSS(4), SM(4), IVT(4), Z500(4), U250(4)] -> 24 channels
        obs_stack = np.concatenate([sst_val, sss_val, sm_val,
                                    ivt_val, z500_val, u250_val], axis=0) 
        
        if self.normalize and self.bounds is not None:
            # Normalize Obs using Min-Max scaling to [-1, 1] range
            def min_max_scale(val, vmin, vmax):
                return 2.0 * (np.clip(val, vmin, vmax) - vmin) / (vmax - vmin + 1e-6) - 1.0

            obs_stack[0:4]   = min_max_scale(obs_stack[0:4],   self.bounds["sst"]["min"],  self.bounds["sst"]["max"])
            obs_stack[4:8]   = min_max_scale(obs_stack[4:8],   self.bounds["sss"]["min"],  self.bounds["sss"]["max"])
            obs_stack[8:12]  = min_max_scale(obs_stack[8:12],  self.bounds["sm"]["min"],   self.bounds["sm"]["max"])
            obs_stack[12:16] = min_max_scale(obs_stack[12:16], self.bounds["ivt"]["min"],  self.bounds["ivt"]["max"])
            obs_stack[16:20] = min_max_scale(obs_stack[16:20], self.bounds["z500"]["min"], self.bounds["z500"]["max"])
            obs_stack[20:24] = min_max_scale(obs_stack[20:24], self.bounds["u250"]["min"], self.bounds["u250"]["max"])
        
        obs_tensor = torch.from_numpy(obs_stack).float()
        
        if torch.isnan(obs_tensor).any() or torch.isinf(obs_tensor).any():
            obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 3. Load Target (GPCP)
        ds_gpcp = xr.open_zarr(meta["gpcp_path"], consolidated=False)
        target_val = ds_gpcp['precip'].isel(S=meta['s_idx']).values 
        ds_gpcp.close()
            
        target_tensor = torch.from_numpy(target_val).float()
        
        # Handle NaNs and Fill Values in Target
        if torch.isnan(target_tensor).any() or torch.isinf(target_tensor).any():
             target_tensor = torch.nan_to_num(target_tensor, nan=0.0, posinf=100.0, neginf=0.0)
             
        # Clamp negative values to 0
        target_tensor = torch.clamp(target_tensor, min=0.0)
        
        if self.normalize and self.bounds is not None:
            # We want to return the normalized RELATIVE residual directly to the model as our target
            r_min = self.bounds["residual_raw"]["min"]
            r_max = self.bounds["residual_raw"]["max"]
            
            # Extract mathematically sound linear residual
            residual_raw = target_tensor - pure_geos_raw
            
            # y_target becomes the purely normalized linear residual [-1, 1]
            target_tensor = 2.0 * (torch.clamp(residual_raw, r_min, r_max) - r_min) / (r_max - r_min + 1e-6) - 1.0
        else:
            # If not normalizing (e.g. scanning), we just expose the raw GPCP target and do math externally
            pass
        
        # We no longer apply a generic [-20, 20] clamp here because it destroys the true physical values
        # of variables like SST (300) and Z500 (50000) before they can be globally normalized or scanned.
        
        return {
            "x_geos": geos_tensor, # (1, 1, L, H, W)
            "x_obs": obs_tensor,   # (24, H, W)
            "y_target": target_tensor, # (L, H, W)
            "month": meta['date'].month
        }
