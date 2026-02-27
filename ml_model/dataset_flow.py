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
                 transform=None, preload=False, normalize=True, stats_file="v5_global_stats.pt"):
        self.data_root = data_root
        self.years = range(start_year, end_year + 1)
        self.transform = transform
        self.preload = preload
        self.normalize = normalize
        self.stats_file = stats_file
        
        # Load Stats
        if self.normalize:
            self.load_stats()
        
        # Index samples
        self.prepare_samples()
        
        if self.preload:
            self._preload_data()
        
    def load_stats(self):
        # Load Z-Score stats for Precip
        self.bounds = None
        if self.normalize:
            # Look for stats relative to the file location
            full_stats_path = os.path.join(os.path.dirname(__file__), self.stats_file)
            if os.path.exists(full_stats_path):
                self.bounds = torch.load(full_stats_path, weights_only=True)
                print(f"Loaded strict global bounds from {full_stats_path}")
            else:
                print(f"CRITICAL WARNING: {self.stats_file} not found. Normalization enabled but no bounds available!")
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
            mjo_path = os.path.join(self.data_root, f"mjowave_weekly_{year}.zarr")
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
                has_mjo = os.path.exists(mjo_path)
                has_z500u250 = os.path.exists(z500u250_path)
                
                # DIAGNOSTIC: Print once per data_root if files are missing
                if year == self.years[0]:
                    print(f"--- PATH CHECK: {self.data_root} ---")
                    print(f"  GEOS: {'OK' if os.path.exists(geos_path) else 'MISSING'}")
                    print(f"  GPCP: {'OK' if os.path.exists(gpcp_path) else 'MISSING'}")
                    print(f"  SST : {'OK' if has_sst else 'MISSING'} ({os.path.basename(sst_path)})")
                    print(f"  SSS : {'OK' if has_sss else 'MISSING'} ({os.path.basename(sss_path)})")
                    print(f"  SM  : {'OK' if has_sm else 'MISSING'} ({os.path.basename(sm_path)})")
                    print(f"  IVT : {'OK' if has_ivt else 'MISSING'} ({os.path.basename(ivt_path)})")
                    print(f"  MJO : {'OK' if has_mjo else 'MISSING'} ({os.path.basename(mjo_path)})")
                    print(f"  Z500: {'OK' if has_z500u250 else 'MISSING'} ({os.path.basename(z500u250_path)})")
                    print("-----------------------------")
                
                n_samples = ds_geos.sizes['S']
                init_dates = pd.to_datetime(ds_geos['S'].values)
                
                for s_idx, s_date in enumerate(init_dates):
                    all_init_dates.append(s_date)
                    # Every initialization has 4 leads (weeks 1-4)
                    for lead_idx in range(4):
                        samples_tmp.append({
                            "year": year,
                            "s_idx": s_idx,
                            "lead_idx": lead_idx, # NEW: Track specific lead
                            "date": s_date,
                            "s_key": str(s_date),
                            "geos_path": geos_path,
                            "gpcp_path": gpcp_path,
                            "sst_path": sst_path if has_sst else None,
                            "sss_path": sss_path if has_sss else None,
                            "sm_path": sm_path if has_sm else None,
                            "ivt_path": ivt_path if has_ivt else None,
                            "mjo_path": mjo_path if has_mjo else None,
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
        self.data_cache = [None] * len(self.samples)
        
        # Group samples by year to reuse handles
        from collections import defaultdict
        year_to_indices = defaultdict(list)
        for i, meta in enumerate(self.samples):
            year_to_indices[meta['year']].append(i)
            
        loaded_count = 0
        for year in sorted(year_to_indices.keys()):
            indices = year_to_indices[year]
            
            # Group these indices by initialization (s_idx)
            init_to_indices = defaultdict(list)
            for idx in indices:
                init_to_indices[self.samples[idx]['s_idx']].append(idx)
            
            # Open handles for this year
            handles = {}
            m = self.samples[indices[0]]
            if m["geos_path"]: handles["geos"] = xr.open_zarr(m["geos_path"], consolidated=False)
            if m["gpcp_path"]: handles["gpcp"] = xr.open_zarr(m["gpcp_path"], consolidated=False)
            if m["sst_path"]:  handles["sst"]  = xr.open_zarr(m["sst_path"], consolidated=False)
            if m["sss_path"]:  handles["sss"]  = xr.open_zarr(m["sss_path"], consolidated=False)
            if m["sm_path"]:   handles["sm"]   = xr.open_zarr(m["sm_path"], consolidated=False)
            if m["ivt_path"]:  handles["ivt"]  = xr.open_zarr(m["ivt_path"], consolidated=False)
            if m.get("mjo_path"): handles["mjo"] = xr.open_zarr(m["mjo_path"], consolidated=False)
            if m["z500u250_path"]: handles["z500"] = xr.open_zarr(m["z500u250_path"], consolidated=False)
            
            # Load all inits for this year
            for s_idx in sorted(init_to_indices.keys()):
                s_indices = init_to_indices[s_idx]
                
                # Load common features ONCE for all 4 leads
                # We use the first lead sample to trigger the 'common' load
                common_data = self._load_sample(s_indices[0], handles=handles, return_common_only=True)
                
                for idx in s_indices:
                    self.data_cache[idx] = self._load_sample(idx, handles=handles, cached_common=common_data)
                    loaded_count += 1
                    if loaded_count % 100 == 0:
                        print(f"Loaded {loaded_count}/{len(self.samples)}")
            
            # Close handles
            for h in handles.values(): h.close()
                
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
                         # Robust GPCP detection
                         gpcp_var = next((v for v in ['precip', 'target', 'total_precipitation'] if v in ds), list(ds.data_vars)[0])
                         val = ds[gpcp_var].isel(S=idx).values # (L, H, W)
                         ds.close()
                         return val
                 ds.close()
        
        # Fallback: Zeros (L=4, H, W)
        return np.zeros((4, 181, 360), dtype=np.float32)

    def __getitem__(self, idx):
        # We only use cache if it was loaded in the same normalization state as current request.
        # However, to avoid complexity, we just re-load if states don't match.
        if self.preload and self.data_cache:
            # We assume cache was loaded with normalize=True by default in __init__
            # If currently normalize=False (e.g. for diagnostics), we ignore cache.
            if self.normalize == True:
                return self.data_cache[idx]
        
        return self._load_sample(idx)

    def _load_sample(self, idx, handles=None, cached_common=None, return_common_only=False):
        meta = self.samples[idx]
        
        # Helper to get dataset (either from handles or by opening)
        def get_ds(path, key):
            if handles and key in handles:
                return handles[key], False # Return handle, don't close
            if path and os.path.exists(path):
                return xr.open_zarr(path, consolidated=False), True # Return fresh ds, do close
            return None, False

        # If we have cached_common, we skip the heavy lifting
        if cached_common is not None:
            # We still need lead-specific parts
            pass # Continue below to lead-specific part
        else:
            # 1. Load GEOS (Dynamic) -> (M, L, H, W)
            ds_geos, close_geos = get_ds(meta["geos_path"], "geos")
            if ds_geos is None: return None # Safety
            
            # Robust Variable Detection: pr, precip, or PRECTOT
            geos_var = next((v for v in ['pr', 'precip', 'PRECTOT', 'flux_precip'] if v in ds_geos), 'pr')
            geos_data = ds_geos[geos_var].isel(S=meta['s_idx']).values 
            if close_geos: ds_geos.close()
            
            geos_tensor = torch.from_numpy(geos_data).float()
            if torch.isnan(geos_tensor).any() or torch.isinf(geos_tensor).any():
                geos_tensor = torch.nan_to_num(geos_tensor, nan=0.0, posinf=10.0, neginf=0.0)
                
            if geos_tensor.ndim == 3: 
                 geos_tensor = geos_tensor.unsqueeze(0)
                 
            # Enforce physical precipitation minimums (0 mm/day) on raw GEOS fields
            geos_tensor = torch.clamp(geos_tensor, min=0.0)
            
            # Save raw ensemble for baseline metrics (M, L, H, W)
            geos_ens_raw = geos_tensor.clone() 
                 
            # Take the ensemble mean across the members for model conditioning
            geos_mean_tensor = geos_tensor.mean(dim=0, keepdim=True) # [1, L, H, W]
                 
            # Add Channel Dim for conditioning: (1, L, H, W) -> (1, 1, L, H, W)
            geos_cond_tensor = geos_mean_tensor.unsqueeze(1) # [1, 1, L, H, W]
            
            # Standard pure copy of the mean for residual mapping (L, H, W)
            pure_geos_mean_raw = geos_mean_tensor.squeeze(0) 
            
            if self.normalize and self.bounds is not None:
                g_min = self.bounds["geos_raw"]["min"]
                g_max = self.bounds["geos_raw"]["max"]
                geos_cond_tensor = 2.0 * (torch.clamp(geos_cond_tensor, g_min, g_max) - g_min) / (g_max - g_min + 1e-6) - 1.0

            # 2. Load Obs (Static/State)
            # SST (4, H, W)
            # SST (4, H, W) - Ocean Data, Land Masked
            sst_val = np.full((4, 181, 360), np.nan, dtype=np.float32)
            if meta["sst_path"]:
                ds_sst, close_sst = get_ds(meta["sst_path"], "sst")
                if ds_sst:
                    var_name = next((c for c in ['sst', 'SST', 'analysed_sst', 'sea_surface_temperature'] if c in ds_sst), None)
                    if var_name:
                        v = ds_sst[var_name].isel(S=meta['s_idx']).values
                        v = np.squeeze(v)
                        if v.ndim == 3: sst_val = v
                        elif v.ndim == 2: sst_val[:] = v 
                    if close_sst: ds_sst.close()
            # Fill Land mask with 0.0 for SST
            sst_val = np.nan_to_num(sst_val, nan=0.0)
                
            # SSS (4, H, W)
            # SSS (4, H, W) - Ocean Data, Land Masked
            sss_val = np.full((4, 181, 360), np.nan, dtype=np.float32)
            if meta["sss_path"]:
                ds_sss, close_sss = get_ds(meta["sss_path"], "sss")
                if ds_sss:
                    var_name = next((c for c in ['sss', 'SSS', 'sos', 'SOS', 'sea_surface_salinity', 's_surface'] if c in ds_sss), None)
                    if var_name:
                        v = ds_sss[var_name].isel(S=meta['s_idx']).values
                        v = np.squeeze(v)
                        if v.ndim == 3: sss_val = v
                        elif v.ndim == 2: sss_val[:] = v
                    if close_sss: ds_sss.close()
            # Fill Land mask with 25.0 (min SSS)
            sss_val = np.nan_to_num(sss_val, nan=25.0)
            sss_val = np.clip(sss_val, a_min=25.0, a_max=None)
                
            # Soil Moisture (4, H, W)
            # Soil Moisture (4, H, W) - Land Data, Ocean Masked
            sm_val = np.full((4, 181, 360), np.nan, dtype=np.float32)
            if meta["sm_path"]:
                ds_sm, close_sm = get_ds(meta["sm_path"], "sm")
                if ds_sm:
                    var_name = next((c for c in ['sm', 'soil_moisture', 'soilw', 'swvl1', 'var40'] if c in ds_sm), None)
                    if var_name:
                        v = ds_sm[var_name].isel(S=meta['s_idx']).values
                        v = np.squeeze(v)
                        if v.ndim == 3: sm_val = v
                        elif v.ndim == 2: sm_val[:] = v
                    if close_sm: ds_sm.close()
            # Fill Ocean mask with 0.0 (Dry)
            sm_val = np.nan_to_num(sm_val, nan=0.0)
                
            
            # IVT (4, H, W)  — H=181(lat), W=360(lon)
            ivt_val = np.zeros((4, 181, 360), dtype=np.float32)
            if meta.get("ivt_path"):
                ds_ivt, close_ivt = get_ds(meta["ivt_path"], "ivt")
                if ds_ivt:
                    v = ds_ivt['ivt'].isel(S=meta['s_idx']).values
                    if v.ndim == 3:
                        if v.shape[1] == 360 and v.shape[2] == 181:
                            v = np.transpose(v, (0, 2, 1))
                        ivt_val = v
                    elif v.ndim == 2:
                        if v.shape[0] == 360 and v.shape[1] == 181:
                            v = v.T
                        ivt_val[:] = v
                    if close_ivt: ds_ivt.close()

            # Z500 (4, H, W) & U250 (4, H, W)
            z500_val = np.zeros((4, 181, 360), dtype=np.float32)
            u250_val = np.zeros((4, 181, 360), dtype=np.float32)
            if meta.get("z500u250_path"):
                ds_zu, close_zu = get_ds(meta["z500u250_path"], "z500")
                if ds_zu:
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
                    # Z500 clip: ERA5 geopotential is ~50,000 m2/s2, but if stored as meters it's ~5,000
                    # Clip at 0 to avoid land masks without destroying meter-scale data
                    z500_val = np.clip(z500_val, a_min=0.0, a_max=None)
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
                    if close_zu: ds_zu.close()

            # MJO Wave Spatial Envelope (4, H, W)
            mjo_val = np.zeros((4, 181, 360), dtype=np.float32)
            if meta.get("mjo_path"):
                ds_mjo, close_mjo = get_ds(meta["mjo_path"], "mjo")
                if ds_mjo:
                    m_var = next((c for c in ['mjo_wave', 'mjo'] if c in ds_mjo), None)
                    if m_var:
                        v = ds_mjo[m_var].isel(S=meta['s_idx']).values
                        if v.ndim == 3:
                            if v.shape[1] == 360 and v.shape[2] == 181:
                                v = np.transpose(v, (0, 2, 1))
                            mjo_val = v
                        elif v.ndim == 2:
                            if v.shape[0] == 360 and v.shape[1] == 181:
                                v = v.T
                            mjo_val[:] = v
                    if close_mjo: ds_mjo.close()

            # --- Z500 Zonal Deviation (Rossby Wave Tracing) ---
            zonal_mean = z500_val.mean(axis=2, keepdims=True) 
            zonal_dev_val = z500_val - zonal_mean # (4, 181, 360)

            # Stack Obs along Channel dimension
            # Added +4 channels for MJO Wave Spatial Map
            obs_stack = np.concatenate([sst_val, sss_val, sm_val,
                                        ivt_val, zonal_dev_val, u250_val, mjo_val], axis=0) 
            
            if self.normalize and self.bounds is not None:
                def min_max_scale(val, vmin, vmax):
                    return 2.0 * (np.clip(val, vmin, vmax) - vmin) / (vmax - vmin + 1e-6) - 1.0

                obs_stack[0:4]   = min_max_scale(obs_stack[0:4],   self.bounds["sst"]["min"],  self.bounds["sst"]["max"])
                obs_stack[4:8]   = min_max_scale(obs_stack[4:8],   self.bounds["sss"]["min"],  self.bounds["sss"]["max"])
                obs_stack[8:12]  = min_max_scale(obs_stack[8:12],  self.bounds["sm"]["min"],   self.bounds["sm"]["max"])
                obs_stack[12:16] = min_max_scale(obs_stack[12:16], self.bounds["ivt"]["min"],  self.bounds["ivt"]["max"])
                if "z500_zonal_dev" in self.bounds:
                    obs_stack[16:20] = min_max_scale(obs_stack[16:20], self.bounds["z500_zonal_dev"]["min"], self.bounds["z500_zonal_dev"]["max"])
                else:
                    obs_stack[16:20] = min_max_scale(obs_stack[16:20], self.bounds["z500"]["min"], self.bounds["z500"]["max"])
                obs_stack[20:24] = min_max_scale(obs_stack[20:24], self.bounds["u250"]["min"], self.bounds["u250"]["max"])
                # MJO Wave spatial array
                if "mjo" in self.bounds:
                    obs_stack[24:28] = min_max_scale(obs_stack[24:28], self.bounds["mjo"]["min"], self.bounds["mjo"]["max"])
                else:
                    # Fallback min/max anomaly if stats file isn't updated yet
                    obs_stack[24:28] = min_max_scale(obs_stack[24:28], -100.0, 100.0)
            
            obs_tensor = torch.from_numpy(obs_stack).float()
            if torch.isnan(obs_tensor).any() or torch.isinf(obs_tensor).any():
                obs_tensor = torch.nan_to_num(obs_tensor, nan=0.0, posinf=10.0, neginf=-10.0)

            # Package common features
            cached_common = {
                "geos_cond": geos_cond_tensor,
                "obs_tensor": obs_tensor,
                "geos_ens_raw": geos_ens_raw,
                "pure_geos_mean_raw": pure_geos_mean_raw
            }
            
            if return_common_only:
                return cached_common

        # --- LEAD-SPECIFIC PART ---
        # 3. Load Target (GPCP)
        ds_gpcp, close_gpcp = get_ds(meta["gpcp_path"], "gpcp")
        if ds_gpcp:
            gpcp_var = next((v for v in ['precip', 'target', 'total_precipitation'] if v in ds_gpcp), list(ds_gpcp.data_vars)[0])
            target_val_lead = ds_gpcp[gpcp_var].isel(S=meta['s_idx'], L=meta['lead_idx']).values 
            target_val_raw_full = ds_gpcp[gpcp_var].isel(S=meta['s_idx']).values 
            if close_gpcp: ds_gpcp.close()
            
        target_tensor = torch.from_numpy(target_val_lead).float() # (H, W)
        target_raw_full = torch.from_numpy(target_val_raw_full).float() # (L=4, H, W)
        
        target_tensor = torch.nan_to_num(target_tensor, nan=0.0, posinf=100.0, neginf=0.0)
        target_raw_full = torch.nan_to_num(target_raw_full, nan=0.0, posinf=100.0, neginf=0.0)
        target_tensor = torch.clamp(target_tensor, min=0.0)
        target_raw_full = torch.clamp(target_raw_full, min=0.0)
        
        target_raw_lead = target_tensor.clone()
        
        if self.normalize and self.bounds is not None:
            # Power Transform: Y = sqrt(GPCP)
            # Map [0, sqrt(50)] -> [-1, 1]
            s_min, s_max = 0.0, 7.071
            target_sqrt = torch.sqrt(target_tensor)
            target_tensor = 2.0 * (torch.clamp(target_sqrt, s_min, s_max) - s_min) / (s_max - s_min + 1e-6) - 1.0
        
        target_tensor = target_tensor.unsqueeze(0)
        target_raw_lead = target_raw_lead.unsqueeze(0)

        return {
            "x_geos": cached_common["geos_cond"], 
            "x_obs": cached_common["obs_tensor"],
            "y_target": target_tensor,
            "target_raw": target_raw_lead,
            "target_raw_full": target_raw_full,
            "month": meta['date'].month,
            "lead_idx": meta['lead_idx'],
            "geos_ens_raw": cached_common["geos_ens_raw"]
        }
