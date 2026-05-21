import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import os
import glob
import pandas as pd
from datetime import datetime, timedelta

GLOBAL_LATS = np.linspace(-90.0, 90.0, 181, dtype=np.float32)
GLOBAL_LONS = np.arange(360, dtype=np.float32)

DEFAULT_TARGET_DOMAINS = {
    # Longitudes use the project-wide 0..359 convention.
    "sa": {
        "label": "South Asia CORDEX-WAS envelope",
        "lat_min": -16.0,
        "lat_max": 45.0,
        "lon_min": 19.0,
        "lon_max": 116.0,
    },
    "sam": {
        "label": "South America CORDEX-SAM envelope",
        "lat_min": -58.0,
        "lat_max": 18.0,
        "lon_min": 254.0,
        "lon_max": 343.0,
    },
    "south_america": {
        "label": "South America CORDEX-SAM envelope",
        "lat_min": -58.0,
        "lat_max": 18.0,
        "lon_min": 254.0,
        "lon_max": 343.0,
    },
    "south_america_cordex_sam": {
        "label": "South America CORDEX-SAM envelope",
        "lat_min": -58.0,
        "lat_max": 18.0,
        "lon_min": 254.0,
        "lon_max": 343.0,
    },
    "south_asia": {
        "label": "South Asia CORDEX-WAS envelope",
        "lat_min": -16.0,
        "lat_max": 45.0,
        "lon_min": 19.0,
        "lon_max": 116.0,
    },
    "sas": {
        "label": "South Asia CORDEX-WAS envelope",
        "lat_min": -16.0,
        "lat_max": 45.0,
        "lon_min": 19.0,
        "lon_max": 116.0,
    },
    "was": {
        "label": "South Asia CORDEX-WAS envelope",
        "lat_min": -16.0,
        "lat_max": 45.0,
        "lon_min": 19.0,
        "lon_max": 116.0,
    },
}


def _normalize_lon(lon):
    return float(lon) % 360.0


def resolve_target_domain(target_domain=None, target_domain_bounds=None):
    if target_domain_bounds:
        bounds = dict(target_domain_bounds)
        label = bounds.get("label") or bounds.get("name") or target_domain or "custom"
    elif target_domain and str(target_domain).lower() not in {"global", "none", "false"}:
        key = str(target_domain).strip().lower()
        if key not in DEFAULT_TARGET_DOMAINS:
            known = ", ".join(sorted(DEFAULT_TARGET_DOMAINS))
            raise ValueError(f"Unknown target_domain={target_domain!r}. Known domains: {known}")
        bounds = DEFAULT_TARGET_DOMAINS[key]
        label = bounds["label"]
    else:
        return None

    lat_min = float(bounds["lat_min"])
    lat_max = float(bounds["lat_max"])
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min

    lon_min = _normalize_lon(bounds["lon_min"])
    lon_max = _normalize_lon(bounds["lon_max"])

    lat_mask = (GLOBAL_LATS >= lat_min) & (GLOBAL_LATS <= lat_max)
    if lon_min <= lon_max:
        lon_mask = (GLOBAL_LONS >= lon_min) & (GLOBAL_LONS <= lon_max)
    else:
        lon_mask = (GLOBAL_LONS >= lon_min) | (GLOBAL_LONS <= lon_max)

    lat_indices = np.where(lat_mask)[0].astype(np.int64)
    lon_indices = np.where(lon_mask)[0].astype(np.int64)
    if lat_indices.size == 0 or lon_indices.size == 0:
        raise ValueError(
            f"Target domain {label!r} selected no grid cells: "
            f"lat={lat_min}..{lat_max}, lon={lon_min}..{lon_max}"
        )

    return {
        "label": label,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_indices": lat_indices,
        "lon_indices": lon_indices,
        "lats": GLOBAL_LATS[lat_indices],
        "lons": GLOBAL_LONS[lon_indices],
    }


def get_target_domain_coords(target_domain=None, target_domain_bounds=None):
    domain_info = resolve_target_domain(target_domain, target_domain_bounds)
    if domain_info is None:
        return GLOBAL_LATS.copy(), GLOBAL_LONS.copy()
    return domain_info["lats"].copy(), domain_info["lons"].copy()


def crop_spatial_to_domain(value, domain_info):
    if domain_info is None:
        return value

    lat_indices = domain_info["lat_indices"]
    lon_indices = domain_info["lon_indices"]

    if isinstance(value, torch.Tensor):
        lat_idx = torch.as_tensor(lat_indices, dtype=torch.long, device=value.device)
        lon_idx = torch.as_tensor(lon_indices, dtype=torch.long, device=value.device)
        value = value.index_select(value.ndim - 2, lat_idx)
        return value.index_select(value.ndim - 1, lon_idx)

    value = np.take(value, lat_indices, axis=-2)
    return np.take(value, lon_indices, axis=-1)


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
                 transform=None, preload=False, normalize=True, stats_file="v5_global_stats.pt",
                 subsample_monthly=False, target_domain=None, target_domain_bounds=None):
        self.data_root = data_root
        self.years = range(start_year, end_year + 1)
        self.transform = transform
        self.preload = preload
        self.normalize = normalize
        self.stats_file = stats_file
        self.subsample_monthly = subsample_monthly
        self.domain_info = resolve_target_domain(target_domain, target_domain_bounds)
        if self.domain_info is None:
            self.lats = GLOBAL_LATS.copy()
            self.lons = GLOBAL_LONS.copy()
        else:
            self.lats = self.domain_info["lats"].copy()
            self.lons = self.domain_info["lons"].copy()
            print(
                f"Target domain: {self.domain_info['label']} "
                f"lat={self.lats[0]:.1f}..{self.lats[-1]:.1f}, "
                f"lon={self.lons[0]:.1f}..{self.lons[-1]:.1f}, "
                f"shape={len(self.lats)}x{len(self.lons)}"
            )
        
        # Load Stats
        if self.normalize:
            self.load_stats()
        
        # Index samples
        self.prepare_samples()
        
        # Load MJO phase lookup for EOF-based noise
        self._load_mjo_phases()
        
        if self.preload:
            self._preload_data()

    def _crop_spatial(self, value):
        return crop_spatial_to_domain(value, self.domain_info)
        
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
            # GEOS Path (Check both S2S legacy and FIMR new format)
            geos_path_s2s = os.path.join(self.data_root, "geos_s2s", f"{year}.zarr")
            geos_path_fimr = os.path.join(self.data_root, f"geos_subc_{year}.zarr")
            geos_path = geos_path_fimr if os.path.exists(geos_path_fimr) else geos_path_s2s
            
            # GPCP Path (Check both formats)
            gpcp_path_old = os.path.join(self.data_root, "gpcp", f"{year}.zarr")
            gpcp_path_new = os.path.join(self.data_root, f"gpcp_weekly_{year}.zarr")
            gpcp_path = gpcp_path_new if os.path.exists(gpcp_path_new) else gpcp_path_old
            
            # Observational Paths (Check both formats)
            sst_path_old = os.path.join(self.data_root, "sst", f"{year}.zarr")
            sst_path_new = os.path.join(self.data_root, f"sst_weekly_{year}.zarr")
            sst_path = sst_path_new if os.path.exists(sst_path_new) else sst_path_old
            
            sss_path_old = os.path.join(self.data_root, "sss", f"{year}.zarr")
            sss_path_new = os.path.join(self.data_root, f"sss_weekly_{year}.zarr")
            sss_path = sss_path_new if os.path.exists(sss_path_new) else sss_path_old
            
            sm_path_old = os.path.join(self.data_root, "soilw", f"{year}.zarr")
            sm_path_new = os.path.join(self.data_root, f"soilw_weekly_{year}.zarr")
            sm_path = sm_path_new if os.path.exists(sm_path_new) else sm_path_old
            
            ivt_path_old = os.path.join(self.data_root, "ivt", f"{year}.zarr")
            ivt_path_new = os.path.join(self.data_root, f"ivt_weekly_{year}.zarr")
            ivt_path = ivt_path_new if os.path.exists(ivt_path_new) else ivt_path_old
            
            mjo_path_old = os.path.join(self.data_root, "mjo", f"{year}.zarr")
            mjo_path_new = os.path.join(self.data_root, f"mjowave_weekly_{year}.zarr")
            mjo_path = mjo_path_new if os.path.exists(mjo_path_new) else mjo_path_old
            
            z500u250_path_old = os.path.join(self.data_root, "z500_u250", f"{year}.zarr")
            z500u250_path_new = os.path.join(self.data_root, f"z500_u250_weekly_{year}.zarr")
            z500u250_path = z500u250_path_new if os.path.exists(z500u250_path_new) else z500u250_path_old
            
            t2m_target_path = os.path.join(self.data_root, f"t2m_weekly_{year}.zarr")
            
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
                    print(f"  T2M : {'OK' if os.path.exists(t2m_target_path) else 'MISSING'} ({os.path.basename(t2m_target_path)})")
                    print("-----------------------------")
                
                n_samples = ds_geos.sizes['S']
                init_dates = pd.to_datetime(ds_geos['S'].values)
                
                seen_months_this_year = set()
                
                for s_idx, s_date in enumerate(init_dates):
                    if self.subsample_monthly:
                        if s_date.month in seen_months_this_year:
                            continue
                        seen_months_this_year.add(s_date.month)
                        
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
                            "z500u250_path": z500u250_path if has_z500u250 else None,
                            "t2m_target_path": t2m_target_path if os.path.exists(t2m_target_path) else None
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

    def _load_mjo_phases(self):
        """Load MJO phase lookup from mjo_processed.csv using lagged RMM indices."""
        self.mjo_phase_map = {}
        mjo_csv = os.path.join(self.data_root, "mjo_processed.csv")
        
        if not os.path.exists(mjo_csv):
            print(f"  ⚠️ MJO CSV not found at {mjo_csv}. All phases default to 0 (weak MJO).")
            return
        
        mjo_df = pd.read_csv(mjo_csv, parse_dates=['S'])
        
        for _, row in mjo_df.iterrows():
            rmm1 = row['RMM1_lagged']
            rmm2 = row['RMM2_lagged']
            init_date_str = str(row['S'])[:10]  # 'YYYY-MM-DD'
            
            if pd.isna(rmm1) or pd.isna(rmm2):
                self.mjo_phase_map[init_date_str] = 0
                continue
            
            amplitude = np.sqrt(rmm1**2 + rmm2**2)
            if amplitude < 1.0:
                self.mjo_phase_map[init_date_str] = 0
            else:
                angle = np.arctan2(rmm2, rmm1) % (2 * np.pi)
                phase = int(angle / (2 * np.pi / 8)) + 1
                self.mjo_phase_map[init_date_str] = min(phase, 8)
        
        # Count distribution
        phase_dist = {}
        for p in self.mjo_phase_map.values():
            phase_dist[p] = phase_dist.get(p, 0) + 1
        print(f"  MJO Phase Map: {len(self.mjo_phase_map)} dates loaded. Distribution: {phase_dist}")


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
            if m.get("t2m_target_path"): handles["t2m_target"] = xr.open_zarr(m["t2m_target_path"], consolidated=False)
            
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
            geos_pr_var = next((v for v in ['pr', 'precip', 'PRECTOT', 'flux_precip'] if v in ds_geos), 'pr')
            geos_tas_var = next((v for v in ['tas', 't2m', 'T2M', 'TAS', 'tempt2m', 'T2MS'] if v in ds_geos), 'tas')
            
            geos_pr_data = ds_geos[geos_pr_var].isel(S=meta['s_idx']).values 
            
            # Retrieve 'tas' if it exists, else use placeholder
            if geos_tas_var in ds_geos:
                geos_tas_data = ds_geos[geos_tas_var].isel(S=meta['s_idx']).values
            else:
                geos_tas_data = np.zeros_like(geos_pr_data)
            
            if close_geos: ds_geos.close()
            
            geos_pr_tensor = torch.from_numpy(geos_pr_data).float()
            geos_tas_tensor = torch.from_numpy(geos_tas_data).float()
            
            if torch.isnan(geos_pr_tensor).any() or torch.isinf(geos_pr_tensor).any():
                geos_pr_tensor = torch.nan_to_num(geos_pr_tensor, nan=0.0, posinf=10.0, neginf=0.0)
            if torch.isnan(geos_tas_tensor).any() or torch.isinf(geos_tas_tensor).any():
                # TAS usually around 200-320K. Set NaNs to ~280K
                geos_tas_tensor = torch.nan_to_num(geos_tas_tensor, nan=280.0, posinf=320.0, neginf=200.0)
                
            # GEOS data from FIMR (2017+) might be deterministic [L, H, W]
            # while earlier GEOS S2S3 had ensembles [M, L, H, W]
            if geos_pr_tensor.ndim == 3: # missing M
                 geos_pr_tensor = geos_pr_tensor.unsqueeze(0) # [1, L, H, W]
                 geos_tas_tensor = geos_tas_tensor.unsqueeze(0)
                 
            # Enforce physical precip minimum (0 mm/day) 
            geos_pr_tensor = torch.clamp(geos_pr_tensor, min=0.0)
            
            # Save raw ensemble for baseline metrics (M, L, H, W)
            geos_ens_pr_raw = geos_pr_tensor.clone()  
            geos_ens_tas_raw = geos_tas_tensor.clone()
                 
            # Take the ensemble mean across the members for model conditioning
            geos_pr_mean = geos_pr_tensor.mean(dim=0) # [L, H, W]
            geos_tas_mean = geos_tas_tensor.mean(dim=0) # [L, H, W]
            
            # Stack into multiple channels: [C=2, L, H, W]
            geos_mean_tensor = torch.stack([geos_pr_mean, geos_tas_mean], dim=0) # [2, L, H, W]
                 
            # Add Channel Dim for conditioning framework: (1, 2, L, H, W)
            geos_cond_tensor = geos_mean_tensor.unsqueeze(0) # [1, 2, L, H, W]
            
            # Standard pure copy of the mean for residual mapping (2, L, H, W)
            pure_geos_mean_raw = geos_mean_tensor.clone() 
            
            if self.normalize and self.bounds is not None:
                g_min = self.bounds["geos_raw"]["min"]
                g_max = self.bounds["geos_raw"]["max"]
                # Normalize pr (channel 0)
                geos_cond_tensor[:, 0] = 2.0 * (torch.clamp(geos_cond_tensor[:, 0], g_min, g_max) - g_min) / (g_max - g_min + 1e-6) - 1.0
                
                # Normalize tas (channel 1) - using estimated physical bounds 200K to 320K for now
                tas_min = 200.0
                tas_max = 320.0
                geos_cond_tensor[:, 1] = 2.0 * (torch.clamp(geos_cond_tensor[:, 1], tas_min, tas_max) - tas_min) / (tas_max - tas_min + 1e-6) - 1.0

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

            geos_cond_tensor = self._crop_spatial(geos_cond_tensor)
            obs_tensor = self._crop_spatial(obs_tensor)
            geos_ens_pr_raw = self._crop_spatial(geos_ens_pr_raw)
            geos_ens_tas_raw = self._crop_spatial(geos_ens_tas_raw)
            pure_geos_mean_raw = self._crop_spatial(pure_geos_mean_raw)

            # Package common features
            cached_common = {
                "geos_cond": geos_cond_tensor,
                "obs_tensor": obs_tensor,
                "geos_ens_pr_raw": geos_ens_pr_raw,
                "geos_ens_tas_raw": geos_ens_tas_raw,
                "pure_geos_mean_raw": pure_geos_mean_raw
            }
            
            if return_common_only:
                return cached_common

        # --- LEAD-SPECIFIC PART ---
        # 3. Load Target (GPCP Precip + ERA5 T2M)
        # load precip
        ds_gpcp, close_gpcp = get_ds(meta["gpcp_path"], "gpcp")
        if ds_gpcp:
            gpcp_var = next((v for v in ['precip', 'target', 'total_precipitation'] if v in ds_gpcp), list(ds_gpcp.data_vars)[0])
            gpcp_val_lead = ds_gpcp[gpcp_var].isel(S=meta['s_idx'], L=meta['lead_idx']).values 
            gpcp_val_raw_full = ds_gpcp[gpcp_var].isel(S=meta['s_idx']).values 
            if close_gpcp: ds_gpcp.close()
        
        # load t2m
        t2m_val_lead = np.zeros_like(gpcp_val_lead)
        t2m_val_raw_full = np.zeros_like(gpcp_val_raw_full)
        if meta.get("t2m_target_path"):
            ds_t2m, close_t2m = get_ds(meta["t2m_target_path"], "t2m_target")
            if ds_t2m:
                t2m_var = next((v for v in ['t2m'] if v in ds_t2m), list(ds_t2m.data_vars)[0])
                v_lead = ds_t2m[t2m_var].isel(S=meta['s_idx'], L=meta['lead_idx']).values 
                v_full = ds_t2m[t2m_var].isel(S=meta['s_idx']).values 
                
                # Handle ERA5 [lon, lat] (360, 181) orientation
                if v_lead.shape == (360, 181):
                    v_lead = v_lead.T
                if v_full.ndim == 3 and v_full.shape[1] == 360 and v_full.shape[2] == 181:
                    v_full = np.transpose(v_full, (0, 2, 1))
                elif v_full.ndim == 2 and v_full.shape == (360, 181):
                    v_full = v_full.T
                    
                t2m_val_lead = v_lead
                t2m_val_raw_full = v_full
                
                if close_t2m: ds_t2m.close()
            
        gpcp_tensor = torch.from_numpy(gpcp_val_lead).float() # (H, W)
        t2m_tensor = torch.from_numpy(t2m_val_lead).float() # (H, W)
        
        gpcp_raw_full = torch.from_numpy(gpcp_val_raw_full).float() # (L=4, H, W)
        t2m_raw_full = torch.from_numpy(t2m_val_raw_full).float() # (L=4, H, W)
        
        gpcp_tensor = torch.nan_to_num(gpcp_tensor, nan=0.0, posinf=100.0, neginf=0.0)
        t2m_tensor = torch.nan_to_num(t2m_tensor, nan=280.0, posinf=320.0, neginf=200.0)
        
        gpcp_raw_full = torch.nan_to_num(gpcp_raw_full, nan=0.0, posinf=100.0, neginf=0.0)
        t2m_raw_full = torch.nan_to_num(t2m_raw_full, nan=280.0, posinf=320.0, neginf=200.0)
        
        gpcp_tensor = torch.clamp(gpcp_tensor, min=0.0)
        gpcp_raw_full = torch.clamp(gpcp_raw_full, min=0.0)
        
        gpcp_raw_lead = gpcp_tensor.clone()
        t2m_raw_lead = t2m_tensor.clone()
        
        if self.normalize and self.bounds is not None:
            # Precip Power Transform: Y = sqrt(GPCP)
            # Map [0, sqrt(50)] -> [-1, 1]
            s_min, s_max = 0.0, 7.071
            gpcp_sqrt = torch.sqrt(gpcp_tensor)
            gpcp_tensor = 2.0 * (torch.clamp(gpcp_sqrt, s_min, s_max) - s_min) / (s_max - s_min + 1e-6) - 1.0
            
            # Temp Transform: min-max [200, 320]
            t_min, t_max = 200.0, 320.0
            t2m_tensor = 2.0 * (torch.clamp(t2m_tensor, t_min, t_max) - t_min) / (t_max - t_min + 1e-6) - 1.0
        
        gpcp_tensor = self._crop_spatial(gpcp_tensor)
        t2m_tensor = self._crop_spatial(t2m_tensor)
        gpcp_raw_lead = self._crop_spatial(gpcp_raw_lead)
        t2m_raw_lead = self._crop_spatial(t2m_raw_lead)
        gpcp_raw_full = self._crop_spatial(gpcp_raw_full)
        t2m_raw_full = self._crop_spatial(t2m_raw_full)

        # Stack targets: [C=2, H, W]
        target_tensor = torch.stack([gpcp_tensor, t2m_tensor], dim=0)
        target_raw_lead = torch.stack([gpcp_raw_lead, t2m_raw_lead], dim=0)
        target_raw_full = torch.stack([gpcp_raw_full, t2m_raw_full], dim=0)

        # Stack GEOS ensemble raw PR and TAS into [M, C=2, L, H, W]
        geos_ens_stacked = torch.stack([cached_common["geos_ens_pr_raw"], cached_common["geos_ens_tas_raw"]], dim=1)
        
        return {
            "x_geos": cached_common["geos_cond"], 
            "x_obs": cached_common["obs_tensor"],
            "y_target": target_tensor,
            "target_raw": target_raw_lead,
            "target_raw_full": target_raw_full,
            "year": meta['date'].year,
            "month": meta['date'].month,
            "day": meta['date'].day,
            "lead_idx": meta['lead_idx'],
            "geos_ens_raw": geos_ens_stacked,  # [M, 2, L, H, W]
            "mjo_phase": self.mjo_phase_map.get(str(meta['date'])[:10], 0)
        }
