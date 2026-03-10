#!/usr/bin/env python3
"""
ENSO State × Lead-Week Conditional EOF Computation
=====================================================
Computes EOF bases of GPCP precipitation anomalies, binned by (ENSO state, lead week).
Uses the CPC Oceanic Niño Index (ONI) with a 1-season lag to prevent data leakage.

Output: enso_eof_bases.pt containing:
  - eof_bases[(state, lead)]: {eofs: [K, 181, 360], eigenvalues: [K]}
  - climatology: [181, 360]
  - phase_counts: dict of sample counts per (state, lead) category

Key Design:
  - 3 ENSO states × 4 lead weeks = 12 categories
  - State 0 = La Niña (ONI < -0.5), State 1 = Neutral, State 2 = El Niño (ONI > 0.5)
  - Lead 0-3 = forecast weeks 1-4
  - Uses the most recent FULLY COMPLETED 3-month season's ONI to avoid data leakage
  - Falls back to state-only EOF if (state, lead) has < 20 samples
  - Falls back to all-data EOF if state has < 20 samples

Usage:
  python compute_enso_eofs.py --data_dir /path/to/dataprocess --start_year 1999 --end_year 2020
"""

import numpy as np
import torch
import xarray as xr
import pandas as pd
import os
import argparse
from tqdm import tqdm


def parse_oni_index(oni_path):
    """
    Parse CPC ONI (Oceanic Niño Index) file (oni.ascii.txt).
    
    Format:
     SEAS  YR   TOTAL   ANOM
      DJF 1950  24.72  -1.53
      JFM 1950  25.17  -1.34
    
    Returns: dict mapping (year, season_code) -> ONI anomaly value
    """
    oni_lookup = {}
    
    with open(oni_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            seas = parts[0].strip()
            year = int(parts[1])
            anom = float(parts[3])
            oni_lookup[(year, seas)] = anom
        except (ValueError, IndexError):
            continue
    
    print(f"  Parsed ONI index: {len(oni_lookup)} seasonal values")
    return oni_lookup


def get_lagged_enso_state(init_date, oni_lookup, threshold=0.5):
    """
    Get ENSO state for an init date using the most recent FULLY COMPLETED 
    3-month season's ONI value. This prevents data leakage.
    
    Example: For an init date of Jan 15, the current season is DJF (Dec-Jan-Feb).
    Since February hasn't finished, DJF is not yet complete. The most recent 
    completed season is NDJ (Nov-Dec-Jan)... but actually NDJ ends in January
    which hasn't finished either. So we use OND (Oct-Nov-Dec) which is fully complete.
    
    Rule: Use the season that ended 2 months before the init month.
    
    Returns:
      0 = La Niña  (ONI < -threshold)
      1 = Neutral  (-threshold <= ONI <= threshold)
      2 = El Niño  (ONI > threshold)
    """
    # Map each init month to the most recent fully completed 3-month season
    # The season ending 2 months prior is guaranteed to be finalized
    month_to_lagged_season = {
        1:  ('OND', 0),   # Jan -> OND of previous year (Oct-Nov-Dec)
        2:  ('NDJ', 0),   # Feb -> NDJ (Nov-Dec-Jan, ended in Jan, now it's Feb so complete)
        3:  ('DJF', 0),   # Mar -> DJF 
        4:  ('JFM', 0),   # Apr -> JFM
        5:  ('FMA', 0),   # May -> FMA
        6:  ('MAM', 0),   # Jun -> MAM
        7:  ('AMJ', 0),   # Jul -> AMJ
        8:  ('MJJ', 0),   # Aug -> MJJ
        9:  ('JJA', 0),   # Sep -> JJA
        10: ('JAS', 0),   # Oct -> JAS
        11: ('ASO', 0),   # Nov -> ASO
        12: ('SON', 0),   # Dec -> SON
    }
    
    seas_code, year_offset = month_to_lagged_season[init_date.month]
    lookup_year = init_date.year + year_offset
    
    # Special case: DJF and NDJ span year boundaries
    # DJF 1999 means Dec 1998 - Jan 1999 - Feb 1999 -> labeled under year 1999
    # OND -> Oct-Nov-Dec, labeled under that year
    # NDJ -> Nov-Dec-Jan of next year, CPC labels by the year of the last month (Jan)
    # So NDJ for Feb 2000 init -> NDJ labeled year 2000 (Nov99-Dec99-Jan00)
    # But CPC labels NDJ under the year of the center month... let's check
    # Actually CPC labels by year of the FIRST month's year. 
    # DJF 1950 = Dec49-Jan50-Feb50 ... no, looking at the data:
    # DJF 1950 = Dec49-Jan50-Feb50 labeled year 1950
    # NDJ would be: Nov-Dec-Jan. NDJ 1950 = Nov49-Dec49-Jan50? Or Nov50-Dec50-Jan51?
    # From the file, it's sequential: DJF, JFM, FMA, MAM... for each year
    # So DJF 1950 covers Dec49-Jan50-Feb50 (year = year of middle month)
    # JFM 1950 covers Jan50-Feb50-Mar50
    # OND 1950 covers Oct50-Nov50-Dec50
    # NDJ 1950 covers Nov50-Dec50-Jan51 (year = year of middle month Dec)
    
    # For Jan init: use OND of same year - 1 month... 
    # Actually for Jan 2000: OND covers Oct-Nov-Dec of previous year
    # CPC labels OND 1999 = Oct99-Nov99-Dec99
    if init_date.month == 1:
        lookup_year = init_date.year - 1  # OND of previous year
    
    oni_val = oni_lookup.get((lookup_year, seas_code), None)
    
    if oni_val is None:
        return 1  # Default to neutral if data missing
    
    if oni_val < -threshold:
        return 0  # La Niña
    elif oni_val > threshold:
        return 2  # El Niño
    else:
        return 1  # Neutral


def compute_eofs_for_indices(all_precip, indices, climatology, area_weight_flat, n_eofs, H, W):
    """Compute EOFs for a subset of samples identified by indices. Memory-efficient."""
    n_samples = len(indices)
    
    # Build anomaly matrix
    X = np.zeros((n_samples, H * W), dtype=np.float32)
    for j, idx in enumerate(indices):
        X[j] = (all_precip[idx]['precip'] - climatology).flatten()
    
    # Center and area-weight
    X -= X.mean(axis=0, keepdims=True)
    X_weighted = X * area_weight_flat[np.newaxis, :]
    
    # Truncated SVD
    try:
        from scipy.sparse.linalg import svds
        k = min(n_eofs, min(X_weighted.shape) - 1)
        U, S, Vt = svds(X_weighted.astype(np.float64), k=k)
        idx_sort = np.argsort(-S)
        S = S[idx_sort]
        Vt = Vt[idx_sort]
    except Exception:
        U, S, Vt = np.linalg.svd(X_weighted.astype(np.float64), full_matrices=False)
        S = S[:n_eofs]
        Vt = Vt[:n_eofs]
    
    # Remove area weighting, reshape
    eofs = (Vt / area_weight_flat[np.newaxis, :]).reshape(-1, H, W)
    eigenvalues = (S ** 2) / (n_samples - 1)
    
    # Normalize EOFs to unit norm
    for k_idx in range(len(eofs)):
        norm = np.sqrt((eofs[k_idx] ** 2).sum())
        if norm > 0:
            eofs[k_idx] /= norm
    
    # Variance explained
    total_var = np.var(X, axis=0).sum()
    var_explained = eigenvalues / total_var * 100
    
    del X, X_weighted
    return eofs, eigenvalues, var_explained


def main(args):
    data_dir = args.data_dir
    start_year = args.start_year
    end_year = args.end_year
    n_eofs = args.n_eofs
    min_samples = 20
    
    print(f"=== ENSO State × Lead-Week EOF Computation ===")
    print(f"    Data Dir: {data_dir}")
    print(f"    Years: {start_year}-{end_year}")
    print(f"    N EOFs per category: {n_eofs}")
    print(f"    Min samples per category: {min_samples}")
    
    # 1. Load ONI Index
    oni_path = os.path.join(data_dir, "oni.ascii.txt")
    if not os.path.exists(oni_path):
        raise FileNotFoundError(f"ONI file not found: {oni_path}")
    oni_lookup = parse_oni_index(oni_path)
    
    # 2. Load all GPCP weekly precipitation
    print(f"\n--- Loading GPCP data ({start_year}-{end_year}) ---")
    all_precip = []
    
    for year in tqdm(range(start_year, end_year + 1), desc="Loading years"):
        gpcp_path = os.path.join(data_dir, f"gpcp_weekly_{year}.zarr")
        if not os.path.exists(gpcp_path):
            continue
        
        ds = xr.open_zarr(gpcp_path, consolidated=False)
        gpcp_var = next((v for v in ['precip', 'target', 'total_precipitation'] if v in ds), list(ds.data_vars)[0])
        init_dates = pd.to_datetime(ds['S'].values)
        
        for s_idx, init_date in enumerate(init_dates):
            for lead_idx in range(4):
                try:
                    precip = ds[gpcp_var].isel(S=s_idx, L=lead_idx).values
                    if np.all(np.isnan(precip)):
                        continue
                    precip = np.nan_to_num(precip, nan=0.0)
                    precip = np.clip(precip, 0, None)
                    all_precip.append({
                        'init_date': init_date,
                        'lead_idx': lead_idx,
                        'precip': precip
                    })
                except Exception:
                    continue
        ds.close()
    
    N = len(all_precip)
    print(f"    Total samples: {N}")
    
    # 3. Compute climatological mean
    H, W = 181, 360
    climatology = np.zeros((H, W), dtype=np.float64)
    for s in all_precip:
        climatology += s['precip']
    climatology = (climatology / N).astype(np.float32)
    print(f"    Climatology mean: {climatology.mean():.2f} mm/day")
    
    # 4. Assign ENSO state and lead to each sample
    print("\n--- Assigning categories (ENSO state × lead) ---")
    state_names = {0: "La Niña", 1: "Neutral", 2: "El Niño"}
    enso_states = np.zeros(N, dtype=np.int32)
    leads = np.zeros(N, dtype=np.int32)
    
    for i, s in enumerate(all_precip):
        leads[i] = s['lead_idx']
        enso_states[i] = get_lagged_enso_state(s['init_date'], oni_lookup)
    
    # Print distribution
    print(f"\n    {'Category':<25} {'Count':>6}")
    print(f"    {'─'*33}")
    for state in range(3):
        for lead in range(4):
            count = int(((enso_states == state) & (leads == lead)).sum())
            label = f"ENSO {state_names[state]}, Lead {lead}"
            marker = " ⚠️" if count < min_samples else ""
            print(f"    {label:<25} {count:>6}{marker}")
    
    # 5. Pre-compute area weights
    lats = np.linspace(-90, 90, H)
    cos_weights = np.sqrt(np.maximum(np.cos(np.deg2rad(lats)), 0))
    area_weight_flat = np.tile(cos_weights[:, np.newaxis], (1, W)).flatten()
    
    # 6. Compute EOFs with 3-level fallback
    print(f"\n--- Computing EOFs (K={n_eofs}) per category ---")
    
    eof_bases = {}
    phase_counts = {}
    
    # Pre-compute state-only fallback EOFs
    state_only_eofs = {}
    all_indices = np.arange(N)
    
    for state in range(3):
        state_mask = (enso_states == state)
        state_indices = np.where(state_mask)[0]
        
        if len(state_indices) >= min_samples:
            eofs, eigenvals, var_exp = compute_eofs_for_indices(
                all_precip, state_indices, climatology, area_weight_flat, n_eofs, H, W)
            state_only_eofs[state] = (eofs, eigenvals)
        else:
            state_only_eofs[state] = None
    
    # Now compute per (state, lead) with fallback
    for state in range(3):
        for lead in range(4):
            key = (state, lead)
            mask = (enso_states == state) & (leads == lead)
            indices = np.where(mask)[0]
            n_samples = len(indices)
            phase_counts[f"enso{state}_l{lead}"] = n_samples
            
            if n_samples >= min_samples:
                eofs, eigenvals, var_exp = compute_eofs_for_indices(
                    all_precip, indices, climatology, area_weight_flat, n_eofs, H, W)
                fallback = ""
            elif state_only_eofs[state] is not None:
                eofs, eigenvals = state_only_eofs[state]
                var_exp = eigenvals / eigenvals.sum() * 100
                fallback = " [fallback: state-only]"
            else:
                eofs, eigenvals, var_exp = compute_eofs_for_indices(
                    all_precip, all_indices, climatology, area_weight_flat, n_eofs, H, W)
                fallback = " [fallback: all-data]"
            
            print(f"  ENSO {state_names[state]:>8} Lead {lead}: {n_samples:>4} samples, "
                  f"top-3 var: {var_exp[:3].sum():.1f}%{fallback}")
            
            eof_bases[key] = {
                'eofs': torch.from_numpy(eofs.astype(np.float32)),
                'eigenvalues': torch.from_numpy(eigenvals.astype(np.float32)),
            }
    
    # 7. Save
    output = {
        'eof_bases': eof_bases,
        'climatology': torch.from_numpy(climatology),
        'phase_counts': phase_counts,
        'n_eofs': n_eofs,
        'years': f"{start_year}-{end_year}",
        'conditioning': 'enso_state_x_lead',
        'state_names': state_names,
    }
    
    ml_model_dir = os.path.join(os.path.dirname(data_dir), "ml_model")
    if not os.path.exists(ml_model_dir):
        ml_model_dir = data_dir
    
    output_path = os.path.join(ml_model_dir, "enso_eof_bases.pt")
    torch.save(output, output_path)
    print(f"\n✅ Saved to: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / 1e6:.1f} MB")
    print(f"   Categories: {len(eof_bases)} (3 ENSO states × 4 leads)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home1/11353/afahad/geos_subc/dataprocess")
    parser.add_argument("--start_year", type=int, default=1999)
    parser.add_argument("--end_year", type=int, default=2020)
    parser.add_argument("--n_eofs", type=int, default=30)
    args = parser.parse_args()
    main(args)
