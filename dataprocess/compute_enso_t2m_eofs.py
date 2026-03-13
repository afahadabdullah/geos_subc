#!/usr/bin/env python3
"""
ENSO State × Lead-Week Conditional EOF Computation for T2M
==========================================================
Computes EOF bases of ERA5 2m Temperature anomalies, binned by (ENSO state, lead week).
Uses the CPC Oceanic Niño Index (ONI) with a 1-season lag to prevent data leakage.

Output: enso_t2m_eof_bases.pt containing:
  - eof_bases[(state, lead)]: {eofs: [K, 181, 360], eigenvalues: [K]}
  - climatology: [181, 360]
  - phase_counts: dict of sample counts per (state, lead) category

Usage:
  python compute_enso_t2m_eofs.py --data_dir /path/to/dataprocess --start_year 1999 --end_year 2020
"""

import numpy as np
import torch
import xarray as xr
import pandas as pd
import os
import argparse
from tqdm import tqdm


def parse_oni_index(oni_path):
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
    return oni_lookup


def get_lagged_enso_state(init_date, oni_lookup, threshold=0.5):
    month_to_lagged_season = {
        1:  ('OND', 0),   
        2:  ('NDJ', 0),   
        3:  ('DJF', 0),   
        4:  ('JFM', 0),   
        5:  ('FMA', 0),   
        6:  ('MAM', 0),   
        7:  ('AMJ', 0),   
        8:  ('MJJ', 0),   
        9:  ('JJA', 0),   
        10: ('JAS', 0),   
        11: ('ASO', 0),   
        12: ('SON', 0),   
    }
    
    seas_code, year_offset = month_to_lagged_season[init_date.month]
    lookup_year = init_date.year + year_offset
    
    if init_date.month == 1:
        lookup_year = init_date.year - 1 
    
    oni_val = oni_lookup.get((lookup_year, seas_code), None)
    
    if oni_val is None:
        return 1 
    
    if oni_val < -threshold:
        return 0 
    elif oni_val > threshold:
        return 2 
    else:
        return 1


def compute_eofs_for_indices(all_t2m, indices, climatology, area_weight_flat, n_eofs, H, W):
    n_samples = len(indices)
    
    # Build anomaly matrix
    X = np.zeros((n_samples, H * W), dtype=np.float32)
    for j, idx in enumerate(indices):
        X[j] = (all_t2m[idx]['t2m'] - climatology).flatten()
    
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
    
    print(f"=== ENSO State × Lead-Week EOF Computation (T2M) ===")
    print(f"    Data Dir: {data_dir}")
    print(f"    Years: {start_year}-{end_year}")
    print(f"    N EOFs per category: {n_eofs}")
    print(f"    Min samples per category: {min_samples}")
    
    # 1. Load ONI Index
    oni_path = os.path.join(data_dir, "oni.ascii.txt")
    if not os.path.exists(oni_path):
        raise FileNotFoundError(f"ONI file not found: {oni_path}")
    oni_lookup = parse_oni_index(oni_path)
    
    # 2. Load all T2M weekly temperatures
    print(f"\n--- Loading T2M data ({start_year}-{end_year}) ---")
    all_t2m = []
    
    for year in tqdm(range(start_year, end_year + 1), desc="Loading years"):
        t2m_path = os.path.join(data_dir, f"t2m_weekly_{year}.zarr")
        if not os.path.exists(t2m_path):
            continue
        
        ds = xr.open_zarr(t2m_path, consolidated=False)
        t2m_var = next((v for v in ['t2m', 'target', 'temperature'] if v in ds), list(ds.data_vars)[0])
        init_dates = pd.to_datetime(ds['S'].values)
        
        for s_idx, init_date in enumerate(init_dates):
            for lead_idx in range(4):
                try:
                    t2m = ds[t2m_var].isel(S=s_idx, L=lead_idx).values.T
                    if np.all(np.isnan(t2m)):
                        continue
                    t2m = np.nan_to_num(t2m, nan=0.0)
                    all_t2m.append({
                        'init_date': init_date,
                        'lead_idx': lead_idx,
                        't2m': t2m
                    })
                except Exception:
                    continue
        ds.close()
    
    N = len(all_t2m)
    print(f"    Total samples: {N}")
    
    # 3. Compute climatological mean
    H, W = 181, 360
    climatology = np.zeros((H, W), dtype=np.float64)
    for s in all_t2m:
        climatology += s['t2m']
    climatology = (climatology / N).astype(np.float32)
    print(f"    Climatology mean: {climatology.mean():.2f} K")
    
    # 4. Assign ENSO state and lead to each sample
    print("\n--- Assigning categories (ENSO state × lead) ---")
    state_names = {0: "La Niña", 1: "Neutral", 2: "El Niño"}
    enso_states = np.zeros(N, dtype=np.int32)
    leads = np.zeros(N, dtype=np.int32)
    
    for i, s in enumerate(all_t2m):
        leads[i] = s['lead_idx']
        enso_states[i] = get_lagged_enso_state(s['init_date'], oni_lookup)
    
    # 5. Pre-compute area weights
    lats = np.linspace(-90, 90, H)
    cos_weights = np.sqrt(np.maximum(np.cos(np.deg2rad(lats)), 0))
    area_weight_flat = np.tile(cos_weights[:, np.newaxis], (1, W)).flatten()
    
    # 6. Compute EOFs with 3-level fallback
    print(f"\n--- Computing EOFs (K={n_eofs}) per category ---")
    
    eof_bases = {}
    phase_counts = {}
    state_only_eofs = {}
    all_indices = np.arange(N)
    
    for state in range(3):
        state_indices = np.where(enso_states == state)[0]
        if len(state_indices) >= min_samples:
            eofs, eigenvals, _ = compute_eofs_for_indices(all_t2m, state_indices, climatology, area_weight_flat, n_eofs, H, W)
            state_only_eofs[state] = (eofs, eigenvals)
        else:
            state_only_eofs[state] = None
    
    for state in range(3):
        for lead in range(4):
            key = (state, lead)
            mask = (enso_states == state) & (leads == lead)
            indices = np.where(mask)[0]
            n_samples = len(indices)
            phase_counts[f"enso{state}_l{lead}"] = n_samples
            
            if n_samples >= min_samples:
                eofs, eigenvals, var_exp = compute_eofs_for_indices(all_t2m, indices, climatology, area_weight_flat, n_eofs, H, W)
                fallback = ""
            elif state_only_eofs[state] is not None:
                eofs, eigenvals = state_only_eofs[state]
                var_exp = eigenvals / eigenvals.sum() * 100
                fallback = " [fallback: state-only]"
            else:
                eofs, eigenvals, var_exp = compute_eofs_for_indices(all_t2m, all_indices, climatology, area_weight_flat, n_eofs, H, W)
                fallback = " [fallback: all-data]"
            
            print(f"  ENSO {state_names[state]:>8} Lead {lead}: {n_samples:>4} samples, top-3 var: {var_exp[:3].sum():.1f}%{fallback}")
            
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
    
    output_path = os.path.join(ml_model_dir, "enso_t2m_eof_bases.pt")
    torch.save(output, output_path)
    print(f"\n✅ Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home1/11353/afahad/geos_subc/dataprocess")
    parser.add_argument("--start_year", type=int, default=1999)
    parser.add_argument("--end_year", type=int, default=2021)
    parser.add_argument("--n_eofs", type=int, default=30)
    args = parser.parse_args()
    main(args)
