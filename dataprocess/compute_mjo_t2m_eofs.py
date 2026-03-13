#!/usr/bin/env python3
"""
MJO Phase × Lead-Week Conditional EOF Computation for T2M
=========================================================
Computes EOF bases of ERA5 2m Temperature anomalies, binned by (MJO phase, lead week).
Uses lagged RMM indices (28 days before init date) to determine MJO phase.

Output: mjo_t2m_eof_bases.pt containing:
  - eof_bases[(phase, lead)]: {eofs: [K, 181, 360], eigenvalues: [K]}
  - climatology: [181, 360]
  - phase_counts: dict of sample counts per (phase, lead) category

Usage:
  python compute_mjo_t2m_eofs.py --data_dir /path/to/dataprocess --start_year 1999 --end_year 2020
"""

import numpy as np
import torch
import xarray as xr
import pandas as pd
import os
import argparse
from tqdm import tqdm


def compute_mjo_phase(rmm1, rmm2, amplitude_threshold=1.0):
    amplitude = np.sqrt(rmm1**2 + rmm2**2)
    if np.isnan(rmm1) or np.isnan(rmm2) or amplitude < amplitude_threshold:
        return 0
    angle = np.arctan2(rmm2, rmm1) % (2 * np.pi)
    phase = int(angle / (2 * np.pi / 8)) + 1
    return min(phase, 8)


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
    
    print(f"=== MJO Phase × Lead-Week EOF Computation (T2M) ===")
    print(f"    Data Dir: {data_dir}")
    print(f"    Years: {start_year}-{end_year}")
    print(f"    N EOFs per category: {n_eofs}")
    print(f"    Min samples per category: {min_samples}")
    
    # 1. Load MJO RMM data
    mjo_csv = os.path.join(data_dir, "mjo_processed.csv")
    if not os.path.exists(mjo_csv):
        raise FileNotFoundError(f"MJO CSV not found: {mjo_csv}. Run download_mjo.py first.")
    
    mjo_df = pd.read_csv(mjo_csv, parse_dates=['S'])
    mjo_df = mjo_df.set_index('S')
    print(f"    MJO data: {len(mjo_df)} init dates")
    
    # 2. Load all T2M weekly temperature
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
                    t2m = ds[t2m_var].isel(S=s_idx, L=lead_idx).values.T  # Transpose to (181, 360)
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
    
    # 4. Assign MJO phase and lead to each sample
    print("\n--- Assigning categories (MJO phase × lead) ---")
    phases = np.zeros(N, dtype=np.int32)
    leads = np.zeros(N, dtype=np.int32)
    
    for i, s in enumerate(all_t2m):
        leads[i] = s['lead_idx']
        init_date = s['init_date']
        if init_date in mjo_df.index:
            rmm1 = mjo_df.loc[init_date, 'RMM1_lagged']
            rmm2 = mjo_df.loc[init_date, 'RMM2_lagged']
            if isinstance(rmm1, pd.Series):
                rmm1 = rmm1.iloc[0]
                rmm2 = rmm2.iloc[0]
            phases[i] = compute_mjo_phase(rmm1, rmm2)
    
    # 5. Pre-compute area weights
    lats = np.linspace(-90, 90, H)
    cos_weights = np.sqrt(np.maximum(np.cos(np.deg2rad(lats)), 0))
    area_weight_flat = np.tile(cos_weights[:, np.newaxis], (1, W)).flatten()
    
    # 6. Compute EOFs with 3-level fallback
    print(f"\n--- Computing EOFs (K={n_eofs}) per category ---")
    
    eof_bases = {}
    phase_counts = {}
    phase_only_eofs = {}
    all_indices = np.arange(N)
    
    for phase in range(9):
        phase_indices = np.where(phases == phase)[0]
        if len(phase_indices) >= min_samples:
            eofs, eigenvals, _ = compute_eofs_for_indices(all_t2m, phase_indices, climatology, area_weight_flat, n_eofs, H, W)
            phase_only_eofs[phase] = (eofs, eigenvals)
        else:
            phase_only_eofs[phase] = None
    
    for phase in range(9):
        for lead in range(4):
            key = (phase, lead)
            mask = (phases == phase) & (leads == lead)
            indices = np.where(mask)[0]
            n_samples = len(indices)
            phase_counts[f"p{phase}_l{lead}"] = n_samples
            
            if n_samples >= min_samples:
                eofs, eigenvals, var_exp = compute_eofs_for_indices(all_t2m, indices, climatology, area_weight_flat, n_eofs, H, W)
                fallback = ""
            elif phase_only_eofs[phase] is not None:
                eofs, eigenvals = phase_only_eofs[phase]
                var_exp = eigenvals / eigenvals.sum() * 100
                fallback = " [fallback: phase-only]"
            else:
                eofs, eigenvals, var_exp = compute_eofs_for_indices(all_t2m, all_indices, climatology, area_weight_flat, n_eofs, H, W)
                fallback = " [fallback: all-data]"
            
            print(f"  Phase {phase} Lead {lead}: {n_samples:>4} samples, top-3 var: {var_exp[:3].sum():.1f}%{fallback}")
            
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
        'conditioning': 'mjo_phase_x_lead',
    }
    
    ml_model_dir = os.path.join(os.path.dirname(data_dir), "ml_model")
    if not os.path.exists(ml_model_dir):
        ml_model_dir = data_dir
    
    output_path = os.path.join(ml_model_dir, "mjo_t2m_eof_bases.pt")
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
