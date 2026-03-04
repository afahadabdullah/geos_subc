#!/usr/bin/env python3
"""
NAO Phase × Lead-Week Conditional EOF Computation
====================================================
Computes EOF bases of GPCP precipitation anomalies, binned by (NAO phase, lead week).
Uses the CPC monthly NAO index with a 1-month lag to prevent data leakage.

Output: nao_eof_bases.pt containing:
  - eof_bases[(phase, lead)]: {eofs: [K, 181, 360], eigenvalues: [K]}
  - climatology: [181, 360]
  - phase_counts: dict of sample counts per (phase, lead) category

Key Design:
  - 3 NAO phases × 4 lead weeks = 12 categories
  - Phase 0 = Negative (NAO < -0.5), Phase 1 = Neutral, Phase 2 = Positive (NAO > 0.5)
  - Lead 0-3 = forecast weeks 1-4
  - Uses PREVIOUS month's NAO to avoid data leakage
  - Falls back to phase-only EOF if (phase, lead) has < 20 samples
  - Falls back to all-data EOF if phase has < 20 samples

Usage:
  python compute_nao_eofs.py --data_dir /path/to/dataprocess --start_year 1999 --end_year 2020
"""

import numpy as np
import torch
import xarray as xr
import pandas as pd
import os
import argparse
from tqdm import tqdm


def parse_nao_index(nao_path):
    """
    Parse CPC monthly NAO index file (norm.nao.monthly.b5001.current.ascii.table).
    
    Format:
              Jan    Feb    Mar    Apr    May    Jun    Jul    Aug    Sep    Oct    Nov    Dec
    1950   0.92   0.40  -0.36   0.73  -0.59  -0.06  -1.26  -0.05   0.25   0.85  -1.26  -1.02
    
    Returns: dict mapping (year, month) -> NAO index value
    """
    nao_lookup = {}
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    with open(nao_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header line(s)
    for line in lines:
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue  # Skip header
        
        for m_idx, val_str in enumerate(parts[1:13]):
            try:
                val = float(val_str)
                nao_lookup[(year, m_idx + 1)] = val
            except ValueError:
                continue
    
    print(f"  Parsed NAO index: {len(nao_lookup)} monthly values ({min(nao_lookup.keys())[0]}-{max(nao_lookup.keys())[0]})")
    return nao_lookup


def get_lagged_nao_phase(init_date, nao_lookup, threshold=0.5):
    """
    Get NAO phase for an init date using the PREVIOUS month's NAO value.
    This prevents data leakage since the current month's NAO isn't finalized yet.
    
    Returns:
      0 = Negative (NAO < -threshold)
      1 = Neutral  (-threshold <= NAO <= threshold)
      2 = Positive (NAO > threshold)
    """
    # Use previous month
    if init_date.month == 1:
        lag_year, lag_month = init_date.year - 1, 12
    else:
        lag_year, lag_month = init_date.year, init_date.month - 1
    
    nao_val = nao_lookup.get((lag_year, lag_month), None)
    
    if nao_val is None:
        return 1  # Default to neutral if data missing
    
    if nao_val < -threshold:
        return 0  # Negative
    elif nao_val > threshold:
        return 2  # Positive
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
    
    print(f"=== NAO Phase × Lead-Week EOF Computation ===")
    print(f"    Data Dir: {data_dir}")
    print(f"    Years: {start_year}-{end_year}")
    print(f"    N EOFs per category: {n_eofs}")
    print(f"    Min samples per category: {min_samples}")
    
    # 1. Load NAO Index
    nao_path = os.path.join(data_dir, "norm.nao.monthly.b5001.current.ascii.table")
    if not os.path.exists(nao_path):
        raise FileNotFoundError(f"NAO index not found: {nao_path}")
    nao_lookup = parse_nao_index(nao_path)
    
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
    
    # 4. Assign NAO phase and lead to each sample
    print("\n--- Assigning categories (NAO phase × lead) ---")
    phase_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    nao_phases = np.zeros(N, dtype=np.int32)
    leads = np.zeros(N, dtype=np.int32)
    
    for i, s in enumerate(all_precip):
        leads[i] = s['lead_idx']
        nao_phases[i] = get_lagged_nao_phase(s['init_date'], nao_lookup)
    
    # Print distribution
    print(f"\n    {'Category':<25} {'Count':>6}")
    print(f"    {'─'*33}")
    for phase in range(3):
        for lead in range(4):
            count = int(((nao_phases == phase) & (leads == lead)).sum())
            label = f"NAO {phase_names[phase]}, Lead {lead}"
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
    
    # Pre-compute phase-only fallback EOFs
    phase_only_eofs = {}
    all_indices = np.arange(N)
    
    for phase in range(3):
        phase_mask = (nao_phases == phase)
        phase_indices = np.where(phase_mask)[0]
        
        if len(phase_indices) >= min_samples:
            eofs, eigenvals, var_exp = compute_eofs_for_indices(
                all_precip, phase_indices, climatology, area_weight_flat, n_eofs, H, W)
            phase_only_eofs[phase] = (eofs, eigenvals)
        else:
            phase_only_eofs[phase] = None
    
    # Now compute per (phase, lead) with fallback
    for phase in range(3):
        for lead in range(4):
            key = (phase, lead)
            mask = (nao_phases == phase) & (leads == lead)
            indices = np.where(mask)[0]
            n_samples = len(indices)
            phase_counts[f"nao{phase}_l{lead}"] = n_samples
            
            if n_samples >= min_samples:
                eofs, eigenvals, var_exp = compute_eofs_for_indices(
                    all_precip, indices, climatology, area_weight_flat, n_eofs, H, W)
                fallback = ""
            elif phase_only_eofs[phase] is not None:
                eofs, eigenvals = phase_only_eofs[phase]
                var_exp = eigenvals / eigenvals.sum() * 100
                fallback = " [fallback: phase-only]"
            else:
                eofs, eigenvals, var_exp = compute_eofs_for_indices(
                    all_precip, all_indices, climatology, area_weight_flat, n_eofs, H, W)
                fallback = " [fallback: all-data]"
            
            print(f"  NAO {phase_names[phase]:>8} Lead {lead}: {n_samples:>4} samples, "
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
        'conditioning': 'nao_phase_x_lead',
        'phase_names': phase_names,
    }
    
    ml_model_dir = os.path.join(os.path.dirname(data_dir), "ml_model")
    if not os.path.exists(ml_model_dir):
        ml_model_dir = data_dir
    
    output_path = os.path.join(ml_model_dir, "nao_eof_bases.pt")
    torch.save(output, output_path)
    print(f"\n✅ Saved to: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / 1e6:.1f} MB")
    print(f"   Categories: {len(eof_bases)} (3 NAO phases × 4 leads)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home1/11353/afahad/geos_subc/dataprocess")
    parser.add_argument("--start_year", type=int, default=1999)
    parser.add_argument("--end_year", type=int, default=2020)
    parser.add_argument("--n_eofs", type=int, default=30)
    args = parser.parse_args()
    main(args)
