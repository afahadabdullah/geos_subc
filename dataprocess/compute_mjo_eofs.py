#!/usr/bin/env python3
"""
MJO Phase-Conditional EOF Computation
======================================
Computes EOF bases of GPCP precipitation anomalies, binned by MJO phase.
Uses lagged RMM indices (28 days before init date) to determine phase.

Output: mjo_eof_bases.pt containing:
  - eofs[phase]: [K, 181, 360] tensor of EOF spatial patterns
  - eigenvalues[phase]: [K] tensor of eigenvalues (variance explained)
  - climatology: [181, 360] seasonal climatology (for anomaly computation)
  - phase_counts: dict of sample counts per phase

Phase Categories:
  0: Weak MJO (amplitude < 1.0)
  1-8: Standard RMM MJO phases

Usage:
  python compute_mjo_eofs.py --data_dir /path/to/dataprocess --start_year 1999 --end_year 2020
"""

import numpy as np
import torch
import xarray as xr
import pandas as pd
import os
import argparse
from tqdm import tqdm


def compute_mjo_phase(rmm1, rmm2, amplitude_threshold=1.0):
    """
    Compute MJO phase (0-8) from RMM1 and RMM2.
    Phase 0 = weak MJO (amplitude < threshold).
    Phases 1-8 follow the standard Wheeler-Hendon convention.
    """
    amplitude = np.sqrt(rmm1**2 + rmm2**2)
    
    if np.isnan(rmm1) or np.isnan(rmm2) or amplitude < amplitude_threshold:
        return 0  # Weak/inactive MJO
    
    # Angle in radians, then map to 8 phases
    angle = np.arctan2(rmm2, rmm1)  # [-pi, pi]
    # Shift to [0, 2*pi]
    angle = angle % (2 * np.pi)
    # Map to phases 1-8 (each phase spans 45 degrees)
    phase = int(angle / (2 * np.pi / 8)) + 1
    phase = min(phase, 8)  # Clamp to 8
    return phase


def main(args):
    data_dir = args.data_dir
    start_year = args.start_year
    end_year = args.end_year
    n_eofs = args.n_eofs
    
    print(f"=== MJO Phase-Conditional EOF Computation ===")
    print(f"    Data Dir: {data_dir}")
    print(f"    Years: {start_year}-{end_year}")
    print(f"    N EOFs per phase: {n_eofs}")
    
    # 1. Load MJO RMM data
    mjo_csv = os.path.join(data_dir, "mjo_processed.csv")
    if not os.path.exists(mjo_csv):
        raise FileNotFoundError(f"MJO processed CSV not found: {mjo_csv}. Run download_mjo.py first.")
    
    mjo_df = pd.read_csv(mjo_csv, parse_dates=['S'])
    mjo_df = mjo_df.set_index('S')
    print(f"    MJO data: {len(mjo_df)} init dates, {mjo_df.index.min()} to {mjo_df.index.max()}")
    
    # 2. Load all GPCP weekly precipitation and compute anomalies
    print(f"\n--- Loading GPCP data ({start_year}-{end_year}) ---")
    
    all_precip = []  # List of (init_date, lead_idx, precip_map [181, 360])
    
    for year in tqdm(range(start_year, end_year + 1), desc="Loading years"):
        gpcp_path = os.path.join(data_dir, f"gpcp_weekly_{year}.zarr")
        if not os.path.exists(gpcp_path):
            print(f"  GPCP file not found: {gpcp_path}. Skipping.")
            continue
        
        ds = xr.open_zarr(gpcp_path, consolidated=False)
        gpcp_var = next((v for v in ['precip', 'target', 'total_precipitation'] if v in ds), list(ds.data_vars)[0])
        
        init_dates = pd.to_datetime(ds['S'].values)
        
        for s_idx, init_date in enumerate(init_dates):
            for lead_idx in range(4):
                try:
                    precip = ds[gpcp_var].isel(S=s_idx, L=lead_idx).values  # [181, 360]
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
    
    print(f"    Total samples loaded: {len(all_precip)}")
    
    if len(all_precip) == 0:
        raise ValueError("No GPCP data loaded!")
    
    # 3. Compute climatological mean incrementally (memory-efficient)
    print("\n--- Computing climatological mean ---")
    H, W = 181, 360
    N = len(all_precip)
    climatology = np.zeros((H, W), dtype=np.float64)
    for s in all_precip:
        climatology += s['precip']
    climatology /= N
    climatology = climatology.astype(np.float32)
    print(f"    Climatology shape: {climatology.shape}, mean: {climatology.mean():.2f} mm/day")
    
    # 4. Assign MJO phase to each sample
    print("\n--- Assigning MJO phases ---")
    phases = np.zeros(N, dtype=np.int32)
    for i, s in enumerate(all_precip):
        init_date = s['init_date']
        if init_date in mjo_df.index:
            rmm1 = mjo_df.loc[init_date, 'RMM1_lagged']
            rmm2 = mjo_df.loc[init_date, 'RMM2_lagged']
            if isinstance(rmm1, pd.Series):
                rmm1 = rmm1.iloc[0]
                rmm2 = rmm2.iloc[0]
            phases[i] = compute_mjo_phase(rmm1, rmm2)
        else:
            phases[i] = 0
    
    phase_counts = {}
    for p in range(9):
        count = int((phases == p).sum())
        phase_counts[p] = count
        print(f"    Phase {p}: {count} samples")
    
    # 5. Compute EOFs per phase (memory-efficient: load only one phase at a time)
    print(f"\n--- Computing EOFs (K={n_eofs}) per MJO phase ---")
    
    # Pre-compute area weights
    lats = np.linspace(-90, 90, H)
    cos_weights = np.cos(np.deg2rad(lats))
    cos_weights = np.sqrt(np.maximum(cos_weights, 0))
    area_weight_map = np.tile(cos_weights[:, np.newaxis], (1, W))
    area_weight_flat = area_weight_map.flatten()
    
    eof_bases = {}
    
    for phase in range(9):
        indices = np.where(phases == phase)[0]
        n_samples = len(indices)
        
        if n_samples < n_eofs + 5:
            print(f"  Phase {phase}: Only {n_samples} samples. Using all data as fallback.")
            indices = np.arange(N)
            n_samples = N
        
        # Build anomaly matrix for this phase only (memory-efficient)
        X = np.zeros((n_samples, H * W), dtype=np.float32)
        for j, idx in enumerate(indices):
            X[j] = (all_precip[idx]['precip'] - climatology).flatten()
        
        # Center
        X -= X.mean(axis=0, keepdims=True)
        
        # Area-weight
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
            print(f"    Phase {phase}: Falling back to full SVD")
            U, S, Vt = np.linalg.svd(X_weighted.astype(np.float64), full_matrices=False)
            S = S[:n_eofs]
            Vt = Vt[:n_eofs]
        
        # Remove area weighting
        eofs = Vt / area_weight_flat[np.newaxis, :]
        eofs = eofs.reshape(-1, H, W)
        
        # Eigenvalues
        eigenvalues = (S ** 2) / (n_samples - 1)
        
        # Normalize EOFs
        for k_idx in range(len(eofs)):
            norm = np.sqrt((eofs[k_idx] ** 2).sum())
            if norm > 0:
                eofs[k_idx] /= norm
        
        # Variance explained
        total_var = np.var(X, axis=0).sum()
        var_explained = eigenvalues / total_var * 100
        
        print(f"  Phase {phase}: {n_samples} samples, top-5 var explained: "
              f"{var_explained[:5].sum():.1f}% "
              f"({', '.join(f'{v:.1f}%' for v in var_explained[:5])})")
        
        eof_bases[phase] = {
            'eofs': torch.from_numpy(eofs.astype(np.float32)),
            'eigenvalues': torch.from_numpy(eigenvalues.astype(np.float32)),
        }
        
        # Free memory
        del X, X_weighted
    
    # 7. Save
    output = {
        'eof_bases': eof_bases,
        'climatology': torch.from_numpy(climatology.astype(np.float32)),
        'phase_counts': phase_counts,
        'n_eofs': n_eofs,
        'years': f"{start_year}-{end_year}",
    }
    
    # Save to ml_model directory so it's alongside the model
    ml_model_dir = os.path.join(os.path.dirname(data_dir), "ml_model")
    if not os.path.exists(ml_model_dir):
        ml_model_dir = data_dir  # Fallback
    
    output_path = os.path.join(ml_model_dir, "mjo_eof_bases.pt")
    torch.save(output, output_path)
    print(f"\n✅ Saved MJO EOF bases to: {output_path}")
    print(f"   Total file size: {os.path.getsize(output_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MJO Phase-Conditional EOF bases")
    parser.add_argument("--data_dir", type=str, default="/home1/11353/afahad/geos_subc/dataprocess")
    parser.add_argument("--start_year", type=int, default=1999)
    parser.add_argument("--end_year", type=int, default=2020)
    parser.add_argument("--n_eofs", type=int, default=30, help="Number of EOFs to retain per phase")
    args = parser.parse_args()
    main(args)
