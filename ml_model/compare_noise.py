#!/usr/bin/env python3
"""
Noise Strategy Comparison
=========================
Loads the best model checkpoint, picks 12 monthly init dates from the validation set,
and evaluates CRPS under 4 different ensemble noise strategies:

  1. Pure Random:        noise ~ N(0, 1)
  2. Tightened Random:   noise ~ N(0, 0.3)  (reduced variance → more deterministic)
  3. MJO EOF:            noise from phase×lead conditional EOF subspace
  4. Model Variance:     noise × σ_predicted (variance head scaling)

Prints per-sample and aggregate CRPS for each strategy.

Usage:
  python compare_noise.py --output_dir ml_output_flow4 --year 2021
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import glob
import yaml
import argparse
from torch.utils.data import DataLoader

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))

from flow_matching import FlowMatchingModel, CustomFlowMatcher
from dataset_flow import S2SHybridDataset


def compute_crps(ensemble_preds, target, area_weights):
    """
    Computes CRPS for a small ensemble.
    ensemble_preds: [E, B, C, H, W]
    target: [B, C, H, W]
    area_weights: [1, 1, H, 1]
    """
    mask = ~torch.isnan(target)
    if not mask.any():
        return 0.0
    
    E = ensemble_preds.shape[0]
    diff = torch.abs(ensemble_preds - target.unsqueeze(0))
    mae_term = diff.mean(dim=0)
    
    spread_term = torch.zeros_like(mae_term)
    if E > 1:
        for i in range(E):
            for j in range(E):
                spread_term += torch.abs(ensemble_preds[i] - ensemble_preds[j])
        spread_term = spread_term / (2 * E * E)
    
    crps_map = mae_term - spread_term
    crps_map_clean = torch.where(mask, crps_map, torch.zeros_like(crps_map))
    weights_clean = torch.where(mask, area_weights, torch.zeros_like(area_weights))
    
    weighted_crps = (crps_map_clean * weights_clean).sum() / (weights_clean.sum() + 1e-8)
    return weighted_crps.item()


@torch.no_grad()
def run_single_strategy(model, flow_matcher, batch, device, noise_fn, num_ensemble=8, num_steps=10):
    """
    Run inference with a given noise generation function.
    Returns CRPS for this batch.
    """
    fb_target_norm = batch['y_target'].to(device)
    vB, _, H, W = fb_target_norm.shape
    num_inits = vB // 4
    
    true_target_precip = batch['target_raw_full'][0::4].to(device)  # [num_inits, 4, H, W]
    
    # Build condition tensor
    fx_obs = batch['x_obs'].to(device)
    fx_geos = batch['x_geos'].to(device)
    fx_geos_cat = fx_geos.view(vB, -1, H, W)
    
    f_month = batch['month'].to(device).float()
    fsin_month = torch.sin(2 * np.pi * (f_month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
    fcos_month = torch.cos(2 * np.pi * (f_month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
    
    fl_idx = batch['lead_idx'].to(device).float()
    f_lead_val = (fl_idx / 1.5) - 1.0
    f_lead_channel = f_lead_val.view(vB, 1, 1, 1).expand(vB, 1, H, W)
    
    fx_cond = torch.cat([fx_obs, fx_geos_cat, fsin_month, fcos_month, f_lead_channel], dim=1)
    
    # Expand for ensemble
    fx_cond_expanded = fx_cond.unsqueeze(1).expand(vB, num_ensemble, -1, H, W).reshape(vB * num_ensemble, -1, H, W)
    lead_idx_expanded = batch['lead_idx'].to(device).unsqueeze(1).expand(vB, num_ensemble).reshape(-1).long()
    
    # Generate noise using the provided strategy function
    noise_expanded = noise_fn(vB, num_ensemble, H, W, batch)
    
    # ODE solve (no variance head scaling — we control noise externally for strategies 1-3)
    unwrapped_model = model
    p_x1_expanded = flow_matcher.euler_solve(
        unwrapped_model, noise_expanded, fx_cond_expanded,
        num_steps=num_steps, lead_idx=lead_idx_expanded, apply_flow_variance=False
    )
    
    p_x1_batch = p_x1_expanded.view(vB, num_ensemble, H, W)
    
    # Denormalize: [-1, 1] → [0, sqrt(50)] → precip
    target_sqrt_min, target_sqrt_max = 0.0, 7.071
    week_sqrt = ((p_x1_batch + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
    week_precip = torch.clamp(week_sqrt ** 2, min=0.0)
    
    ensemble_preds_precip = week_precip.transpose(0, 1)  # [E, vB, H, W]
    
    # Reshape to [E, num_inits, 4, H, W]
    ensemble_4L = ensemble_preds_precip.view(num_ensemble, num_inits, 4, H, W)
    full_pred = ensemble_4L.mean(dim=0)  # [num_inits, 4, H, W]
    
    # Area weights
    lats = np.linspace(-90, 90, H)
    cos_weights = np.cos(np.deg2rad(lats))
    cos_weights = np.maximum(cos_weights, 0)
    area_weights = torch.from_numpy(cos_weights).float().to(device)
    area_weights = area_weights / area_weights.sum()
    area_weights = area_weights.view(1, 1, H, 1)
    
    crps = compute_crps(ensemble_4L, true_target_precip, area_weights)
    return crps


@torch.no_grad()
def run_variance_strategy(model, flow_matcher, batch, device, num_ensemble=8, num_steps=10):
    """
    Strategy 4: Model-predicted variance head scaling.
    Uses apply_flow_variance=True in euler_solve.
    """
    fb_target_norm = batch['y_target'].to(device)
    vB, _, H, W = fb_target_norm.shape
    num_inits = vB // 4
    
    true_target_precip = batch['target_raw_full'][0::4].to(device)
    
    fx_obs = batch['x_obs'].to(device)
    fx_geos = batch['x_geos'].to(device)
    fx_geos_cat = fx_geos.view(vB, -1, H, W)
    
    f_month = batch['month'].to(device).float()
    fsin_month = torch.sin(2 * np.pi * (f_month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
    fcos_month = torch.cos(2 * np.pi * (f_month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
    
    fl_idx = batch['lead_idx'].to(device).float()
    f_lead_val = (fl_idx / 1.5) - 1.0
    f_lead_channel = f_lead_val.view(vB, 1, 1, 1).expand(vB, 1, H, W)
    
    fx_cond = torch.cat([fx_obs, fx_geos_cat, fsin_month, fcos_month, f_lead_channel], dim=1)
    
    fx_cond_expanded = fx_cond.unsqueeze(1).expand(vB, num_ensemble, -1, H, W).reshape(vB * num_ensemble, -1, H, W)
    lead_idx_expanded = batch['lead_idx'].to(device).unsqueeze(1).expand(vB, num_ensemble).reshape(-1).long()
    
    # Standard noise — variance head will scale it
    noise_expanded = torch.randn((vB * num_ensemble, 1, H, W), device=device)
    
    # ODE solve WITH variance head scaling
    p_x1_expanded = flow_matcher.euler_solve(
        model, noise_expanded, fx_cond_expanded,
        num_steps=num_steps, lead_idx=lead_idx_expanded, apply_flow_variance=True
    )
    
    p_x1_batch = p_x1_expanded.view(vB, num_ensemble, H, W)
    
    target_sqrt_min, target_sqrt_max = 0.0, 7.071
    week_sqrt = ((p_x1_batch + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
    week_precip = torch.clamp(week_sqrt ** 2, min=0.0)
    
    ensemble_preds_precip = week_precip.transpose(0, 1)
    ensemble_4L = ensemble_preds_precip.view(num_ensemble, num_inits, 4, H, W)
    
    lats = np.linspace(-90, 90, H)
    cos_weights = np.maximum(np.cos(np.deg2rad(lats)), 0)
    area_weights = torch.from_numpy(cos_weights).float().to(device)
    area_weights = area_weights / area_weights.sum()
    area_weights = area_weights.view(1, 1, H, 1)
    
    crps = compute_crps(ensemble_4L, true_target_precip, area_weights)
    return crps


def main():
    parser = argparse.ArgumentParser(description="Compare Noise Strategies")
    parser.add_argument("--output_dir", type=str, default="ml_output_flow4")
    parser.add_argument("--year", type=int, default=2021, help="Validation year")
    parser.add_argument("--num_ensemble", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config_flow.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    print(f"\n{'='*70}")
    print(f"  NOISE STRATEGY COMPARISON — Year: {args.year}")
    print(f"  Ensemble: {args.num_ensemble} members, {args.num_steps} Euler steps")
    print(f"{'='*70}\n")
    
    test_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=args.year,
        end_year=args.year,
        normalize=True,
        preload=config.get("preload", False),
        stats_file="v5_global_stats.pt",
        subsample_monthly=True  # 1 init/month = 12 samples
    )
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    print(f"  Dataset: {len(test_dataset)} samples (monthly subsample)")
    
    # Load model
    model = FlowMatchingModel(in_channels=36, out_channels=1).to(device)
    
    if args.ckpt is None:
        model_paths = glob.glob(os.path.join(output_dir, "best_model_epoch_*_crps_*.pt"))
        if not model_paths:
            raise FileNotFoundError(f"No best models in {output_dir}")
        
        def extract_crps(path):
            try:
                return float(os.path.basename(path).replace('.pt', '').split('_crps_')[-1])
            except:
                return 999.0
        
        best_ckpt = min(model_paths, key=extract_crps)
        print(f"  Checkpoint: {os.path.basename(best_ckpt)}")
        args.ckpt = best_ckpt
    
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    flow_matcher = CustomFlowMatcher(device=device)
    
    # Load EOF bases
    eof_bases_path = os.path.join(os.path.dirname(__file__), "mjo_eof_bases.pt")
    eof_bases = None
    if os.path.exists(eof_bases_path):
        eof_data = torch.load(eof_bases_path, map_location='cpu', weights_only=False)
        eof_bases = eof_data['eof_bases']
        print(f"  EOF bases loaded: {eof_data.get('conditioning', 'phase-only')} format")
    else:
        print(f"  ⚠️ No EOF bases found. Strategy 3 will use isotropic noise.")
    
    # Define 4 noise strategies
    def noise_pure_random(vB, num_ens, H, W, batch):
        """Strategy 1: Pure N(0,1) noise."""
        return torch.randn((vB * num_ens, 1, H, W), device=device)
    
    def noise_tightened(vB, num_ens, H, W, batch):
        """Strategy 2: Tightened noise — N(0, 0.3) for more deterministic predictions."""
        return torch.randn((vB * num_ens, 1, H, W), device=device) * 0.3
    
    def noise_eof(vB, num_ens, H, W, batch):
        """Strategy 3: MJO phase × lead EOF noise."""
        if eof_bases is None:
            return torch.randn((vB * num_ens, 1, H, W), device=device)
        mjo_phases = batch.get('mjo_phase', torch.zeros(vB, dtype=torch.long))
        if not isinstance(mjo_phases, torch.Tensor):
            mjo_phases = torch.tensor(mjo_phases)
        lead_ids = batch['lead_idx']
        if not isinstance(lead_ids, torch.Tensor):
            lead_ids = torch.tensor(lead_ids)
        return flow_matcher.eof_sample(eof_bases, mjo_phases, vB * num_ens, H, W, lead_ids=lead_ids)
    
    strategies = {
        "1. Pure Random N(0,1)": noise_pure_random,
        "2. Tightened N(0,0.3)": noise_tightened,
        "3. MJO EOF (phase×lead)": noise_eof,
    }
    
    # Run comparison
    print(f"\n{'─'*90}")
    print(f"  {'Sample':<10} {'Month':>5} | {'Pure Random':>12} {'Tightened':>12} {'MJO EOF':>12} {'Var Head':>12}")
    print(f"{'─'*90}")
    
    results = {name: [] for name in strategies}
    results["4. Variance Head"] = []
    
    for b_idx, batch in enumerate(test_loader):
        if b_idx >= 12:  # Only 12 monthly samples
            break
        
        month = batch['month'][0].item()
        
        # Run strategies 1-3
        crps_values = {}
        for name, noise_fn in strategies.items():
            crps = run_single_strategy(
                model, flow_matcher, batch, device, noise_fn,
                num_ensemble=args.num_ensemble, num_steps=args.num_steps
            )
            results[name].append(crps)
            crps_values[name] = crps
        
        # Strategy 4: Variance head
        crps_var = run_variance_strategy(
            model, flow_matcher, batch, device,
            num_ensemble=args.num_ensemble, num_steps=args.num_steps
        )
        results["4. Variance Head"].append(crps_var)
        crps_values["4. Variance Head"] = crps_var
        
        # Print row
        print(f"  Batch {b_idx:<4} {month:>5} | "
              f"{crps_values['1. Pure Random N(0,1)']:>12.4f} "
              f"{crps_values['2. Tightened N(0,0.3)']:>12.4f} "
              f"{crps_values['3. MJO EOF (phase×lead)']:>12.4f} "
              f"{crps_values['4. Variance Head']:>12.4f}")
    
    # Aggregate
    print(f"{'─'*90}")
    print(f"  {'MEAN':<10} {'':>5} | ", end="")
    all_names = list(strategies.keys()) + ["4. Variance Head"]
    for name in all_names:
        avg = np.mean(results[name])
        print(f"{avg:>12.4f} ", end="")
    print()
    
    # Find best strategy
    avgs = {name: np.mean(vals) for name, vals in results.items()}
    best = min(avgs, key=avgs.get)
    print(f"\n  🏆 Best Strategy: {best} (CRPS: {avgs[best]:.4f})")
    
    # How many months each strategy wins
    print(f"\n  Per-month wins:")
    n_samples = len(results[all_names[0]])
    for name in all_names:
        wins = sum(1 for i in range(n_samples) 
                   if results[name][i] == min(results[n][i] for n in all_names))
        print(f"    {name:<30} wins {wins}/{n_samples} months")
    
    print(f"\n{'='*90}")
    print(f"  Done! Comparison complete.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
