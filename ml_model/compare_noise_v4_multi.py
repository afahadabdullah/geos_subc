#!/usr/bin/env python3
"""
Multi-Variate Noise Strategy Comparison (v4-multi)
==================================================
Evaluates CRPS and RMSE for PR and T2M under 3 ensemble noise strategies:
  0. GEOS Baseline:          Raw GEOS S2S ensemble (10 members in multi-v1)
  1. Pure Random:            noise ~ N(0, 1)
  2. EOF LHS:                EOF-based Latin Hypercube Sampling (MJO+NAO+ENSO)

Usage:
  python compare_noise_v4_multi.py --output_dir ml_output_flowmulti --year 2021
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import yaml
import argparse
import json
import warnings
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import datetime

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, os.path.dirname(__file__))
from train_flow_multiv1 import FlowMatchingModel, CustomFlowMatcher, compute_crps, compute_rmse
from dataset_flow import S2SHybridDataset
import noise_utils

# Styling
BLUE = '\033[94m'
ORANGE = '\033[38;5;214m'
BOLD = '\033[1m'
RESET = '\033[0m'

@torch.no_grad()
def run_strategy(model, flow_matcher, batch, device, num_ensemble, num_steps, noise_fn, use_flow_variance=False):
    model.eval()
    
    vB = batch['y_target'].shape[0] if 'y_target' in batch else batch['input_forecast'].shape[0]
    _, _, H, W = batch['y_target'].shape
    num_inits = vB // 4
    
    true_target_raw = batch['target_raw_full'][0::4].to(device) # [num_inits, 2, 4, H, W]
    true_target_pr = true_target_raw[:, 0]
    true_target_t2m = true_target_raw[:, 1]
    
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
    
    fx_cond_expanded = fx_cond.unsqueeze(1).expand(vB, num_ensemble, -1, H, W).reshape(vB * num_ensemble, -1, H, W).clone()
    lead_idx_expanded = batch['lead_idx'].to(device).unsqueeze(1).expand(vB, num_ensemble).reshape(-1).long()
    
    # Generate noise
    noise_expanded = noise_fn(vB, num_ensemble, H, W, batch, device)
    
    # Solve ODE
    p_x1_expanded = flow_matcher.euler_solve(
        model, noise_expanded, fx_cond_expanded,
        num_steps=num_steps, lead_idx=lead_idx_expanded, apply_flow_variance=use_flow_variance
    )
    
    p_x1_batch = p_x1_expanded.view(vB, num_ensemble, 2, H, W)
    
    # Denormalize PR
    target_sqrt_min, target_sqrt_max = 0.0, 7.071
    p_x1_pr = p_x1_batch[:, :, 0]
    week_sqrt = ((p_x1_pr + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
    week_precip = torch.clamp(week_sqrt ** 2, min=0.0)
    
    # Denormalize T2M
    t2m_min, t2m_max = 200.0, 320.0
    p_x1_t2m = p_x1_batch[:, :, 1]
    week_t2m = ((p_x1_t2m + 1.0) / 2.0) * (t2m_max - t2m_min) + t2m_min
    
    # Reshape for metrics
    ensemble_preds_pr = week_precip.transpose(0, 1).reshape(num_ensemble, num_inits, 4, H, W)
    ensemble_preds_t2m = week_t2m.transpose(0, 1).reshape(num_ensemble, num_inits, 4, H, W)
    
    # Area weights
    lats = np.linspace(-90, 90, H)
    cos_weights = np.maximum(np.cos(np.deg2rad(lats)), 0)
    area_weights = torch.from_numpy(cos_weights).float().to(device)
    area_weights = area_weights / area_weights.sum()
    area_weights = area_weights.view(1, 1, H, 1)
    
    # PR Metrics
    crps_pr = compute_crps(ensemble_preds_pr, true_target_pr, area_weights)
    rmse_pr = compute_rmse(ensemble_preds_pr.mean(dim=0), true_target_pr, area_weights)
    
    # T2M Metrics
    crps_t2m = compute_crps(ensemble_preds_t2m, true_target_t2m, area_weights)
    rmse_t2m = compute_rmse(ensemble_preds_t2m.mean(dim=0), true_target_t2m, area_weights)
    
    return {
        'pr_crps': crps_pr, 'pr_rmse': rmse_pr,
        't2m_crps': crps_t2m, 't2m_rmse': rmse_t2m
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-Variate Noise Comparison")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--num_ensemble", type=int, default=30)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--config", type=str, default="ml_model/config_flow_multiv1.yaml")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    stats_file = config.get("stats_file", "v1_multi_global_stats.pt")
    test_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=args.year, end_year=args.year,
        normalize=True, preload=False,
        stats_file=stats_file, subsample_monthly=True
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Auto-detect best model
    registry_path = os.path.join(args.output_dir, "model_registry.json")
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        best_entry = registry[0]
        ckpt_filename = os.path.basename(best_entry['path'])
        ckpt_path = os.path.join(args.output_dir, ckpt_filename)
        print(f"📂 Auto-selected best model from registry: {ckpt_path} (CRPS: {best_entry['val_loss']:.4f})")
    else:
        ckpt_path = os.path.join(args.output_dir, "best_flow_ckpt.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.output_dir, "latest_flow_ckpt.pt")
        print(f"📂 Model registry not found. Falling back to: {ckpt_path}")
        
    model = FlowMatchingModel(in_channels=41, out_channels=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    flow_matcher = CustomFlowMatcher(device=device)
    
    # EOF Loading
    ml_dir = os.path.dirname(__file__)
    data_dir = config["data_dir"]
    mjo_eof_path = os.path.join(ml_dir, "mjo_eof_bases.pt")
    mjo_bases = torch.load(mjo_eof_path, map_location='cpu', weights_only=False)['eof_bases']
    
    nao_eof_path = os.path.join(ml_dir, "nao_eof_bases.pt")
    nao_idx_path = os.path.join(data_dir, "norm.daily.nao.index.b500101.current.ascii")
    nao_bases = torch.load(nao_eof_path, map_location='cpu', weights_only=False)['eof_bases'] if os.path.exists(nao_eof_path) else None
    nao_lookup = noise_utils.parse_nao_index(nao_idx_path) if os.path.exists(nao_idx_path) else None
    
    enso_eof_path = os.path.join(ml_dir, "enso_eof_bases.pt")
    oni_idx_path = os.path.join(data_dir, "oni.ascii.txt")
    enso_bases = torch.load(enso_eof_path, map_location='cpu', weights_only=False)['eof_bases'] if os.path.exists(enso_eof_path) else None
    oni_lookup = noise_utils.parse_oni_index(oni_idx_path) if os.path.exists(oni_idx_path) else None
    
    mjo_csv_path = os.path.join(data_dir, "mjo_processed.csv")
    mjo_df = None
    if os.path.exists(mjo_csv_path):
        import pandas as pd
        mjo_df = pd.read_csv(mjo_csv_path, parse_dates=['S']).set_index(pd.to_datetime(pd.read_csv(mjo_csv_path)['S']).dt.strftime('%Y-%m-%d'))

    def noise_pure(vB, E, H, W, b, d):
        return torch.randn((vB*E, 2, H, W), device=d)
    
    def noise_eof_lhs(vB, E, H, W, b, d):
        return noise_utils.generate_dynamic_multimodal_noise(b, E, d, mjo_bases, nao_bases, nao_lookup, enso_bases, oni_lookup, mjo_df, flow_matcher, args.year, use_lhs=True)

    strategies = [
        ("Pure Noise", noise_pure,    False),
        ("EOF LHS",    noise_eof_lhs, True)
    ]
    
    def fmt_row(label, pr_vals, t2m_vals):
        """
        pr_vals, t2m_vals: list of tuples (CRPS, RMSE)
        """
        def format_col_group(vals):
            crps_list = [v[0] for v in vals]
            rmse_list = [v[1] for v in vals]
            best_c = np.argmin(crps_list)
            best_r = np.argmin(rmse_list)
            
            parts = []
            for i, (c, r) in enumerate(vals):
                sc = f"{c:>7.4f}"
                sr = f"({r:>7.4f})"
                if i == best_c: sc = f"{BLUE}{BOLD}{sc}{RESET}"
                if i == best_r: sr = f"{ORANGE}{BOLD}{sr}{RESET}"
                parts.append(f"{sc} {sr}")
            return " | ".join(parts)

        s_pr = format_col_group(pr_vals)
        s_t2m = format_col_group(t2m_vals)
        print(f"{label:<10} | {s_pr} | {s_t2m}")

    print("\n" + "─"*180)
    header_pr = f"{'PR GEOS':>17} | {'PR Pure':>17} | {'PR EOF':>17}"
    header_t2m = f"{'T2M GEOS':>17} | {'T2M Pure':>17} | {'T2M EOF':>17}"
    print(f"{'Month':<10} | {header_pr} | {header_t2m}")
    print("─"*180)
    
    all_results = []
    
    for b_idx, batch in enumerate(test_loader):
        month = int(batch['month'][0].item())
        vB = batch['y_target'].shape[0]
        num_inits = vB // 4
        _, _, H, W = batch['y_target'].shape
        
        # Area weights
        lats = np.linspace(-90, 90, H)
        cos_weights = np.maximum(np.cos(np.deg2rad(lats)), 0)
        area_weights = torch.from_numpy(cos_weights).float().to(device)
        area_weights = area_weights / area_weights.sum()
        area_weights = area_weights.view(1, 1, H, 1)
        
        # 0. GEOS Baseline
        true_target_raw = batch['target_raw_full'][0::4].to(device)
        geos_ens_sample = batch['geos_ens_raw'][0::4].to(device)
        
        geos_pr_crps = compute_crps(geos_ens_sample[:, :, 0].transpose(0, 1), true_target_raw[:, 0], area_weights)
        geos_pr_rmse = compute_rmse(geos_ens_sample[:, :, 0].transpose(0, 1).mean(dim=0), true_target_raw[:, 0], area_weights)
        
        geos_t2m_crps = compute_crps(geos_ens_sample[:, :, 1].transpose(0, 1), true_target_raw[:, 1], area_weights)
        geos_t2m_rmse = compute_rmse(geos_ens_sample[:, :, 1].transpose(0, 1).mean(dim=0), true_target_raw[:, 1], area_weights)
        
        # Diagnostic
        if b_idx == 0:
            print(f"\n🔍 [Diagnostic - Month {month}]")
            print(f"   Target PR   : Min={true_target_raw[:, 0].min():.2f}, Max={true_target_raw[:, 0].max():.2f}, Mean={true_target_raw[:, 0].mean():.2f}")
            print(f"   Target T2M  : Min={true_target_raw[:, 1].min():.2f}, Max={true_target_raw[:, 1].max():.2f}, Mean={true_target_raw[:, 1].mean():.2f}")
            print(f"   GEOS PR     : Min={geos_ens_sample[:, :, 0].min():.2f}, Max={geos_ens_sample[:, :, 0].max():.2f}, Mean={geos_ens_sample[:, :, 0].mean():.2f}")
            print(f"   GEOS T2M    : Min={geos_ens_sample[:, :, 1].min():.2f}, Max={geos_ens_sample[:, :, 1].max():.2f}, Mean={geos_ens_sample[:, :, 1].mean():.2f}")
            if geos_ens_sample[:, :, 1].max() < 1e-1:
                print("   ⚠️ WARNING: GEOS T2M appears to be all zeros! Check dataset_flow.py variable detection.")
            print("   " + "─"*50 + "\n")

        row_pr = [(geos_pr_crps, geos_pr_rmse)]
        row_t2m = [(geos_t2m_crps, geos_t2m_rmse)]
        
        for name, fn, use_var in strategies:
            res = run_strategy(model, flow_matcher, batch, device, args.num_ensemble, args.num_steps, fn, use_flow_variance=use_var)
            row_pr.append((res['pr_crps'], res['pr_rmse']))
            row_t2m.append((res['t2m_crps'], res['t2m_rmse']))
            
        all_results.append({'month': month, 'pr': row_pr, 't2m': row_t2m})
        fmt_row(month, row_pr, row_t2m)

        # Running Average Row
        if (b_idx + 1) % 1 == 0:
            n = len(all_results)
            avg_pr = []
            avg_t2m = []
            for i in range(3): # 3 strategies: GEOS, Pure, EOF
                c_pr = np.mean([r['pr'][i][0] for r in all_results])
                r_pr = np.mean([r['pr'][i][1] for r in all_results])
                avg_pr.append((c_pr, r_pr))
                
                c_t2m = np.mean([r['t2m'][i][0] for r in all_results])
                r_t2m = np.mean([r['t2m'][i][1] for r in all_results])
                avg_t2m.append((c_t2m, r_t2m))
            fmt_row(f"AVG({n})", avg_pr, avg_t2m)
            print("─"*180)

    # Final Save to CSV
    import pandas as pd
    csv_rows = []
    for r in all_results:
        d = {'month': r['month']}
        strat_names = ['GEOS', 'Pure', 'EOF']
        for i, name in enumerate(strat_names):
            d[f'PR_{name}_CRPS'] = r['pr'][i][0]
            d[f'PR_{name}_RMSE'] = r['pr'][i][1]
            d[f'T2M_{name}_CRPS'] = r['t2m'][i][0]
            d[f'T2M_{name}_RMSE'] = r['t2m'][i][1]
        csv_rows.append(d)
    
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(args.output_dir, f"noise_comparison_v4_multi_results_{args.year}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Final results saved to: {csv_path}")

if __name__ == "__main__":
    main()
