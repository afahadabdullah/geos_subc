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
    
    # --- Inline EOF helpers (matching v4 exactly) ---
    def _sample_eof_field(eof_bases, phase, lead, d, H, W):
        """Sample a single noise field from an EOF basis dict (v4-compatible)."""
        key = (phase, lead)
        if key not in eof_bases: key = phase
        if key not in eof_bases: key = (1, lead)
        if key not in eof_bases: return torch.randn(H, W, device=d)
        eofs = eof_bases[key]['eofs'].to(d)
        K = eofs.shape[0]
        alpha = torch.randn(K, device=d)
        noise_field = torch.einsum('k,khw->hw', alpha, eofs)
        std = noise_field.std()
        if std > 1e-6: noise_field = noise_field / std
        return noise_field
    
    def _get_mjo_eof_2ch(vB, E, H, W, b, d):
        """MJO EOF noise, independent per channel (2-ch version of v4's _get_mjo_eof)."""
        noise = torch.zeros((vB*E, 2, H, W), device=d)
        mjo = b.get('mjo_phase', torch.zeros(vB, dtype=torch.long))
        if isinstance(mjo, torch.Tensor): mjo = mjo.clone().detach()
        else: mjo = torch.tensor(mjo)
        lead = b['lead_idx'].clone().detach() if isinstance(b['lead_idx'], torch.Tensor) else torch.tensor(b['lead_idx'])
        for i in range(vB * E):
            b_idx = i // E
            p = int(mjo[b_idx])
            l = int(lead[b_idx])
            for c in range(2):
                noise[i, c] = _sample_eof_field(mjo_bases, p, l, d, H, W)
        return noise
    
    def _get_nao_eof_2ch(vB, E, H, W, b, d):
        """NAO EOF noise, independent per channel."""
        noise = torch.zeros((vB*E, 2, H, W), device=d)
        months = b['month']
        leads = b['lead_idx']
        for i in range(vB * E):
            b_idx = i // E
            month = int(months[b_idx])
            lead = int(leads[b_idx])
            init_date = datetime.date(args.year, month, 15)
            nao_phase = noise_utils.get_nao_phase(init_date, nao_lookup)
            for c in range(2):
                noise[i, c] = _sample_eof_field(nao_bases, nao_phase, lead, d, H, W)
        return noise
    
    def _get_enso_eof_2ch(vB, E, H, W, b, d):
        """ENSO EOF noise, independent per channel."""
        noise = torch.zeros((vB*E, 2, H, W), device=d)
        months = b['month']
        leads = b['lead_idx']
        for i in range(vB * E):
            b_idx = i // E
            month = int(months[b_idx])
            lead = int(leads[b_idx])
            enso_state = noise_utils.get_enso_state(month, args.year, oni_lookup)
            for c in range(2):
                noise[i, c] = _sample_eof_field(enso_bases, enso_state, lead, d, H, W)
        return noise

    def noise_multimodal_dynamic(vB, E, H, W, b, d):
        """
        Dynamically weighted multi-modal blended noise (v4's winning strategy, ported to 2-ch).
        90% amplitude-weighted EOF + 10% isotropic. Normalized to unit variance.
        """
        import pandas as pd
        
        mjo_noise = _get_mjo_eof_2ch(vB, E, H, W, b, d)
        pure_noise = torch.randn((vB*E, 2, H, W), device=d)
        
        nao_noise = _get_nao_eof_2ch(vB, E, H, W, b, d) if (nao_bases is not None and nao_lookup is not None) else torch.randn((vB*E, 2, H, W), device=d)
        enso_noise = _get_enso_eof_2ch(vB, E, H, W, b, d) if (enso_bases is not None and oni_lookup is not None) else torch.randn((vB*E, 2, H, W), device=d)
        
        month = int(b['month'][0])
        init_date = datetime.date(args.year, month, 15)
        
        # MJO amplitude
        mjo_amp = 1.0
        date_str = init_date.strftime('%Y-%m-%d')
        if mjo_df is not None and date_str in mjo_df.index:
            row = mjo_df.loc[date_str]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            r1, r2 = row.get('RMM1_lagged', 0.0), row.get('RMM2_lagged', 0.0)
            if not (pd.isna(r1) or pd.isna(r2)):
                mjo_amp = float(np.sqrt(r1**2 + r2**2))
        
        # NAO amplitude
        nao_amp = abs(noise_utils.get_nao_value(init_date, nao_lookup)) if nao_lookup is not None else 0.5
        
        # ENSO amplitude
        enso_amp = abs(noise_utils.get_enso_value(month, args.year, oni_lookup)) if oni_lookup is not None else 0.5
        
        # Cap / Floor
        mjo_amp = max(min(mjo_amp, 3.0), 0.1)
        nao_amp = max(min(nao_amp, 2.5), 0.1)
        enso_amp = max(min(enso_amp, 2.5), 0.1)
        
        total = mjo_amp + nao_amp + enso_amp
        w_mjo, w_nao, w_enso = mjo_amp / total, nao_amp / total, enso_amp / total
        
        blend = 0.90 * (w_mjo * mjo_noise + w_nao * nao_noise + w_enso * enso_noise) + 0.10 * pure_noise
        std = blend.std(dim=(2, 3), keepdim=True)
        return blend / (std + 1e-6)
    
    def noise_eof_lhs(vB, E, H, W, b, d):
        return noise_utils.generate_dynamic_multimodal_noise(b, E, d, mjo_bases, nao_bases, nao_lookup, enso_bases, oni_lookup, mjo_df, flow_matcher, args.year, use_lhs=True)

    strategies = [
        ("Pure Noise", noise_pure,    False),
        ("EOF Dyn",    noise_multimodal_dynamic, False),
        ("EOF LHS",    noise_eof_lhs, False)
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

    print("\n" + "─"*220)
    header_pr = f"{'PR GEOS':>17} | {'PR Pure':>17} | {'PR EOFDyn':>17} | {'PR EOFLHS':>17}"
    header_t2m = f"{'T2M GEOS':>17} | {'T2M Pure':>17} | {'T2M EOFDyn':>17} | {'T2M EOFLHS':>17}"
    print(f"{'Month':<10} | {header_pr} | {header_t2m}")
    print("─"*220)
    
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
            for i in range(4): # 4 strategies: GEOS, Pure, EOF Dyn, EOF LHS
                c_pr = np.mean([r['pr'][i][0] for r in all_results])
                r_pr = np.mean([r['pr'][i][1] for r in all_results])
                avg_pr.append((c_pr, r_pr))
                
                c_t2m = np.mean([r['t2m'][i][0] for r in all_results])
                r_t2m = np.mean([r['t2m'][i][1] for r in all_results])
                avg_t2m.append((c_t2m, r_t2m))
            fmt_row(f"AVG({n})", avg_pr, avg_t2m)
            print("─"*220)

    # Final Save to CSV
    import pandas as pd
    csv_rows = []
    for r in all_results:
        d = {'month': r['month']}
        strat_names = ['GEOS', 'Pure', 'EOFDyn', 'EOFLHS']
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
