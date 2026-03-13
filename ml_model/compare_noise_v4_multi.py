#!/usr/bin/env python3
"""
Multi-Variate Noise Strategy Comparison (v4-multi)
====================================================
Direct adaptation of compare_noise_v4.py for the 2-channel (PR + T2M) model.
Uses the SAME noise_utils.py (1-channel) by running each EOF pipeline
independently for PR and T2M, then concatenating.

Usage:
  python compare_noise_v4_multi.py --output_dir ml_output_flowmulti --year 2021
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import glob
import yaml
import argparse
import warnings
import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, os.path.dirname(__file__))
from flow_matching_multi import FlowMatchingModel, CustomFlowMatcher
from dataset_flow import S2SHybridDataset
from train_flow_multiv1 import compute_crps, compute_rmse

# ─── Index Parsers (same as v4) ───

def parse_nao_index(nao_path):
    nao_lookup = {}
    with open(nao_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) < 4: continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            val = float(parts[3])
            d = datetime.date(year, month, day)
            nao_lookup[d] = val
        except ValueError:
            continue
    return nao_lookup

def parse_oni_index(oni_path):
    oni_lookup = {}
    with open(oni_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) < 4: continue
        try:
            oni_lookup[(int(parts[1]), parts[0].strip())] = float(parts[3])
        except (ValueError, IndexError):
            continue
    return oni_lookup

def get_nao_phase(init_date, nao_lookup, threshold=0.5):
    import pandas as pd
    base_date = init_date.date() if isinstance(init_date, pd.Timestamp) else init_date
    vals = []
    for lag in range(1, 8):
        d = base_date - datetime.timedelta(days=lag)
        if d in nao_lookup: vals.append(nao_lookup[d])
    if not vals: return 1
    val = sum(vals) / len(vals)
    if val < -threshold: return 0
    elif val > threshold: return 2
    return 1

def get_nao_value(init_date, nao_lookup):
    import pandas as pd
    base_date = init_date.date() if isinstance(init_date, pd.Timestamp) else init_date
    vals = []
    for lag in range(1, 8):
        d = base_date - datetime.timedelta(days=lag)
        if d in nao_lookup: vals.append(nao_lookup[d])
    if not vals: return 0.0
    return sum(vals) / len(vals)

def get_enso_state(month, year, oni_lookup, threshold=0.5):
    month_to_season = {
        1: ('OND', -1), 2: ('NDJ', 0), 3: ('DJF', 0), 4: ('JFM', 0),
        5: ('FMA', 0), 6: ('MAM', 0), 7: ('AMJ', 0), 8: ('MJJ', 0),
        9: ('JJA', 0), 10: ('JAS', 0), 11: ('ASO', 0), 12: ('SON', 0),
    }
    seas, yr_off = month_to_season[month]
    lookup_year = year + yr_off
    if month == 1: lookup_year = year - 1
    val = oni_lookup.get((lookup_year, seas), 0.0)
    if val < -threshold: return 0
    elif val > threshold: return 2
    return 1

def get_enso_value(month, year, oni_lookup):
    month_to_season = {
        1: ('OND', -1), 2: ('NDJ', 0), 3: ('DJF', 0), 4: ('JFM', 0),
        5: ('FMA', 0), 6: ('MAM', 0), 7: ('AMJ', 0), 8: ('MJJ', 0),
        9: ('JJA', 0), 10: ('JAS', 0), 11: ('ASO', 0), 12: ('SON', 0),
    }
    seas, yr_off = month_to_season[month]
    lookup_year = year + yr_off
    if month == 1: lookup_year = year - 1
    return oni_lookup.get((lookup_year, seas), 0.0)

def sample_from_eof_basis(eof_bases, phase, lead, device, H, W):
    key = (phase, lead)
    if key not in eof_bases: key = phase
    if key not in eof_bases: key = (1, lead)
    if key not in eof_bases: return torch.randn(H, W, device=device)
    eofs = eof_bases[key]['eofs'].to(device)
    K = eofs.shape[0]
    alpha = torch.randn(K, device=device)
    noise_field = torch.einsum('k,khw->hw', alpha, eofs)
    std = noise_field.std()
    if std > 1e-6: noise_field = noise_field / std
    return noise_field


# ─── Core Inference Runner (2-channel adaptation of v4) ───

@torch.no_grad()
def run_strategy(model, flow_matcher, batch, device, num_ensemble, num_steps, noise_fn, use_var_head=False, perturb_cond=False):
    model.eval()
    
    vB = batch['y_target'].shape[0]
    _, _, H, W = batch['y_target'].shape
    num_inits = vB // 4
    
    # Raw targets: [num_inits, 2, 4, H, W] -> separate PR and T2M
    true_target_raw = batch['target_raw_full'][0::4].to(device)
    true_target_pr = true_target_raw[:, 0]    # [num_inits, 4, H, W]
    true_target_t2m = true_target_raw[:, 1]   # [num_inits, 4, H, W]
    
    # Build conditioning (identical to v4)
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
    
    # Generate 2-channel noise
    noise_expanded = noise_fn(vB, num_ensemble, H, W, batch, device)
    
    if perturb_cond:
        # Perturb only channel 0 of noise (PR-like) onto atmospheric dynamics channels
        fx_cond_expanded[:, 4:6, :, :] += (noise_expanded[:, 0:1, :, :] * 0.10)
    
    # Solve ODE -> output [vB*E, 2, H, W]
    p_x1_expanded = flow_matcher.euler_solve(
        model, noise_expanded, fx_cond_expanded,
        num_steps=num_steps, lead_idx=lead_idx_expanded, apply_flow_variance=use_var_head
    )
    
    # Separate channels
    p_x1_batch = p_x1_expanded.view(vB, num_ensemble, 2, H, W)
    
    # Denormalize PR (channel 0) - same as v4
    target_sqrt_min, target_sqrt_max = 0.0, 7.071
    p_x1_pr = p_x1_batch[:, :, 0]
    week_sqrt = ((p_x1_pr + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
    week_precip = torch.clamp(week_sqrt ** 2, min=0.0)
    
    # Denormalize T2M (channel 1)
    t2m_min, t2m_max = 200.0, 320.0
    p_x1_t2m = p_x1_batch[:, :, 1]
    week_t2m = ((p_x1_t2m + 1.0) / 2.0) * (t2m_max - t2m_min) + t2m_min
    
    # Reshape to [E, num_inits, 4, H, W]
    ensemble_pr = week_precip.transpose(0, 1).reshape(num_ensemble, num_inits, 4, H, W)
    ensemble_t2m = week_t2m.transpose(0, 1).reshape(num_ensemble, num_inits, 4, H, W)
    
    # Area weights
    lats = np.linspace(-90, 90, H)
    cos_weights = np.maximum(np.cos(np.deg2rad(lats)), 0)
    area_weights = torch.from_numpy(cos_weights).float().to(device)
    area_weights = area_weights / area_weights.sum()
    area_weights = area_weights.view(1, 1, H, 1)
    
    # PR metrics
    pr_crps = compute_crps(ensemble_pr, true_target_pr, area_weights)
    pr_rmse = compute_rmse(ensemble_pr.mean(dim=0), true_target_pr, area_weights)
    
    # T2M metrics
    t2m_crps = compute_crps(ensemble_t2m, true_target_t2m, area_weights)
    t2m_rmse = compute_rmse(ensemble_t2m.mean(dim=0), true_target_t2m, area_weights)
    
    return {
        'pr_crps': pr_crps, 'pr_rmse': pr_rmse,
        't2m_crps': t2m_crps, 't2m_rmse': t2m_rmse,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Variate Noise Comparison (v4-multi)")
    parser.add_argument("--output_dir", type=str, default="ml_output_flowmulti")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--num_ensemble", type=int, default=30)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=None)
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
    
    # ─── Model Loading (41-in, 2-out for multi) ───
    model = FlowMatchingModel(in_channels=41, out_channels=2).to(device)
    
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        # Use latest checkpoint instead of best
        ckpt_path = os.path.join(args.output_dir, "latest_flow_ckpt.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    
    print("\n" + "="*80)
    print(f"🚀 MODEL CHECKPOINT LOADED:")
    print(f"   ► {os.path.abspath(ckpt_path)}")
    print("="*80 + "\n")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    flow_matcher = CustomFlowMatcher(device=device)
    
    # ─── Load All EOF Bases (identical to v4) ───
    ml_dir = os.path.dirname(__file__)
    data_dir = config["data_dir"]
    
    mjo_eof_path = os.path.join(ml_dir, "mjo_eof_bases.pt")
    mjo_data = torch.load(mjo_eof_path, map_location='cpu', weights_only=False)
    mjo_bases = mjo_data['eof_bases']
    print(f"  ✅ MJO EOFs loaded: {len(mjo_bases)} categories")
    
    nao_bases = None
    nao_lookup = None
    nao_eof_path = os.path.join(ml_dir, "nao_eof_bases.pt")
    nao_idx_path = os.path.join(data_dir, "norm.daily.nao.index.b500101.current.ascii")
    if os.path.exists(nao_eof_path) and os.path.exists(nao_idx_path):
        nao_data = torch.load(nao_eof_path, map_location='cpu', weights_only=False)
        nao_bases = nao_data['eof_bases']
        nao_lookup = parse_nao_index(nao_idx_path)
        print(f"  ✅ NAO EOFs loaded: {len(nao_bases)} categories")
    else:
        print(f"  ⚠️ NAO EOFs not found.")
    
    enso_bases = None
    oni_lookup = None
    enso_eof_path = os.path.join(ml_dir, "enso_eof_bases.pt")
    oni_idx_path = os.path.join(data_dir, "oni.ascii.txt")
    if os.path.exists(enso_eof_path) and os.path.exists(oni_idx_path):
        enso_data = torch.load(enso_eof_path, map_location='cpu', weights_only=False)
        enso_bases = enso_data['eof_bases']
        oni_lookup = parse_oni_index(oni_idx_path)
        print(f"  ✅ ENSO EOFs loaded: {len(enso_bases)} categories")
    else:
        print(f"  ⚠️ ENSO EOFs not found.")
    
    mjo_df = None
    mjo_csv_path = os.path.join(data_dir, "mjo_processed.csv")
    if os.path.exists(mjo_csv_path):
        import pandas as pd
        mjo_df = pd.read_csv(mjo_csv_path, parse_dates=['S'])
        mjo_df['date_str'] = mjo_df['S'].dt.strftime('%Y-%m-%d')
        mjo_df = mjo_df.set_index('date_str')
        print(f"  ✅ MJO RMM CSV loaded: {len(mjo_df)} entries")
    
    # ─── Noise Functions ───
    # Strategy: run v4's 1-channel EOF pipeline independently for each channel,
    # then concatenate to get [vB*E, 2, H, W].
    
    def noise_pure(vB, E, H, W, b, d):
        return torch.randn((vB*E, 2, H, W), device=d)
    
    def _get_mjo_eof_1ch(vB, E, H, W, b, d):
        """v4's _get_mjo_eof — returns [vB*E, 1, H, W]."""
        mjo = b.get('mjo_phase', torch.zeros(vB, dtype=torch.long))
        if isinstance(mjo, torch.Tensor): mjo = mjo.clone().detach()
        else: mjo = torch.tensor(mjo)
        lead = b['lead_idx'].clone().detach() if isinstance(b['lead_idx'], torch.Tensor) else torch.tensor(b['lead_idx'])
        # flow_matcher.eof_sample returns [vB*E, 2, H, W] now, take only ch 0
        eof_2ch = flow_matcher.eof_sample(mjo_bases, mjo, vB*E, H, W, lead_ids=lead)
        return eof_2ch[:, 0:1, :, :]  # [vB*E, 1, H, W]
    
    def _get_nao_eof_1ch(vB, E, H, W, b, d):
        """v4's _get_nao_eof — returns [vB*E, 1, H, W]."""
        noise = torch.zeros((vB*E, 1, H, W), device=d)
        months = b['month']
        leads = b['lead_idx']
        for i in range(vB * E):
            b_idx = i // E
            month = int(months[b_idx])
            lead = int(leads[b_idx])
            init_date = datetime.date(args.year, month, 15)
            nao_phase = get_nao_phase(init_date, nao_lookup)
            noise[i, 0] = sample_from_eof_basis(nao_bases, nao_phase, lead, d, H, W)
        return noise
    
    def _get_enso_eof_1ch(vB, E, H, W, b, d):
        """v4's _get_enso_eof — returns [vB*E, 1, H, W]."""
        noise = torch.zeros((vB*E, 1, H, W), device=d)
        months = b['month']
        leads = b['lead_idx']
        for i in range(vB * E):
            b_idx = i // E
            month = int(months[b_idx])
            lead = int(leads[b_idx])
            enso_state = get_enso_state(month, args.year, oni_lookup)
            noise[i, 0] = sample_from_eof_basis(enso_bases, enso_state, lead, d, H, W)
        return noise
    
    def noise_multimodal_dynamic(vB, E, H, W, b, d):
        """
        v4's noise_multimodal_dynamic, run independently for PR and T2M.
        Returns [vB*E, 2, H, W].
        """
        import pandas as pd
        
        # Run the ENTIRE v4 blend independently for each channel
        channels = []
        for _ch in range(2):
            mjo_noise = _get_mjo_eof_1ch(vB, E, H, W, b, d)
            pure_noise = torch.randn((vB*E, 1, H, W), device=d)
            
            nao_noise = _get_nao_eof_1ch(vB, E, H, W, b, d) if (nao_bases is not None and nao_lookup is not None) else torch.randn((vB*E, 1, H, W), device=d)
            enso_noise = _get_enso_eof_1ch(vB, E, H, W, b, d) if (enso_bases is not None and oni_lookup is not None) else torch.randn((vB*E, 1, H, W), device=d)
            
            month = int(b['month'][0])
            init_date = datetime.date(args.year, month, 15)
            
            # MJO amplitude
            mjo_amp = 1.0
            date_str = init_date.strftime('%Y-%m-%d')
            if mjo_df is not None and date_str in mjo_df.index:
                row = mjo_df.loc[date_str]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                rmm1 = row.get('RMM1_lagged', 0.0)
                rmm2 = row.get('RMM2_lagged', 0.0)
                if pd.isna(rmm1) or pd.isna(rmm2):
                    mjo_amp = 1.0
                else:
                    mjo_amp = float(np.sqrt(rmm1**2 + rmm2**2))
            
            # NAO amplitude
            nao_amp = abs(get_nao_value(init_date, nao_lookup)) if nao_lookup is not None else 0.5
            
            # ENSO amplitude
            enso_amp = abs(get_enso_value(month, args.year, oni_lookup)) if oni_lookup is not None else 0.5
            
            mjo_amp = max(min(mjo_amp, 3.0), 0.1)
            nao_amp = max(min(nao_amp, 2.5), 0.1)
            enso_amp = max(min(enso_amp, 2.5), 0.1)
            
            total = mjo_amp + nao_amp + enso_amp
            w_mjo, w_nao, w_enso = mjo_amp / total, nao_amp / total, enso_amp / total
            
            blend = 0.90 * (w_mjo * mjo_noise + w_nao * nao_noise + w_enso * enso_noise) + 0.10 * pure_noise
            std = blend.std(dim=(2, 3), keepdim=True)
            blend = blend / (std + 1e-6)
            channels.append(blend)
        
        return torch.cat(channels, dim=1)  # [vB*E, 2, H, W]
    
    def noise_multimodal_dynamic_lhs(vB, E, H, W, b, d):
        """
        LHS version: run noise_utils for each channel independently, concat.
        """
        import noise_utils
        ch0 = noise_utils.generate_dynamic_multimodal_noise(b, E, d, mjo_bases, nao_bases, nao_lookup, enso_bases, oni_lookup, mjo_df, flow_matcher, args.year, use_lhs=True)
        ch1 = noise_utils.generate_dynamic_multimodal_noise(b, E, d, mjo_bases, nao_bases, nao_lookup, enso_bases, oni_lookup, mjo_df, flow_matcher, args.year, use_lhs=True)
        return torch.cat([ch0, ch1], dim=1)  # [vB*E, 2, H, W]
    
    # ─── Build Strategy List ───
    # Format: (Name, noise_fn, use_var_head, perturb_cond)
    strategies = [
        ("1. Pure Random",      noise_pure,                  False, False),
        ("2. EOF Cent(LHS)",    noise_multimodal_dynamic_lhs, True,  False),
    ]
    
    n_ml = len(strategies)
    
    # ─── Run Comparison ───
    BLUE = '\033[94m'
    ORANGE = '\033[38;5;214m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    strat_names = ["0. GEOS"] + [s[0] for s in strategies]
    
    print(f"\n{'─'*180}")
    # Two-row header: PR and T2M
    header_parts = [f"  {'Sample':<8} {'Mon':>4} |"]
    for nm in strat_names:
        header_parts.append(f" {nm + ' PR':>17} {nm + ' T2M':>17}")
    print("".join(header_parts))
    print(f"{'─'*180}")
    
    results = {"0. GEOS Baseline": []}
    for name, _, _, _ in strategies:
        results[name] = []
    
    for b_idx, batch in enumerate(test_loader):
        if b_idx >= 12:
            break
        
        month = batch['month'][0].item()
        vB = batch['y_target'].shape[0]
        num_inits = vB // 4
        H, W = batch['y_target'].shape[-2:]
        
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
        
        geos_out = {'pr_crps': geos_pr_crps, 'pr_rmse': geos_pr_rmse, 't2m_crps': geos_t2m_crps, 't2m_rmse': geos_t2m_rmse}
        
        # Diagnostic (first batch only)
        if b_idx == 0:
            print(f"\n🔍 [Diagnostic - Month {month}]")
            print(f"   Target PR   : Min={true_target_raw[:, 0].min():.2f}, Max={true_target_raw[:, 0].max():.2f}, Mean={true_target_raw[:, 0].mean():.2f}")
            print(f"   Target T2M  : Min={true_target_raw[:, 1].min():.2f}, Max={true_target_raw[:, 1].max():.2f}, Mean={true_target_raw[:, 1].mean():.2f}")
            print(f"   GEOS PR     : Min={geos_ens_sample[:, :, 0].min():.2f}, Max={geos_ens_sample[:, :, 0].max():.2f}, Mean={geos_ens_sample[:, :, 0].mean():.2f}")
            print(f"   GEOS T2M    : Min={geos_ens_sample[:, :, 1].min():.2f}, Max={geos_ens_sample[:, :, 1].max():.2f}, Mean={geos_ens_sample[:, :, 1].mean():.2f}")
            print("   " + "─"*50 + "\n")
        
        results["0. GEOS Baseline"].append(geos_out)
        
        print(f"  [Batch {b_idx}/11] Starting inference for {n_ml} ML methods ({args.num_ensemble} mem × {args.num_steps} steps)...", flush=True)
        
        from tqdm import tqdm
        for name, fn, use_var, perturb_cond in tqdm(strategies, desc=f"Batch {b_idx} (Month {month})", leave=False, ncols=100):
            res = run_strategy(model, flow_matcher, batch, device, args.num_ensemble, args.num_steps, fn, use_var, perturb_cond)
            results[name].append(res)
            torch.cuda.empty_cache()
        
        # Print table row
        sys.stdout.write('\033[F\033[K')
        
        def fmt_row(label, all_vals):
            """all_vals: list of dicts with pr_crps, pr_rmse, t2m_crps, t2m_rmse"""
            pr_crps_list = [v['pr_crps'] for v in all_vals]
            pr_rmse_list = [v['pr_rmse'] for v in all_vals]
            t2m_crps_list = [v['t2m_crps'] for v in all_vals]
            t2m_rmse_list = [v['t2m_rmse'] for v in all_vals]
            
            best_pr_c = int(np.argmin(pr_crps_list))
            best_pr_r = int(np.argmin(pr_rmse_list))
            best_t2m_c = int(np.argmin(t2m_crps_list))
            best_t2m_r = int(np.argmin(t2m_rmse_list))
            
            parts = []
            for j, v in enumerate(all_vals):
                # PR
                s_pc = f"{v['pr_crps']:>7.4f}"
                s_pr = f"({v['pr_rmse']:>7.4f})"
                if j == best_pr_c: s_pc = f"{BLUE}{BOLD}{s_pc}{RESET}"
                if j == best_pr_r: s_pr = f"{ORANGE}{BOLD}{s_pr}{RESET}"
                # T2M
                s_tc = f"{v['t2m_crps']:>7.4f}"
                s_tr = f"({v['t2m_rmse']:>7.4f})"
                if j == best_t2m_c: s_tc = f"{BLUE}{BOLD}{s_tc}{RESET}"
                if j == best_t2m_r: s_tr = f"{ORANGE}{BOLD}{s_tr}{RESET}"
                parts.append(f"{s_pc} {s_pr} {s_tc} {s_tr}")
            print(f"  {label:<13} | {' | '.join(parts)}", flush=True)
        
        all_vals = [geos_out] + [results[name][-1] for name, _, _, _ in strategies]
        fmt_row(f"Batch {b_idx:<2} {month:>4}", all_vals)
        
        # Running average
        n_done = b_idx + 1
        all_names = ["0. GEOS Baseline"] + [n for n, _, _, _ in strategies]
        run_avg = []
        for nm in all_names:
            run_avg.append({
                'pr_crps': np.mean([r['pr_crps'] for r in results[nm]]),
                'pr_rmse': np.mean([r['pr_rmse'] for r in results[nm]]),
                't2m_crps': np.mean([r['t2m_crps'] for r in results[nm]]),
                't2m_rmse': np.mean([r['t2m_rmse'] for r in results[nm]]),
            })
        fmt_row(f"  RunAvg({n_done})", run_avg)
        print(f"  {'─'*180}")
    
    # Final CSV
    import pandas as pd
    csv_rows = []
    all_names = ["0. GEOS Baseline"] + [n for n, _, _, _ in strategies]
    for b_idx in range(len(results["0. GEOS Baseline"])):
        row = {'batch': b_idx}
        for nm in all_names:
            r = results[nm][b_idx]
            row[f'{nm} PR_CRPS'] = r['pr_crps']
            row[f'{nm} PR_RMSE'] = r['pr_rmse']
            row[f'{nm} T2M_CRPS'] = r['t2m_crps']
            row[f'{nm} T2M_RMSE'] = r['t2m_rmse']
        csv_rows.append(row)
    
    df = pd.DataFrame(csv_rows)
    mean_row = {col: np.mean(df[col]) for col in df.columns if col != 'batch'}
    mean_row['batch'] = 'MEAN'
    df.loc[len(df)] = mean_row
    csv_path = os.path.join(args.output_dir, f"noise_comparison_v4_multi_results_{args.year}.csv")
    df.to_csv(csv_path, float_format='%.4f', index=False)
    print(f"\n💾 Final CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
