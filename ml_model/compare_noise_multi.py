#!/usr/bin/env python3
"""
Multi-Variate Noise Strategy Comparison (multi)
====================================================
Direct adaptation of compare_noise_v4.py for the 2-channel (PR + T2M) model.
Uses the SAME noise_utils.py (1-channel) by running each EOF pipeline
independently for PR and T2M, then concatenating.

Usage:
  python compare_noise_multi.py --output_dir ml_output_flowmulti --year 2021
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
from dataset_flow_multi import S2SHybridDataset
from train_flow_multiv1 import compute_crps, compute_rmse
import noise_utils_multi

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


# ─── Core Inference Runner (2-channel adaptation of v4) ───

@torch.no_grad()
def run_strategy(model, flow_matcher, batch, device, num_ensemble, num_steps, noise_fn, use_var_head=False, perturb_cond=False):
    model.eval()
    
    vB = batch['y_target'].shape[0]
    H, W = batch['y_target'].shape[-2:]
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
    
    # ─── DIAGNOSTIC: Print noise statistics for first batch ───
    print(f"\n    📊 [Noise Diag] Shape: {list(noise_expanded.shape)}")
    for c in range(noise_expanded.shape[1]):
        ch = noise_expanded[:, c]
        print(f"       Ch{c}: Mean={ch.mean():.4f}, Std={ch.std():.4f}, Min={ch.min():.4f}, Max={ch.max():.4f}")
    
    if perturb_cond:
        fx_cond_expanded[:, 4:6, :, :] += (noise_expanded[:, 0:1, :, :] * 0.10)
    
    # Solve ODE -> output [vB*E, 2, H, W]
    p_x1_expanded = flow_matcher.euler_solve(
        model, noise_expanded, fx_cond_expanded,
        num_steps=num_steps, lead_idx=lead_idx_expanded, apply_flow_variance=use_var_head
    )
    
    # ─── DIAGNOSTIC: Print ODE output statistics ───
    print(f"    📊 [ODE Output] Shape: {list(p_x1_expanded.shape)}")
    for c in range(p_x1_expanded.shape[1]):
        ch = p_x1_expanded[:, c]
        print(f"       Ch{c}: Mean={ch.mean():.4f}, Std={ch.std():.4f}, Min={ch.min():.4f}, Max={ch.max():.4f}")
    
    # Separate channels
    p_x1_batch = p_x1_expanded.view(vB, num_ensemble, 2, H, W)
    
    # Denormalize PR (channel 0)
    target_sqrt_min, target_sqrt_max = 0.0, 7.071
    p_x1_pr = torch.clamp(p_x1_batch[:, :, 0], min=-1.0, max=1.0)
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
    
    # Total PR/T2M metrics
    pr_crps = compute_crps(ensemble_pr, true_target_pr, area_weights)
    pr_rmse = compute_rmse(ensemble_pr.mean(dim=0), true_target_pr, area_weights)
    t2m_crps = compute_crps(ensemble_t2m, true_target_t2m, area_weights)
    t2m_rmse = compute_rmse(ensemble_t2m.mean(dim=0), true_target_t2m, area_weights)
    
    # Per-lead-week metrics (W1-W4)
    pr_leads = []
    t2m_leads = []
    for l in range(4):
        ens_pr_l = ensemble_pr[:, :, l:l+1, :, :]
        tgt_pr_l = true_target_pr[:, l:l+1, :, :]
        c_pr = compute_crps(ens_pr_l, tgt_pr_l, area_weights)
        r_pr = compute_rmse(ens_pr_l.mean(dim=0), tgt_pr_l, area_weights)
        pr_leads.append((c_pr, r_pr))
        
        ens_t2m_l = ensemble_t2m[:, :, l:l+1, :, :]
        tgt_t2m_l = true_target_t2m[:, l:l+1, :, :]
        c_t2m = compute_crps(ens_t2m_l, tgt_t2m_l, area_weights)
        r_t2m = compute_rmse(ens_t2m_l.mean(dim=0), tgt_t2m_l, area_weights)
        t2m_leads.append((c_t2m, r_t2m))
    
    # Return list of 5 dicts: [Total, W1, W2, W3, W4]
    out = [{'pr_crps': pr_crps, 'pr_rmse': pr_rmse, 't2m_crps': t2m_crps, 't2m_rmse': t2m_rmse}]
    for l in range(4):
        out.append({
            'pr_crps': pr_leads[l][0], 'pr_rmse': pr_leads[l][1],
            't2m_crps': t2m_leads[l][0], 't2m_rmse': t2m_leads[l][1],
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Multi-Variate Noise Comparison (multi)")
    parser.add_argument("--output_dir", type=str, default="ml_output_flowmulti")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--num_ensemble", type=int, default=15)
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
        # Match compare_noise_v4_multi.py: use the latest flow checkpoint by default,
        # because the EOF-favorable validation state came from that path.
        ckpt_path = os.path.join(args.output_dir, "best_flow_ckpt.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.output_dir, "BEST_model.pt")
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
    eof_dir = os.path.join(data_dir, "eof")

    def resolve_eof_path(filename):
        for base_dir in (eof_dir, data_dir, ml_dir):
            candidate = os.path.join(base_dir, filename)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(eof_dir, filename)
    
    mjo_eof_path = resolve_eof_path("mjo_eof_bases.pt")
    mjo_data = torch.load(mjo_eof_path, map_location='cpu', weights_only=False)
    mjo_bases = mjo_data['eof_bases']
    print(f"  ✅ MJO EOFs loaded: {len(mjo_bases)} categories from {mjo_eof_path}")
    
    nao_bases = None
    nao_lookup = None
    nao_eof_path = resolve_eof_path("nao_eof_bases.pt")
    nao_idx_path = os.path.join(data_dir, "norm.daily.nao.index.b500101.current.ascii")
    if os.path.exists(nao_eof_path) and os.path.exists(nao_idx_path):
        nao_data = torch.load(nao_eof_path, map_location='cpu', weights_only=False)
        nao_bases = nao_data['eof_bases']
        nao_lookup = parse_nao_index(nao_idx_path)
        print(f"  ✅ NAO EOFs loaded: {len(nao_bases)} categories from {nao_eof_path}")
    else:
        print(f"  ⚠️ NAO EOFs not found.")
    
    enso_bases = None
    oni_lookup = None
    enso_eof_path = resolve_eof_path("enso_eof_bases.pt")
    oni_idx_path = os.path.join(data_dir, "oni.ascii.txt")
    if os.path.exists(enso_eof_path) and os.path.exists(oni_idx_path):
        enso_data = torch.load(enso_eof_path, map_location='cpu', weights_only=False)
        enso_bases = enso_data['eof_bases']
        oni_lookup = parse_oni_index(oni_idx_path)
        print(f"  ✅ ENSO EOFs loaded: {len(enso_bases)} categories from {enso_eof_path}")
    else:
        print(f"  ⚠️ ENSO EOFs not found.")
        
    # --- Load T2M EOF Bases ---
    mjo_t2m_eof_path = resolve_eof_path("mjo_t2m_eof_bases.pt")
    t2m_mjo_bases = torch.load(mjo_t2m_eof_path, map_location='cpu', weights_only=False)['eof_bases'] if os.path.exists(mjo_t2m_eof_path) else None

    nao_t2m_eof_path = resolve_eof_path("nao_t2m_eof_bases.pt")
    t2m_nao_bases = torch.load(nao_t2m_eof_path, map_location='cpu', weights_only=False)['eof_bases'] if os.path.exists(nao_t2m_eof_path) else None

    enso_t2m_eof_path = resolve_eof_path("enso_t2m_eof_bases.pt")
    t2m_enso_bases = torch.load(enso_t2m_eof_path, map_location='cpu', weights_only=False)['eof_bases'] if os.path.exists(enso_t2m_eof_path) else None
    
    if t2m_mjo_bases is not None:
        print(f"  ✅ T2M MJO/NAO/ENSO EOFs loaded from {mjo_t2m_eof_path}, {nao_t2m_eof_path}, {enso_t2m_eof_path}.")
    
    mjo_df = None
    mjo_csv_path = os.path.join(data_dir, "mjo_processed.csv")
    if os.path.exists(mjo_csv_path):
        import pandas as pd
        mjo_df = pd.read_csv(mjo_csv_path, parse_dates=['S'])
        mjo_df['date_str'] = mjo_df['S'].dt.strftime('%Y-%m-%d')
        mjo_df = mjo_df.set_index('date_str')
        print(f"  ✅ MJO RMM CSV loaded: {len(mjo_df)} entries")
    
    # ─── Noise Functions ───
    # The sampler now resolves the exact init date from the batch metadata when
    # year/month/day are present, and only falls back to args.year if needed.
    
    def noise_pure(vB, E, H, W, b, d):
        return torch.randn((vB*E, 2, H, W), device=d)
    
    def noise_multimodal_dynamic(vB, E, H, W, b, d):
        return noise_utils_multi.generate_dynamic_multimodal_noise_multi(
            b, E, d,
            mjo_bases, nao_bases, enso_bases,
            t2m_mjo_bases, t2m_nao_bases, t2m_enso_bases,
            nao_lookup, oni_lookup, mjo_df, args.year,
            use_lhs=False,
        )
    
    def noise_multimodal_dynamic_lhs(vB, E, H, W, b, d):
        """
        LHS version: run noise_utils for each channel independently, concat.
        PR uses precipitation EOF bases, T2M uses temperature EOF bases.
        """
        return noise_utils_multi.generate_dynamic_multimodal_noise_multi(
            b, E, d,
            mjo_bases, nao_bases, enso_bases,
            t2m_mjo_bases, t2m_nao_bases, t2m_enso_bases,
            nao_lookup, oni_lookup, mjo_df, args.year,
            use_lhs=True,
        )

    def noise_multimodal_dynamic_lhs_val_replay(vB, E, H, W, b, d):
        """
        Match run_val_inference in train_flow_multiv1.py:
        LHS EOF noise for both PR and T2M, but no ensemble orthogonalization.
        """
        return noise_utils_multi.generate_dynamic_multimodal_noise_multi(
            b, E, d,
            mjo_bases, nao_bases, enso_bases,
            t2m_mjo_bases, t2m_nao_bases, t2m_enso_bases,
            nao_lookup, oni_lookup, mjo_df, args.year,
            use_lhs=True,
            orthogonalize_lhs=False,
        )
        
    def noise_multimodal_dynamic_lhs_pr_only(vB, E, H, W, b, d):
        """
        Exact replication of Epoch 187 validation state:
        PR uses EOF LHS noise. T2M uses pure Random Gaussian noise.
        """
        return noise_utils_multi.generate_dynamic_multimodal_noise_multi(
            b, E, d,
            mjo_bases, nao_bases, enso_bases,
            t2m_mjo_bases, t2m_nao_bases, t2m_enso_bases,
            nao_lookup, oni_lookup, mjo_df, args.year,
            use_lhs=True,
            t2m_random_only=True,
        )

    def noise_multimodal_dynamic_lhs_pr_blend(vB, E, H, W, b, d):
        """
        Softer PR ablation:
        use validation-style non-orthogonalized LHS EOF noise for PR, blend in 2% random,
        and keep T2M random so we isolate whether PR EOF tails are the main problem.
        """
        return noise_utils_multi.generate_dynamic_multimodal_noise_multi(
            b, E, d,
            mjo_bases, nao_bases, enso_bases,
            t2m_mjo_bases, t2m_nao_bases, t2m_enso_bases,
            nao_lookup, oni_lookup, mjo_df, args.year,
            use_lhs=True,
            t2m_random_only=True,
            orthogonalize_lhs=False,
            pr_random_blend=0.02,
        )
    
    # ─── Build Strategy List ───
    # Match compare_noise_v4_multi.py so results are comparable to the earlier runs
    # where EOF-based noise beat pure random.
    # Format: (Name, noise_fn, use_var_head, perturb_cond)
    strategies = [
        ("1. Pure Random",        noise_pure,                           False, False),
        ("2. EOF(LHS)+Var",       noise_multimodal_dynamic_lhs,         True,  False),
        ("3. EOF(LHS) noVar",     noise_multimodal_dynamic_lhs,         False, False),
        ("4. EOF PR + Rnd T2M",   noise_multimodal_dynamic_lhs_pr_only, True,  False),
        ("5. ValReplay EOF+Var",  noise_multimodal_dynamic_lhs_val_replay, True, False),
        ("6. PR EOF98 + Rnd T2M", noise_multimodal_dynamic_lhs_pr_blend, False, False),
    ]
    
    n_ml = len(strategies)
    
    # ─── Run Comparison ───
    BLUE = '\033[94m'
    ORANGE = '\033[38;5;214m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    strat_names = ["0. GEOS"] + [s[0] for s in strategies]
    
    print(f"\n{'─'*180}")
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
        
        current_year = int(batch['year'][0].item()) if 'year' in batch else args.year
        month = int(batch['month'][0].item())
        current_day = int(batch['day'][0].item()) if 'day' in batch else 15
        vB = batch['y_target'].shape[0]
        num_inits = vB // 4
        H, W = batch['y_target'].shape[-2:]
        
        # Area weights
        lats = np.linspace(-90, 90, H)
        cos_weights = np.maximum(np.cos(np.deg2rad(lats)), 0)
        area_weights = torch.from_numpy(cos_weights).float().to(device)
        area_weights = area_weights / area_weights.sum()
        area_weights = area_weights.view(1, 1, H, 1)
        
        # 0. GEOS Baseline — compute total + per-lead metrics
        true_target_raw = batch['target_raw_full'][0::4].to(device)
        geos_ens_sample = batch['geos_ens_raw'][0::4].to(device)
        
        geos_pr_ens = geos_ens_sample[:, :, 0].transpose(0, 1)  # [M, N, 4, H, W]
        geos_t2m_ens = geos_ens_sample[:, :, 1].transpose(0, 1)
        tgt_pr = true_target_raw[:, 0]
        tgt_t2m = true_target_raw[:, 1]
        
        geos_out_total = {
            'pr_crps': compute_crps(geos_pr_ens, tgt_pr, area_weights),
            'pr_rmse': compute_rmse(geos_pr_ens.mean(dim=0), tgt_pr, area_weights),
            't2m_crps': compute_crps(geos_t2m_ens, tgt_t2m, area_weights),
            't2m_rmse': compute_rmse(geos_t2m_ens.mean(dim=0), tgt_t2m, area_weights),
        }
        geos_out = [geos_out_total]
        for l in range(4):
            geos_out.append({
                'pr_crps': compute_crps(geos_pr_ens[:, :, l:l+1], tgt_pr[:, l:l+1], area_weights),
                'pr_rmse': compute_rmse(geos_pr_ens.mean(dim=0)[:, l:l+1], tgt_pr[:, l:l+1], area_weights),
                't2m_crps': compute_crps(geos_t2m_ens[:, :, l:l+1], tgt_t2m[:, l:l+1], area_weights),
                't2m_rmse': compute_rmse(geos_t2m_ens.mean(dim=0)[:, l:l+1], tgt_t2m[:, l:l+1], area_weights),
            })
        
        # Diagnostic (first batch only)
        if b_idx == 0:
            print(f"\n🔍 [Diagnostic - Init {current_year:04d}-{month:02d}-{current_day:02d}]")
            print(f"   Target PR   : Min={tgt_pr.min():.2f}, Max={tgt_pr.max():.2f}, Mean={tgt_pr.mean():.2f}")
            print(f"   Target T2M  : Min={tgt_t2m.min():.2f}, Max={tgt_t2m.max():.2f}, Mean={tgt_t2m.mean():.2f}")
            print(f"   GEOS PR     : Min={geos_ens_sample[:, :, 0].min():.2f}, Max={geos_ens_sample[:, :, 0].max():.2f}, Mean={geos_ens_sample[:, :, 0].mean():.2f}")
            print(f"   GEOS T2M    : Min={geos_ens_sample[:, :, 1].min():.2f}, Max={geos_ens_sample[:, :, 1].max():.2f}, Mean={geos_ens_sample[:, :, 1].mean():.2f}")
            print("   " + "─"*50 + "\n")
        
        results["0. GEOS Baseline"].append(geos_out)
        
        print(f"  [Batch {b_idx}/11] Starting inference for {n_ml} ML methods ({args.num_ensemble} mem × {args.num_steps} steps)...", flush=True)
        
        from tqdm import tqdm
        batch_desc = f"Batch {b_idx} ({current_year:04d}-{month:02d}-{current_day:02d})"
        for name, fn, use_var, perturb_cond in tqdm(strategies, desc=batch_desc, leave=False, ncols=100):
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
                s_pc = f"{v['pr_crps']:>7.4f}"
                s_pr = f"({v['pr_rmse']:>7.4f})"
                if j == best_pr_c: s_pc = f"{BLUE}{BOLD}{s_pc}{RESET}"
                if j == best_pr_r: s_pr = f"{ORANGE}{BOLD}{s_pr}{RESET}"
                s_tc = f"{v['t2m_crps']:>7.4f}"
                s_tr = f"({v['t2m_rmse']:>7.4f})"
                if j == best_t2m_c: s_tc = f"{BLUE}{BOLD}{s_tc}{RESET}"
                if j == best_t2m_r: s_tr = f"{ORANGE}{BOLD}{s_tr}{RESET}"
                parts.append(f"{s_pc} {s_pr} {s_tc} {s_tr}")
            print(f"  {label:<13} | {' | '.join(parts)}", flush=True)
        
        # Total row
        all_crps_out = [geos_out] + [results[name][-1] for name, _, _, _ in strategies]
        fmt_row(f"Batch {b_idx:<2} {month:>4}", [c[0] for c in all_crps_out])
        
        # Per-lead breakdown (W1-W4)
        for w in range(4):
            lead_vals = [c[w+1] for c in all_crps_out]
            fmt_row(f"    W{w+1}", lead_vals)
        
        # Running average (total)
        n_done = b_idx + 1
        all_names = ["0. GEOS Baseline"] + [n for n, _, _, _ in strategies]
        run_avg_total = []
        for nm in all_names:
            run_avg_total.append({
                'pr_crps': np.mean([r[0]['pr_crps'] for r in results[nm]]),
                'pr_rmse': np.mean([r[0]['pr_rmse'] for r in results[nm]]),
                't2m_crps': np.mean([r[0]['t2m_crps'] for r in results[nm]]),
                't2m_rmse': np.mean([r[0]['t2m_rmse'] for r in results[nm]]),
            })
        fmt_row(f"  RunAvg({n_done})", run_avg_total)
        
        # Running average per-week
        for w in range(4):
            run_avg_w = []
            for nm in all_names:
                run_avg_w.append({
                    'pr_crps': np.mean([r[w+1]['pr_crps'] for r in results[nm]]),
                    'pr_rmse': np.mean([r[w+1]['pr_rmse'] for r in results[nm]]),
                    't2m_crps': np.mean([r[w+1]['t2m_crps'] for r in results[nm]]),
                    't2m_rmse': np.mean([r[w+1]['t2m_rmse'] for r in results[nm]]),
                })
            fmt_row(f"  AvgW{w+1}({n_done})", run_avg_w)
        print(f"  {'─'*180}")
    
    # Final CSV
    import pandas as pd
    csv_rows = []
    all_names = ["0. GEOS Baseline"] + [n for n, _, _, _ in strategies]
    lead_suffixes = [" (Total)", " (W1)", " (W2)", " (W3)", " (W4)"]
    for b_idx in range(len(results["0. GEOS Baseline"])):
        row = {'batch': b_idx}
        for nm in all_names:
            for i, suffix in enumerate(lead_suffixes):
                r = results[nm][b_idx][i]
                row[f'{nm}{suffix} PR_CRPS'] = r['pr_crps']
                row[f'{nm}{suffix} PR_RMSE'] = r['pr_rmse']
                row[f'{nm}{suffix} T2M_CRPS'] = r['t2m_crps']
                row[f'{nm}{suffix} T2M_RMSE'] = r['t2m_rmse']
        csv_rows.append(row)
    
    df = pd.DataFrame(csv_rows)
    mean_row = {col: np.mean(df[col]) for col in df.columns if col != 'batch'}
    mean_row['batch'] = 'MEAN'
    df.loc[len(df)] = mean_row
    csv_path = os.path.join(args.output_dir, f"noise_comparison_multi_results_{args.year}.csv")
    df.to_csv(csv_path, float_format='%.4f', index=False)
    print(f"\n💾 Final CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
