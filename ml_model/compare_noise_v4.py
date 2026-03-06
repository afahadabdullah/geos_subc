#!/usr/bin/env python3
"""
Multi-Modal Noise Strategy Comparison & Visualization (v4)
============================================================
Evaluates CRPS under 6 different ensemble noise strategies:
  0. GEOS Baseline:          Raw GEOS S2S ensemble (4 members)
  1. Pure Random:            noise ~ N(0, 1)
  2. MJO EOF (98%):          98% MJO EOF + 2% isotropic
  3. NAO EOF (98%):          98% NAO EOF + 2% isotropic
  4. ENSO EOF (98%):         98% ENSO EOF + 2% isotropic
  5. Dynamic Multi-Modal:    Amplitude-weighted blend of MJO + NAO + ENSO EOFs
  
Also saves a visual comparison plot for the first sample.

Usage:
  python compare_noise_v4.py --output_dir ml_output_flow4 --year 2022
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import glob
import yaml
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from flow_matching import FlowMatchingModel, CustomFlowMatcher
from dataset_flow import S2SHybridDataset
from train_flow import compute_crps


# ─── Index Parsers (same as in compute_*_eofs.py) ───

def parse_nao_index(nao_path):
    """Parse CPC daily NAO index file."""
    nao_lookup = {}
    with open(nao_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            val = float(parts[3])
            import datetime
            d = datetime.date(year, month, day)
            nao_lookup[d] = val
        except ValueError:
            continue
    return nao_lookup


def parse_oni_index(oni_path):
    """Parse CPC ONI file."""
    oni_lookup = {}
    with open(oni_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            oni_lookup[(int(parts[1]), parts[0].strip())] = float(parts[3])
        except (ValueError, IndexError):
            continue
    return oni_lookup


def get_nao_phase(init_date, nao_lookup, threshold=0.5):
    """Get NAO phase using a 7-day trailing average ending 1 day BEFORE init_date."""
    import datetime
    import pandas as pd
    if isinstance(init_date, pd.Timestamp):
        base_date = init_date.date()
    else:
        base_date = init_date
        
    vals = []
    for lag in range(1, 8):
        d = base_date - datetime.timedelta(days=lag)
        if d in nao_lookup:
            vals.append(nao_lookup[d])
            
    if not vals:
        return 1
        
    val = sum(vals) / len(vals)
    if val < -threshold: return 0
    elif val > threshold: return 2
    return 1


def get_nao_value(init_date, nao_lookup):
    """Get raw NAO value (7-day trailing average) for amplitude calculation."""
    import datetime
    import pandas as pd
    if isinstance(init_date, pd.Timestamp):
        base_date = init_date.date()
    else:
        base_date = init_date
        
    vals = []
    for lag in range(1, 8):
        d = base_date - datetime.timedelta(days=lag)
        if d in nao_lookup:
            vals.append(nao_lookup[d])
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def get_enso_state(month, year, oni_lookup, threshold=0.5):
    """Get ENSO state using previous season's ONI (lag to avoid leakage)."""
    month_to_season = {
        1: ('OND', -1), 2: ('NDJ', 0), 3: ('DJF', 0), 4: ('JFM', 0),
        5: ('FMA', 0), 6: ('MAM', 0), 7: ('AMJ', 0), 8: ('MJJ', 0),
        9: ('JJA', 0), 10: ('JAS', 0), 11: ('ASO', 0), 12: ('SON', 0),
    }
    seas, yr_off = month_to_season[month]
    lookup_year = year + yr_off
    if month == 1:
        lookup_year = year - 1
    val = oni_lookup.get((lookup_year, seas), 0.0)
    if val < -threshold: return 0
    elif val > threshold: return 2
    return 1


def get_enso_value(month, year, oni_lookup):
    """Get raw ONI value for amplitude calculation."""
    month_to_season = {
        1: ('OND', -1), 2: ('NDJ', 0), 3: ('DJF', 0), 4: ('JFM', 0),
        5: ('FMA', 0), 6: ('MAM', 0), 7: ('AMJ', 0), 8: ('MJJ', 0),
        9: ('JJA', 0), 10: ('JAS', 0), 11: ('ASO', 0), 12: ('SON', 0),
    }
    seas, yr_off = month_to_season[month]
    lookup_year = year + yr_off
    if month == 1:
        lookup_year = year - 1
    return oni_lookup.get((lookup_year, seas), 0.0)


# ─── EOF Sampling Helpers ───

def sample_from_eof_basis(eof_bases, phase, lead, device, H, W):
    """Sample a single noise field from an EOF basis dict."""
    key = (phase, lead)
    if key not in eof_bases:
        key = phase
    if key not in eof_bases:
        key = (1, lead)  # Neutral fallback
    if key not in eof_bases:
        return torch.randn(H, W, device=device)
    
    eofs = eof_bases[key]['eofs'].to(device)  # [K, H, W]
    K = eofs.shape[0]
    alpha = torch.randn(K, device=device)
    noise_field = torch.einsum('k,khw->hw', alpha, eofs)
    
    std = noise_field.std()
    if std > 1e-6:
        noise_field = noise_field / std
    return noise_field


# ─── Core Inference Runner ───

@torch.no_grad()
def run_strategy(model, flow_matcher, batch, device, num_ensemble, num_steps, noise_fn, use_var_head=False, perturb_cond=False):
    model.eval()
    
    vB = batch['y_target'].shape[0] if 'y_target' in batch else batch['input_forecast'].shape[0]
    _, _, H, W = batch['y_target'].shape
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
    
    # Clone is necessary because expand creates a non-writable view
    fx_cond_expanded = fx_cond.unsqueeze(1).expand(vB, num_ensemble, -1, H, W).reshape(vB * num_ensemble, -1, H, W).clone()
    lead_idx_expanded = batch['lead_idx'].to(device).unsqueeze(1).expand(vB, num_ensemble).reshape(-1).long()
    
    # Generate noise
    noise_expanded = noise_fn(vB, num_ensemble, H, W, batch, device)
    
    if perturb_cond:
        # Add structured physical perturbation to the GEOS Precipitation conditioning channel
        fx_cond_expanded[:, 1:2, :, :] += (noise_expanded * 0.10)
        
    # Solve ODE
    p_x1_expanded = flow_matcher.euler_solve(
        model, noise_expanded, fx_cond_expanded,
        num_steps=num_steps, lead_idx=lead_idx_expanded, apply_flow_variance=use_var_head
    )
    
    # Denormalize
    p_x1_batch = p_x1_expanded.view(vB, num_ensemble, H, W)
    target_sqrt_min, target_sqrt_max = 0.0, 7.071
    week_sqrt = ((p_x1_batch + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
    week_precip = torch.clamp(week_sqrt ** 2, min=0.0)
    
    ensemble_preds_precip = week_precip.transpose(0, 1)  # [E, vB, H, W]
    ensemble_4L = ensemble_preds_precip.view(num_ensemble, num_inits, 4, H, W)
    
    # Area weights
    lats = np.linspace(-90, 90, H)
    cos_weights = np.maximum(np.cos(np.deg2rad(lats)), 0)
    area_weights = torch.from_numpy(cos_weights).float().to(device)
    area_weights = area_weights / area_weights.sum()
    area_weights = area_weights.view(1, 1, H, 1)
    
    crps_all = compute_crps(ensemble_4L, true_target_precip, area_weights)
    ens_mean = ensemble_4L.mean(dim=0)
    rmse_all = torch.sqrt(torch.mean((ens_mean - true_target_precip)**2 * area_weights) * area_weights.numel()).item()
    
    crps_leads = []
    for l in range(4):
        ens_l = ensemble_4L[:, :, l:l+1, :, :]
        tgt_l = true_target_precip[:, l:l+1, :, :]
        c_l = compute_crps(ens_l, tgt_l, area_weights)
        
        ens_mean_l = ens_mean[:, l:l+1, :, :]
        r_l = torch.sqrt(torch.mean((ens_mean_l - tgt_l)**2 * area_weights) * area_weights.numel()).item()
        crps_leads.append((c_l, r_l))
        
    crps_out = [(crps_all, rmse_all)] + crps_leads
    ens_var = ensemble_4L.var(dim=0)
    
    return crps_out, ensemble_4L, ens_var, true_target_precip


def save_strategy_plot(target, results_dict, output_path):
    """Plots target GPCP alongside Mean and Variance for each strategy."""
    strategies = list(results_dict.keys())
    n_strats = len(strategies)
    leads_to_plot = [0, 3]
    
    fig, axes = plt.subplots(n_strats + 1, 4, figsize=(24, 4 * (n_strats + 1)))
    
    t_img = target[0].cpu().numpy()
    for col, l in enumerate(leads_to_plot):
        t_min, t_max = t_img[l].min(), t_img[l].max()
        im = axes[0, col*2].imshow(t_img[l], cmap='Blues', vmin=t_min, vmax=t_max)
        fig.colorbar(im, ax=axes[0, col*2], fraction=0.046, pad=0.04)
        axes[0, col*2].set_title(f"TARGET GPCP (Week {l+1})")
        axes[0, col*2+1].axis('off')
    axes[0, 0].set_ylabel("GROUND TRUTH", fontsize=14, fontweight='bold')
    
    for row, (name, (ens_4L, ens_var)) in enumerate(results_dict.items(), start=1):
        ens_img = ens_4L[:, 0].cpu().numpy()
        var_img = ens_var[0].cpu().numpy()
        mean_img = ens_img.mean(axis=0)
        
        axes[row, 0].set_ylabel(name, fontsize=10, fontweight='bold')
        
        for col, l in enumerate(leads_to_plot):
            t_min, t_max = t_img[l].min(), t_img[l].max()
            im1 = axes[row, col*2].imshow(mean_img[l], cmap='Blues', vmin=t_min, vmax=t_max)
            fig.colorbar(im1, ax=axes[row, col*2], fraction=0.046, pad=0.04)
            axes[row, col*2].set_title(f"Ens Mean (Week {l+1})")
            
            v_max = np.percentile(var_img[l], 95) + 1e-3
            im2 = axes[row, col*2+1].imshow(var_img[l], cmap='YlGn', vmin=0, vmax=v_max)
            fig.colorbar(im2, ax=axes[row, col*2+1], fraction=0.046, pad=0.04)
            axes[row, col*2+1].set_title(f"Ens Spread/Var (Week {l+1})")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  📸 Saved visual comparison plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Noise Strategy Comparison (v4)")
    parser.add_argument("--output_dir", type=str, default="ml_output_flow4")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--num_ensemble", type=int, default=30)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint. If None, auto-detects best_model_*.pt")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(os.path.join(os.path.dirname(__file__), "config_flow.yaml")) as f:
        config = yaml.safe_load(f)
    
    test_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=args.year, end_year=args.year,
        normalize=True, preload=True,
        stats_file="v5_global_stats.pt", subsample_monthly=True
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    model = FlowMatchingModel(in_channels=36, out_channels=1).to(device)
    
    # Hardcoded to load BEST_model.pt
    if args.checkpoint:
        best_ckpt = args.checkpoint
    else:
        best_ckpt = os.path.join(args.output_dir, "BEST_model.pt")
        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"Missing required checkpoint: {best_ckpt}")
    
    print("\n" + "="*80)
    print(f"🚀 MODEL CHECKPOINT LOADED:")
    print(f"   ► {os.path.abspath(best_ckpt)}")
    print("="*80 + "\n")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    flow_matcher = CustomFlowMatcher(device=device)
    
    # ─── Load All EOF Bases ───
    ml_dir = os.path.dirname(__file__)
    data_dir = config["data_dir"]
    
    # MJO EOFs (required)
    mjo_eof_path = os.path.join(ml_dir, "mjo_eof_bases.pt")
    mjo_data = torch.load(mjo_eof_path, map_location='cpu', weights_only=False)
    mjo_bases = mjo_data['eof_bases']
    print(f"  ✅ MJO EOFs loaded: {len(mjo_bases)} categories")
    
    # NAO EOFs (optional)
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
        print(f"  ⚠️ NAO EOFs not found. Run compute_nao_eofs.py first. Skipping NAO strategies.")
    
    # ENSO EOFs (optional)
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
        print(f"  ⚠️ ENSO EOFs not found. Run compute_enso_eofs.py first. Skipping ENSO strategies.")
    
    # MJO RMM CSV (for amplitude in dynamic weighting)
    mjo_df = None
    mjo_csv_path = os.path.join(data_dir, "mjo_processed.csv")
    if os.path.exists(mjo_csv_path):
        import pandas as pd
        mjo_df = pd.read_csv(mjo_csv_path, parse_dates=['S'])
        mjo_df['date_str'] = mjo_df['S'].dt.strftime('%Y-%m-%d')
        mjo_df = mjo_df.set_index('date_str')
        print(f"  ✅ MJO RMM CSV loaded: {len(mjo_df)} entries (for dynamic amplitude)")
    else:
        print(f"  ⚠️ MJO CSV not found at {mjo_csv_path}. Dynamic weighting will use default MJO amplitude.")
    
    # ─── Noise Functions ───
    
    def noise_pure(vB, E, H, W, b, d):
        return torch.randn((vB*E, 1, H, W), device=d)
    
    def _get_mjo_eof(vB, E, H, W, b, d):
        mjo = b.get('mjo_phase', torch.zeros(vB, dtype=torch.long))
        if isinstance(mjo, torch.Tensor): mjo = mjo.clone().detach()
        else: mjo = torch.tensor(mjo)
        lead = b['lead_idx'].clone().detach() if isinstance(b['lead_idx'], torch.Tensor) else torch.tensor(b['lead_idx'])
        return flow_matcher.eof_sample(mjo_bases, mjo, vB*E, H, W, lead_ids=lead)
    
    def noise_mjo_98(vB, E, H, W, b, d):
        pure = torch.randn((vB*E, 1, H, W), device=d)
        eof = _get_mjo_eof(vB, E, H, W, b, d)
        blend = 0.02 * pure + 0.98 * eof
        std = blend.std(dim=(2, 3), keepdim=True)
        return blend / (std + 1e-6)
    
    def _get_nao_eof(vB, E, H, W, b, d):
        """Generate NAO-conditioned EOF noise for the entire expanded batch."""
        import datetime
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
    
    def _get_enso_eof(vB, E, H, W, b, d):
        """Generate ENSO-conditioned EOF noise for the entire expanded batch."""
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
    
    def noise_nao_98(vB, E, H, W, b, d):
        pure = torch.randn((vB*E, 1, H, W), device=d)
        eof = _get_nao_eof(vB, E, H, W, b, d)
        blend = 0.02 * pure + 0.98 * eof
        std = blend.std(dim=(2, 3), keepdim=True)
        return blend / (std + 1e-6)
    
    def noise_enso_98(vB, E, H, W, b, d):
        pure = torch.randn((vB*E, 1, H, W), device=d)
        eof = _get_enso_eof(vB, E, H, W, b, d)
        blend = 0.02 * pure + 0.98 * eof
        std = blend.std(dim=(2, 3), keepdim=True)
        return blend / (std + 1e-6)
    
    def noise_multimodal_dynamic(vB, E, H, W, b, d):
        """
        Dynamically weighted multi-modal blended noise.
        
        Weights are computed from the REAL-TIME amplitude of each climate mode,
        instead of fixed seasonal values. This means:
          - If MJO is raging (amp=2.5) but NAO is dead (amp=0.1), MJO dominates.
          - If it's a strong La Niña winter with weak MJO, ENSO dominates.
        
        90% of the blend is amplitude-weighted EOF noise, 10% is isotropic.
        """
        import datetime
        import pandas as pd
        
        mjo_noise = _get_mjo_eof(vB, E, H, W, b, d)
        pure_noise = torch.randn((vB*E, 1, H, W), device=d)
        
        if nao_bases is not None and nao_lookup is not None:
            nao_noise = _get_nao_eof(vB, E, H, W, b, d)
        else:
            nao_noise = torch.randn((vB*E, 1, H, W), device=d)
        
        if enso_bases is not None and oni_lookup is not None:
            enso_noise = _get_enso_eof(vB, E, H, W, b, d)
        else:
            enso_noise = torch.randn((vB*E, 1, H, W), device=d)
        
        month = int(b['month'][0])
        init_date = datetime.date(args.year, month, 15)
        
        # ── Compute MJO amplitude ──
        date_str = init_date.strftime('%Y-%m-%d')
        if mjo_df is not None and date_str in mjo_df.index:
            row = mjo_df.loc[date_str]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            rmm1 = row.get('RMM1_lagged', 0.0)
            rmm2 = row.get('RMM2_lagged', 0.0)
            if pd.isna(rmm1) or pd.isna(rmm2):
                mjo_amp = 1.0
            else:
                mjo_amp = float(np.sqrt(rmm1**2 + rmm2**2))
        else:
            mjo_amp = 1.0  # Default moderate amplitude
        
        # ── Compute NAO amplitude ──
        if nao_lookup is not None:
            nao_amp = abs(get_nao_value(init_date, nao_lookup))
        else:
            nao_amp = 0.5
        
        # ── Compute ENSO amplitude ──
        if oni_lookup is not None:
            enso_amp = abs(get_enso_value(month, args.year, oni_lookup))
        else:
            enso_amp = 0.5
        
        # Cap amplitudes to prevent one extreme outlier from wiping out the others
        mjo_amp = min(mjo_amp, 3.0)
        nao_amp = min(nao_amp, 2.5)
        enso_amp = min(enso_amp, 2.5)
        
        # Ensure minimum floor so no mode is ever completely zeroed out
        mjo_amp = max(mjo_amp, 0.1)
        nao_amp = max(nao_amp, 0.1)
        enso_amp = max(enso_amp, 0.1)
        
        # Normalize to get fractional weights
        total = mjo_amp + nao_amp + enso_amp
        w_mjo = mjo_amp / total
        w_nao = nao_amp / total
        w_enso = enso_amp / total
        
        # 90% physically weighted + 10% isotropic safety net
        blend = 0.90 * (w_mjo * mjo_noise + w_nao * nao_noise + w_enso * enso_noise) + 0.10 * pure_noise
        
        std = blend.std(dim=(2, 3), keepdim=True)
        return blend / (std + 1e-6)
        
    def noise_multimodal_dynamic_lhs(vB, E, H, W, b, d):
        import noise_utils
        return noise_utils.generate_dynamic_multimodal_noise(b, E, d, mjo_bases, nao_bases, nao_lookup, enso_bases, oni_lookup, mjo_df, flow_matcher, year, use_lhs=True)
    
    # ─── Build Strategy List ───
    # Format: (Name, noise_fn, use_var_head, perturb_cond)
    strategies = [
        ("1. Pure Random",          noise_pure,              False, False),
        ("2. MJO EOF(98%)",         noise_mjo_98,            True,  False),
    ]
    
    if nao_bases is not None:
        strategies.append(("3. NAO EOF(98%)",      noise_nao_98,              True, False))
    if enso_bases is not None:
        strategies.append(("4. ENSO EOF(98%)",     noise_enso_98,             True, False))
    if nao_bases is not None or enso_bases is not None:
        strategies.append(("5. Dynamic MM",        noise_multimodal_dynamic,  True, False))
        strategies.append(("6. EOF Cent(LHS)",     noise_multimodal_dynamic_lhs, True, False))
        strategies.append(("7. DynMM+CondP",       noise_multimodal_dynamic,  True, True))
    
    n_ml = len(strategies)
    
    # ─── Run Comparison ───
    
    # Header
    strat_names = ["0. GEOS"] + [s[0] for s in strategies]
    header = f"  {'Sample':<8} {'Mon':>4} |"
    for nm in strat_names:
        header += f" {nm:>17}"
    print(f"\n{'─'*140}")
    print(header)
    print(f"{'─'*140}")
    
    results = {"0. GEOS Baseline": []}
    for name, _, _, _ in strategies:
        results[name] = []
    
    for b_idx, batch in enumerate(test_loader):
        if b_idx >= 12:
            break
        
        month = batch['month'][0].item()
        plot_data = {}
        
        # 0. GEOS Baseline CRPS
        geos_crps_out = [0.0]*5
        vB = batch['y_target'].shape[0]
        num_inits = vB // 4
        true_tgt = batch['target_raw_full'][0::4].to(device)
        H, W = true_tgt.shape[-2:]
        
        if 'geos_ens_raw' in batch:
            geos_ens_sample = batch['geos_ens_raw'][0::4].to(device)
            lats = np.linspace(-90, 90, H)
            cos_weights = np.maximum(np.cos(np.deg2rad(lats)), 0)
            area_weights = torch.from_numpy(cos_weights).float().to(device)
            area_weights = area_weights / area_weights.sum()
            area_weights = area_weights.view(1, 1, H, 1)
            
            geos_ens_t = geos_ens_sample.transpose(0, 1)
            geos_crps_all = compute_crps(geos_ens_t, true_tgt, area_weights)
            geos_mean = geos_ens_t.mean(dim=0)
            geos_rmse_all = torch.sqrt(torch.mean((geos_mean - true_tgt)**2 * area_weights) * area_weights.numel()).item()
            
            geos_crps_leads = []
            for l in range(4):
                c_l = compute_crps(geos_ens_t[:, :, l:l+1, :, :], true_tgt[:, l:l+1, :, :], area_weights)
                geos_mean_l = geos_mean[:, l:l+1, :, :]
                r_l = torch.sqrt(torch.mean((geos_mean_l - true_tgt[:, l:l+1, :, :])**2 * area_weights) * area_weights.numel()).item()
                geos_crps_leads.append((c_l, r_l))
            geos_crps_out = [(geos_crps_all, geos_rmse_all)] + geos_crps_leads
            
            if b_idx == 0:
                plot_data["0. GEOS Baseline"] = (geos_ens_t, geos_ens_t.var(dim=0))
        
        results["0. GEOS Baseline"].append(geos_crps_out)
        crps_vals = [geos_crps_out[0]]
        
        print(f"  [Batch {b_idx}/11] Starting inference for {n_ml} ML methods ({args.num_ensemble} mem × {args.num_steps} steps)...", flush=True)
        
        from tqdm import tqdm
        for name, fn, use_var, perturb_cond in tqdm(strategies, desc=f"Batch {b_idx} (Month {month})", leave=False, ncols=100):
            crps_out, ens_4L, ens_var, tgt = run_strategy(model, flow_matcher, batch, device, args.num_ensemble, args.num_steps, fn, use_var, perturb_cond)
            results[name].append(crps_out)
            crps_vals.append(crps_out[0])
            torch.cuda.empty_cache()
            
            if b_idx == 0:
                plot_data[name] = (ens_4L, ens_var)
                target_plot = tgt
        
        if b_idx == 0:
            plot_path = os.path.join(args.output_dir, f"noise_comparison_v4_month_{month}.png")
            save_strategy_plot(target_plot, plot_data, plot_path)
        
        # Print table row
        sys.stdout.write('\033[F\033[K')
        
        BLUE = '\033[94m'
        ORANGE = '\033[38;5;214m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
        
        def fmt_row(label, vals):
            # vals contains (CRPS, RMSE)
            crps_vals_list = [c for c, r in vals]
            rmse_vals_list = [r for c, r in vals]
            best_c_idx = int(np.argmin(crps_vals_list))
            best_r_idx = int(np.argmin(rmse_vals_list))
            
            parts = []
            for j, (c, r) in enumerate(vals):
                s_c = f"{c:>7.4f}"
                s_r = f"({r:>7.4f})"
                if j == best_c_idx:
                    s_c = f"{BLUE}{BOLD}{s_c}{RESET}"
                if j == best_r_idx:
                    s_r = f"{ORANGE}{BOLD}{s_r}{RESET}"
                parts.append(f"{s_c} {s_r}")
            print(f"  {label:<13} | {' '.join(parts)}", flush=True)
        
        fmt_row(f"Batch {b_idx:<2} {month:>4}", crps_vals)
        
        # Per-lead breakdown
        all_crps_out = [geos_crps_out] + [results[name][-1] for name, _, _, _ in strategies]
        for w in range(4):
            lead_vals = [c[w+1] for c in all_crps_out]
            fmt_row(f"    W{w+1}", lead_vals)
        
        # Running average
        n_done = b_idx + 1
        all_names = ["0. GEOS Baseline"] + [n for n, _, _, _ in strategies]
        run_avg_total = [(np.mean([x[0][0] for x in results[nm]]), np.mean([x[0][1] for x in results[nm]])) for nm in all_names]
        fmt_row(f"  RunAvg({n_done})", run_avg_total)
        for w in range(4):
            run_avg_w = [(np.mean([x[w+1][0] for x in results[nm]]), np.mean([x[w+1][1] for x in results[nm]])) for nm in all_names]
            fmt_row(f"  AvgW{w+1}({n_done})", run_avg_w)
        print(f"  {'─'*140}")
        
        # Incremental CSV (extracting only CRPS to preserve legacy plotting tool compatibility, or add CRPS vs RMSE)
        import pandas as pd
        flat_results = {}
        lead_suffixes = [" (Total)", " (W1)", " (W2)", " (W3)", " (W4)"]
        for strat_name, batch_lists in results.items():
            for i, suffix in enumerate(lead_suffixes):
                col_name = f"{strat_name}{suffix}"
                flat_results[col_name] = [bl[i][0] for bl in batch_lists] # Save CRPS only
                flat_results[f"{strat_name}{suffix} RMSE"] = [bl[i][1] for bl in batch_lists] # Save RMSE
        df = pd.DataFrame(flat_results)
        mean_row = {col: np.mean(vals) for col, vals in flat_results.items()}
        df.loc['MEAN'] = mean_row
        csv_path = os.path.join(args.output_dir, f"noise_comparison_v4_results_{args.year}.csv")
        df.to_csv(csv_path, float_format='%.4f')
    
    # Final Summary
    print(f"{'─'*140}")
    print(f"  {'MEAN (CRPS) (RMSE)':<20} | ", end="")
    
    geos_mean_c = np.mean([x[0][0] for x in results['0. GEOS Baseline']])
    geos_mean_r = np.mean([x[0][1] for x in results['0. GEOS Baseline']])
    print(f"{geos_mean_c:>7.4f} ({geos_mean_r:>7.4f}) ", end="")
    
    for name, _, _, _ in strategies:
        strat_mean_c = np.mean([x[0][0] for x in results[name]])
        strat_mean_r = np.mean([x[0][1] for x in results[name]])
        print(f"{strat_mean_c:>7.4f} ({strat_mean_r:>7.4f}) ", end="")
    print("\n")
    print(f"  💾 Final CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
