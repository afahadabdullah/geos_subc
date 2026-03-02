#!/usr/bin/env python3
"""
Noise Strategy Comparison & Visualization
=========================================
Evaluates CRPS under 5 different ensemble noise strategies:
  1. Pure Random:        noise ~ N(0, 1)
  2. Tightened Random:   noise ~ N(0, 0.3)
  3. MJO EOF:            phase×lead conditional EOFs
  4. Alpha-Scaled EOF:   MJO EOFs scaled by (0.7 + lead_week * 0.2)
  5. Model Var Head:     predicted variance scalar

Also saves a visual comparison plot of the 5 strategies for the first sample.
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

@torch.no_grad()
def run_strategy(model, flow_matcher, batch, device, num_ensemble, num_steps, noise_fn, use_var_head=False):
    """Generic inference runner for a given noise strategy."""
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
    
    # Generate noise
    noise_expanded = noise_fn(vB, num_ensemble, H, W, batch, device)
    
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
    
    # Calculate Lead-specific CRPS (L is dim 2 in both [E, inits, L, H, W] and [inits, L, H, W])
    # However, compute_crps takes full arrays. Let's compute them by slicing.
    crps_leads = []
    for l in range(4):
        # [E, inits, 1, H, W]
        ens_l = ensemble_4L[:, :, l:l+1, :, :]
        # [inits, 1, H, W]
        tgt_l = true_target_precip[:, l:l+1, :, :]
        c_l = compute_crps(ens_l, tgt_l, area_weights)
        crps_leads.append(c_l)
        
    crps_out = [crps_all] + crps_leads # [all, w1, w2, w3, w4]
    
    # Calculate empirical variance of the ensemble for plotting
    ens_var = ensemble_4L.var(dim=0)  # [num_inits, 4, H, W]
    
    return crps_out, ensemble_4L, ens_var, true_target_precip

def save_strategy_plot(target, results_dict, output_path):
    """
    Plots the target GPCP alongside the Mean, single member, and Variance for each strategy.
    results_dict format: { 'Strategy Name': (ensemble_4L, ens_var) }
    """
    strategies = list(results_dict.keys())
    n_strats = len(strategies)
    
    # We will plot only Lead 0 (Week 1) and Lead 3 (Week 4) for the first init date
    leads_to_plot = [0, 3] 
    
    fig, axes = plt.subplots(n_strats + 1, 4, figsize=(24, 4 * (n_strats + 1)))
    
    # Row 0: Target GPCP
    t_img = target[0].cpu().numpy() # [4, H, W]
    for col, l in enumerate(leads_to_plot):
        t_min, t_max = t_img[l].min(), t_img[l].max()
        
        # Col 0/1: Target Week 1/4
        im = axes[0, col*2].imshow(t_img[l], cmap='Blues', vmin=t_min, vmax=t_max)
        fig.colorbar(im, ax=axes[0, col*2], fraction=0.046, pad=0.04)
        axes[0, col*2].set_title(f"TARGET GPCP (Week {l+1})")
        axes[0, col*2+1].axis('off') # Leave blank space next to target
        
    axes[0, 0].set_ylabel("GROUND TRUTH", fontsize=14, fontweight='bold')
    
    # Rows 1-5: Strategies
    for row, (name, (ens_4L, ens_var)) in enumerate(results_dict.items(), start=1):
        # ens_4L: [E, inits, 4, H, W] -> take init 0
        ens_img = ens_4L[:, 0].cpu().numpy() # [E, 4, H, W]
        var_img = ens_var[0].cpu().numpy()   # [4, H, W]
        mean_img = ens_img.mean(axis=0)      # [4, H, W]
        
        axes[row, 0].set_ylabel(name, fontsize=12, fontweight='bold')
        
        # We handle "0. GEOS Baseline" safely which might have a different number of ensemble members (4 vs 10).
        # We already handle that dimension safely above array limits.
        
        for col, l in enumerate(leads_to_plot):
            t_min, t_max = t_img[l].min(), t_img[l].max()
            
            # Plot 1: Ensemble Mean
            im1 = axes[row, col*2].imshow(mean_img[l], cmap='Blues', vmin=t_min, vmax=t_max)
            fig.colorbar(im1, ax=axes[row, col*2], fraction=0.046, pad=0.04)
            axes[row, col*2].set_title(f"Ens Mean (Week {l+1})")
            
            # Plot 2: Ensemble Variance
            v_max = np.percentile(var_img[l], 95) + 1e-3
            im2 = axes[row, col*2+1].imshow(var_img[l], cmap='YlGn', vmin=0, vmax=v_max)
            fig.colorbar(im2, ax=axes[row, col*2+1], fraction=0.046, pad=0.04)
            axes[row, col*2+1].set_title(f"Ens Spread/Var (Week {l+1})")
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  📸 Saved visual comparison plot: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare Noise Strategies")
    parser.add_argument("--output_dir", type=str, default="ml_output_flow4")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--num_ensemble", type=int, default=30)
    parser.add_argument("--num_steps", type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(os.path.join(os.path.dirname(__file__), "config_flow.yaml")) as f:
        config = yaml.safe_load(f)
        
    test_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=args.year, end_year=args.year,  # STRICTLY only test year
        normalize=True, preload=True,  # FORCE preload to RAM
        stats_file="v5_global_stats.pt", subsample_monthly=True
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    model = FlowMatchingModel(in_channels=36, out_channels=1).to(device)
    
    best_ckpt = os.path.join(args.output_dir, "best_model_epoch_115_crps_1.3602.pt")
    print(f"Loaded: {os.path.basename(best_ckpt)}")
    
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    flow_matcher = CustomFlowMatcher(device=device)
    
    eof_bases_path = os.path.join(os.path.dirname(__file__), "mjo_eof_bases.pt")
    eof_data = torch.load(eof_bases_path, map_location='cpu', weights_only=False)
    eof_bases = eof_data['eof_bases']
    
    # Noise Functions
    def noise_pure(vB, E, H, W, b, d): return torch.randn((vB*E, 1, H, W), device=d)
    
    def _get_raw_eof(vB, E, H, W, b, d):
        """Get raw EOF structured noise (unit variance per field)."""
        mjo = b.get('mjo_phase', torch.zeros(vB, dtype=torch.long))
        if isinstance(mjo, torch.Tensor):
            mjo = mjo.clone().detach()
        else:
            mjo = torch.tensor(mjo)
        lead = b['lead_idx'].clone().detach() if isinstance(b['lead_idx'], torch.Tensor) else torch.tensor(b['lead_idx'])
        return flow_matcher.eof_sample(eof_bases, mjo, vB*E, H, W, lead_ids=lead)
    
    def noise_eof_60(vB, E, H, W, b, d):
        pure = torch.randn((vB*E, 1, H, W), device=d)
        eof = _get_raw_eof(vB, E, H, W, b, d)
        blend = 0.4 * pure + 0.6 * eof
        std = blend.std(dim=(2, 3), keepdim=True)
        blend = blend / (std + 1e-6)
        return blend
        
    def noise_eof_75(vB, E, H, W, b, d):
        pure = torch.randn((vB*E, 1, H, W), device=d)
        eof = _get_raw_eof(vB, E, H, W, b, d)
        blend = 0.25 * pure + 0.75 * eof
        std = blend.std(dim=(2, 3), keepdim=True)
        blend = blend / (std + 1e-6)
        return blend

    def noise_eof_90(vB, E, H, W, b, d):
        pure = torch.randn((vB*E, 1, H, W), device=d)
        eof = _get_raw_eof(vB, E, H, W, b, d)
        blend = 0.1 * pure + 0.9 * eof
        std = blend.std(dim=(2, 3), keepdim=True)
        blend = blend / (std + 1e-6)
        return blend

    def noise_eof_98(vB, E, H, W, b, d):
        pure = torch.randn((vB*E, 1, H, W), device=d)
        eof = _get_raw_eof(vB, E, H, W, b, d)
        blend = 0.02 * pure + 0.98 * eof
        std = blend.std(dim=(2, 3), keepdim=True)
        blend = blend / (std + 1e-6)
        return blend

    strategies = [
        ("1. Pure Random", noise_pure, False),
        ("2. VarH(60%)", noise_eof_60, True),
        ("3. VarH(75%)", noise_eof_75, True),
        ("4. VarH(90%)", noise_eof_90, True),
        ("5. VarH(98%)", noise_eof_98, True)
    ]
    
    print(f"\n{'─'*140}")
    print(f"  {'Sample':<8} {'Mon':>4} | {'0. GEOS':>11} {'1. Pure':>19} {'2. VarH(60)':>19} {'3. VarH(75)':>19} {'4. VarH(90)':>19} {'5. VarH(98)':>19}")
    print(f"{'─'*140}")
    
    # Store results as lists of lists: [all, w1, w2, w3, w4]
    results = {"0. GEOS Baseline": []}
    for name, _, _ in strategies:
        results[name] = []
    
    for b_idx, batch in enumerate(test_loader):
        if b_idx >= 12: break
        
        month = batch['month'][0].item()
        plot_data = {}
        
        # 0. Calculate GEOS CRPS directly
        geos_crps_out = [0.0]*5
        vB = batch['y_target'].shape[0]
        num_inits = vB // 4
        true_tgt = batch['target_raw_full'][0::4].to(device)
        H, W = true_tgt.shape[-2:]
        
        if 'geos_ens_raw' in batch:
            geos_ens_sample = batch['geos_ens_raw'][0::4].to(device) # [num_inits, M=4, 4, H, W]
            lats = np.linspace(-90, 90, H)
            cos_weights = np.maximum(np.cos(np.deg2rad(lats)), 0)
            area_weights = torch.from_numpy(cos_weights).float().to(device)
            area_weights = area_weights / area_weights.sum()
            area_weights = area_weights.view(1, 1, H, 1)
            
            # compute_crps expects [E, B, C, H, W] -> transpose to [M, num_inits, 4, H, W]
            geos_ens_t = geos_ens_sample.transpose(0, 1)
            geos_crps_all = compute_crps(geos_ens_t, true_tgt, area_weights)
            
            geos_crps_leads = []
            for l in range(4):
                c_l = compute_crps(geos_ens_t[:, :, l:l+1, :, :], true_tgt[:, l:l+1, :, :], area_weights)
                geos_crps_leads.append(c_l)
            
            geos_crps_out = [geos_crps_all] + geos_crps_leads
                
            if b_idx == 0:
                plot_data["0. GEOS Baseline"] = (geos_ens_t, geos_ens_t.var(dim=0))
                
        results["0. GEOS Baseline"].append(geos_crps_out)
        crps_vals = [geos_crps_out[0]] # Print total CRPS to Terminal
        
        # Diagnostic print before heavy ODE solves
        print(f"  [Batch {b_idx}/11] Starting inference for {len(strategies)} ML methods (30 mem × 10 steps)...", flush=True)
        
        # Run all ML strategies with a progress bar for this batch
        from tqdm import tqdm
        for name, fn, use_var in tqdm(strategies, desc=f"Batch {b_idx} (Month {month})", leave=False, ncols=100):
            crps_out, ens_4L, ens_var, tgt = run_strategy(model, flow_matcher, batch, device, args.num_ensemble, args.num_steps, fn, use_var)
            results[name].append(crps_out)
            crps_vals.append(crps_out[0]) # print total
            torch.cuda.empty_cache()  # Free GPU memory between strategies
            
            if b_idx == 0: # Save plot data for first month
                plot_data[name] = (ens_4L, ens_var)
                target_plot = tgt
                
        if b_idx == 0:
            plot_path = os.path.join(args.output_dir, f"noise_comparison_month_{month}.png")
            save_strategy_plot(target_plot, plot_data, plot_path)

        # Print the structured table row, overwriting the diagnostic line if on interactive terminal
        sys.stdout.write('\033[F\033[K') # move up 1 line and clear it
        
        # Helper: format a row with the best (lowest) value highlighted in blue
        BLUE = '\033[94m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
        def fmt_row(label, vals):
            best_idx = int(np.argmin(vals))
            parts = []
            geos_v = vals[0]
            for j, v in enumerate(vals):
                s = f"{v:>11.4f}"
                if j > 0 and geos_v > 0:
                    pct = ((geos_v - v) / geos_v) * 100.0
                    s += f" ({pct:>+5.1f}%)"
                elif j > 0:
                    s += " " * 9
                if j == best_idx:
                    s = f"{BLUE}{BOLD}{s}{RESET}"
                parts.append(s)
            print(f"  {label:<13} | {' '.join(parts)}", flush=True)
        
        fmt_row(f"Batch {b_idx:<2} {month:>4}", crps_vals)
        
        # Print per-lead breakdown underneath
        all_crps_out = [geos_crps_out] + [results[name][-1] for name, _, _ in strategies]
        for w in range(4):
            lead_vals = [c[w+1] for c in all_crps_out]
            fmt_row(f"    W{w+1}", lead_vals)
        
        # Running average across all completed batches (total + per-lead)
        n_done = b_idx + 1
        all_names = ["0. GEOS Baseline"] + [n for n, _, _ in strategies]
        run_avg_total = [np.mean([x[0] for x in results[nm]]) for nm in all_names]
        fmt_row(f"  RunAvg({n_done})", run_avg_total)
        for w in range(4):
            run_avg_w = [np.mean([x[w+1] for x in results[nm]]) for nm in all_names]
            fmt_row(f"  AvgW{w+1}({n_done})", run_avg_w)
        print(f"  {'─'*140}")
        
        # Incremental CSV save after each batch
        import pandas as pd
        flat_results = {}
        lead_suffixes = [" (Total)", " (W1)", " (W2)", " (W3)", " (W4)"]
        for strat_name, batch_lists in results.items():
            for i, suffix in enumerate(lead_suffixes):
                col_name = f"{strat_name}{suffix}"
                flat_results[col_name] = [batch[i] for batch in batch_lists]
        df = pd.DataFrame(flat_results)
        mean_row = {col: np.mean(vals) for col, vals in flat_results.items()}
        df.loc['MEAN'] = mean_row
        csv_path = os.path.join(args.output_dir, f"noise_comparison_results_{args.year}.csv")
        df.to_csv(csv_path, float_format='%.4f')

    print(f"{'─'*140}")
    print(f"  {'MEAN':<8} {'':>4} | ", end="")
    
    # Calculate means (index 0 is total CRPS)
    geos_mean_total = np.mean([x[0] for x in results['0. GEOS Baseline']])
    print(f"{geos_mean_total:>11.4f} ", end="")
    for name, _, _ in strategies:
        strat_mean_total = np.mean([x[0] for x in results[name]])
        pct = ((geos_mean_total - strat_mean_total) / geos_mean_total) * 100.0
        print(f"{strat_mean_total:>11.4f} ({pct:>+5.1f}%) ", end="")
    print("\n")
    print(f"  💾 Final CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
