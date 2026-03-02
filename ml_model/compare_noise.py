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
    
    crps = compute_crps(ensemble_4L, true_target_precip, area_weights)
    
    # Calculate empirical variance of the ensemble for plotting
    ens_var = ensemble_4L.var(dim=0)  # [num_inits, 4, H, W]
    
    return crps, ensemble_4L, ens_var, true_target_precip

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
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--num_ensemble", type=int, default=20)
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
    def noise_tight(vB, E, H, W, b, d): return torch.randn((vB*E, 1, H, W), device=d) * 0.3
    
    def noise_eof(vB, E, H, W, b, d):
        mjo = torch.tensor(b.get('mjo_phase', torch.zeros(vB, dtype=torch.long)))
        lead = torch.tensor(b['lead_idx'])
        return flow_matcher.eof_sample(eof_bases, mjo, vB*E, H, W, lead_ids=lead)
        
    def noise_eof_alpha(vB, E, H, W, b, d):
        raw_eof = noise_eof(vB, E, H, W, b, d) # [vB*E, 1, H, W]
        # Scale: Week 1 (lead 0) = 0.7x, Week 4 (lead 3) = 1.3x
        lead_ids = b['lead_idx'].to(d) # [vB]
        # expanded to [vB, E] -> [vB*E]
        lead_exp = lead_ids.unsqueeze(1).expand(vB, E).reshape(-1)
        scale = 0.7 + (lead_exp.float() * 0.2)
        scale = scale.view(-1, 1, 1, 1)
        return raw_eof * scale

    strategies = [
        ("1. Pure Random", noise_pure, False),
        ("2. Tight Random (0.3)", noise_tight, False),
        ("3. MJO EOF (PhasexLead)", noise_eof, False),
        ("4. Alpha-Scaled EOF", noise_eof_alpha, False),
        ("5. EOF + Variance Head", noise_eof, True)
    ]
    
    print(f"\n{'─'*105}")
    print(f"  {'Sample':<8} {'Mon':>4} | {'1. Pure':>12} {'2. Tight':>12} {'3. EOF':>12} {'4. Alpha':>12} {'5. VarHead':>12}")
    print(f"{'─'*105}")
    
    results = {name: [] for name, _, _ in strategies}
    
    for b_idx, batch in enumerate(test_loader):
        if b_idx >= 12: break
        
        month = batch['month'][0].item()
        plot_data = {}
        
        # Run all strategies
        crps_vals = []
        for name, fn, use_var in strategies:
            crps, ens_4L, ens_var, tgt = run_strategy(model, flow_matcher, batch, device, args.num_ensemble, args.num_steps, fn, use_var)
            results[name].append(crps)
            crps_vals.append(crps)
            
            if b_idx == 0: # Save plot data for first month
                plot_data[name] = (ens_4L, ens_var)
                target_plot = tgt
                
        if b_idx == 0:
            plot_path = os.path.join(args.output_dir, f"noise_comparison_month_{month}.png")
            save_strategy_plot(target_plot, plot_data, plot_path)

        print(f"  Batch {b_idx:<2} {month:>4} | {crps_vals[0]:>12.4f} {crps_vals[1]:>12.4f} {crps_vals[2]:>12.4f} {crps_vals[3]:>12.4f} {crps_vals[4]:>12.4f}")

    print(f"{'─'*105}")
    print(f"  {'MEAN':<8} {'':>4} | ", end="")
    for name, _, _ in strategies:
        print(f"{np.mean(results[name]):>12.4f} ", end="")
    print("\n")

if __name__ == "__main__":
    main()
