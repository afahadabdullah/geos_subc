import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import pandas as pd
from accelerate import Accelerator

from dataset_flow import S2SHybridDataset
from flow_matching import FlowMatchingModel, CustomFlowMatcher
import noise_utils

def compute_crps_map(ensemble_preds, target):
    """
    Computes spatial CRPS map for a small ensemble.
    ensemble_preds: [E, B, C, H, W]
    target: [B, C, H, W]
    """
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
    return crps_map.squeeze() # [H, W] if B=1 and C=1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/scratch/11353/afahad/geossub/geos_subc/dataprocess")
    parser.add_argument("--out_dir", type=str, default="/scratch/11353/afahad/geossub/geos_subc/dataprocess/noise")
    parser.add_argument("--checkpoint", type=str, default="ml_output_flow4/BEST_model.pt")
    parser.add_argument("--start_year", type=int, default=2010)
    parser.add_argument("--end_year", type=int, default=2020)
    parser.add_argument("--batch_size", type=int, default=1) # Processes 1 (Init Date x Lead) at a time
    parser.add_argument("--num_ensemble", type=int, default=15)
    parser.add_argument("--num_steps", type=int, default=10)
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load Dataset ──
    dataset = S2SHybridDataset(
        data_root=args.data_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        normalize=True,
        preload=False,
        stats_file="v5_global_stats.pt",
        subsample_monthly=False # Get all available weeks constraint
    )
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ── Load Model ──
    model = FlowMatchingModel(in_channels=36, out_channels=1)
    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state['model_state_dict'])
        print(f"Loaded {args.checkpoint}")
    else:
        print(f"ERROR: No checkpoint found at {args.checkpoint}")
        return
        
    model.to(device)
    model.eval()
    flow_matcher = CustomFlowMatcher(device=device)
    
    # ── Load EOFs ──
    mjo_bases_path = os.path.join(args.data_dir, "mjo_eof_bases.pt")
    nao_bases_path = os.path.join(os.path.dirname(args.checkpoint), "nao_eof_bases.pt")
    enso_bases_path = os.path.join(os.path.dirname(args.checkpoint), "enso_eof_bases.pt")
    mjo_csv_path = os.path.join(args.data_dir, "mjo_rmm_monthly.csv")
    
    mjo_bases = torch.load(mjo_bases_path, map_location="cpu") if os.path.exists(mjo_bases_path) else None
    nao_bases = torch.load(nao_bases_path, map_location="cpu") if os.path.exists(nao_bases_path) else None
    enso_bases = torch.load(enso_bases_path, map_location="cpu") if os.path.exists(enso_bases_path) else None
    
    try:
        mjo_df = pd.read_csv(mjo_csv_path, parse_dates=['Date'])
        mjo_df.set_index('Date', inplace=True)
    except:
        mjo_df = None
        
    try:
        oni_lookup = noise_utils.load_oni_index(args.data_dir)
        nao_lookup = noise_utils.load_nao_index(args.data_dir)
    except:
        oni_lookup = None
        nao_lookup = None

    print(f"Generating noise targets for {args.start_year}-{args.end_year}")
    print(f"Saving to {args.out_dir}")

    for idx, batch in enumerate(tqdm(loader, desc="Generating Maps")):
        vB = batch['y_target'].shape[0] if 'y_target' in batch else batch['input_forecast'].shape[0]
        if vB != 1:
            print(f"Batch {idx} vB={vB}. Skipping (expecting vB=1 for exact saving).")
            continue
            
        _, _, H, W = batch['y_target'].shape
        true_target_precip = batch['target_raw'].to(device) # [1, 1, H, W]
        
        fx_obs = batch['x_obs'].to(device)
        fx_geos = batch['x_geos'].to(device).view(vB, -1, H, W)
        
        f_month = batch['month'].to(device).float()
        fsin_month = torch.sin(2 * np.pi * (f_month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
        fcos_month = torch.cos(2 * np.pi * (f_month - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, H, W)
        
        fl_idx = batch['lead_idx'].to(device).float()
        f_lead_val = (fl_idx / 1.5) - 1.0
        f_lead_channel = f_lead_val.view(vB, 1, 1, 1).expand(vB, 1, H, W)
        
        fx_cond = torch.cat([fx_obs, fx_geos, fsin_month, fcos_month, f_lead_channel], dim=1)
        fx_cond_expanded = fx_cond.unsqueeze(1).expand(vB, args.num_ensemble, -1, H, W).reshape(vB * args.num_ensemble, -1, H, W)
        lead_idx_expanded = batch['lead_idx'].to(device).unsqueeze(1).expand(vB, args.num_ensemble).reshape(-1).long()

        crps_collection = []
        
        # Helper to run ODE and get CRPS
        def run_ode_for_noise(noise_tensor):
            with torch.no_grad():
                p_x1_expanded = flow_matcher.euler_solve(
                    model, noise_tensor, fx_cond_expanded.clone(),
                    num_steps=args.num_steps, lead_idx=lead_idx_expanded, apply_flow_variance=True
                )
            p_x1_batch = p_x1_expanded.view(args.num_ensemble, vB, 1, H, W)
            target_sqrt_min, target_sqrt_max = 0.0, 7.071
            week_sqrt = ((p_x1_batch + 1.0) / 2.0) * (target_sqrt_max - target_sqrt_min) + target_sqrt_min
            week_precip = torch.clamp(week_sqrt ** 2, min=0.0)
            return compute_crps_map(week_precip, true_target_precip)
        
        # Base dimensions
        E = args.num_ensemble
        
        # 0. Pure Random
        noise_0 = torch.randn(E, 1, H, W, device=device)
        crps_0 = run_ode_for_noise(noise_0)
        crps_collection.append(crps_0)
        
        # 1. MJO LHS
        if mjo_bases:
            noise_1 = noise_utils.generate_dynamic_multimodal_noise(batch, E, device, mjo_bases, None, None, None, None, None, flow_matcher, args.start_year, use_lhs=True)
            crps_collection.append(run_ode_for_noise(noise_1))
        else:
            crps_collection.append(torch.ones_like(crps_0) * 999.0)
            
        # 2. NAO LHS
        if nao_bases:
            noise_2 = noise_utils.generate_dynamic_multimodal_noise(batch, E, device, None, nao_bases, nao_lookup, None, None, None, flow_matcher, args.start_year, use_lhs=True)
            crps_collection.append(run_ode_for_noise(noise_2))
        else:
            crps_collection.append(torch.ones_like(crps_0) * 999.0)
            
        # 3. ENSO LHS
        if enso_bases:
            noise_3 = noise_utils.generate_dynamic_multimodal_noise(batch, E, device, None, None, None, enso_bases, oni_lookup, None, flow_matcher, args.start_year, use_lhs=True)
            crps_collection.append(run_ode_for_noise(noise_3))
        else:
            crps_collection.append(torch.ones_like(crps_0) * 999.0)
            
        # Stack CRPS [4, H, W]
        crps_tensor = torch.stack(crps_collection, dim=0)
        winner_map = torch.argmin(crps_tensor, dim=0).to(torch.uint8) # [H, W]

        # Get metadata for filename
        month = int(batch['month'][0].item())
        lead = int(batch['lead_idx'][0].item())
        
        # The true initial date is slightly obfuscated in the dataset since we yield `lead_idx`
        # But for uniqueness, we can just use the index if we don't have the date.
        # However, dataset_hybrid usually holds `batch['year']` if we extract it, but it wasn't returned explicitly
        # We will save by flat index padding.
        
        out_file = os.path.join(args.out_dir, f"target_{idx:05d}_m{month}_l{lead+1}.pt")
        
        # Extract MJO from batch if available
        rmm1, rmm2 = 0.0, 0.0
        if 'mjo' in batch and len(batch['mjo'].shape) >= 2:
            rmm1 = batch['mjo'][0, 0].item()
            rmm2 = batch['mjo'][0, 1].item()
            
        import datetime
        fake_date = datetime.date(args.start_year, month, 15)
        nao_val = noise_utils.get_nao_value(fake_date, nao_lookup) if nao_lookup else 0.0
        enso_val = noise_utils.get_enso_value(month, args.start_year, oni_lookup) if oni_lookup else 0.0

        save_data = {
            'winner_map': winner_map.cpu(),
            'month': month,
            'lead': lead,
            'fsin_month': fsin_month[0, 0, 0, 0].cpu().item(),
            'fcos_month': fcos_month[0, 0, 0, 0].cpu().item(),
            'mjo_rmm1': rmm1,
            'mjo_rmm2': rmm2,
            'nao_val': nao_val,
            'enso_val': enso_val
        }
        
        torch.save(save_data, out_file)
        
        # Calculate winner percentages for logging
        unique, counts = torch.unique(winner_map, return_counts=True)
        stats_dict = {int(k): int(v) for k, v in zip(unique, counts)}
        total = sum(stats_dict.values())
        p_rand = (stats_dict.get(0, 0) / total) * 100
        p_mjo = (stats_dict.get(1, 0) / total) * 100
        p_nao = (stats_dict.get(2, 0) / total) * 100
        p_enso = (stats_dict.get(3, 0) / total) * 100
        
        if idx % 5 == 0:
            print(f"[{idx}] Saved {args.start_year}-{args.end_year} | M: {month:02d} | L: {lead+1} | "
                  f"Winners: Rand({p_rand:.0f}%) MJO({p_mjo:.0f}%) NAO({p_nao:.0f}%) ENSO({p_enso:.0f}%)", flush=True)

if __name__ == "__main__":
    main()
