import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import pandas as pd
import datetime

from dataset_flow import S2SHybridDataset
from flow_matching import CustomFlowMatcher
import noise_utils

def min_max_scale(tensor):
    """Min-Max scaling to [0, 1] for a spatial map to allow structural comparisons."""
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max - t_min < 1e-6:
        return torch.zeros_like(tensor)
    return (tensor - t_min) / (t_max - t_min)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/scratch/11353/afahad/geossub/geos_subc/dataprocess")
    parser.add_argument("--out_dir", type=str, default="/scratch/11353/afahad/geossub/geos_subc/dataprocess/noise")
    parser.add_argument("--checkpoint", type=str, default="ml_output_flow4/BEST_model.pt") # Just for path resolution
    parser.add_argument("--start_year", type=int, default=2010)
    parser.add_argument("--end_year", type=int, default=2020)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_ensemble", type=int, default=30) # High ensemble size to get smooth variance maps
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load Dataset ──
    dataset = S2SHybridDataset(
        data_root=args.data_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        normalize=True,
        preload=False,
        stats_file="v5_global_stats.pt",
    )
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ── Load EOFs & Physics Data ──
    mjo_bases_path = os.path.join(args.data_dir, "mjo_eof_bases.pt")
    nao_bases_path = os.path.join(os.path.dirname(args.checkpoint), "nao_eof_bases.pt")
    enso_bases_path = os.path.join(os.path.dirname(args.checkpoint), "enso_eof_bases.pt")
    mjo_csv_path = os.path.join(args.data_dir, "mjo_processed.csv") # Used processed CSV
    
    mjo_data = torch.load(mjo_bases_path, map_location="cpu") if os.path.exists(mjo_bases_path) else None
    mjo_bases = mjo_data['eof_bases'] if mjo_data else None
    
    nao_data = torch.load(nao_bases_path, map_location="cpu") if os.path.exists(nao_bases_path) else None
    nao_bases = nao_data['eof_bases'] if nao_data else None
    
    enso_data = torch.load(enso_bases_path, map_location="cpu") if os.path.exists(enso_bases_path) else None
    enso_bases = enso_data['eof_bases'] if enso_data else None
    
    try:
        mjo_df = pd.read_csv(mjo_csv_path, parse_dates=['S'])
        mjo_df['date_str'] = mjo_df['S'].dt.strftime('%Y-%m-%d')
        mjo_df = mjo_df.set_index('date_str')
    except:
        mjo_df = None
        
    try:
        oni_lookup = noise_utils.parse_oni_index(os.path.join(args.data_dir, "oni.ascii.txt"))
        nao_lookup = noise_utils.parse_nao_index(os.path.join(args.data_dir, "norm.daily.nao.index.b500101.current.ascii"))
    except:
        oni_lookup = None
        nao_lookup = None

    # Dummy matcher merely for EOF API compatibility
    flow_matcher = CustomFlowMatcher(device=device)

    print(f"Generating physical noise targets for {args.start_year}-{args.end_year}...")
    print(f"Using Physical Absolute Error-Spread Matching (Bypassing UNet).")

    for idx, batch in enumerate(tqdm(loader, desc="Generating Maps")):
        vB = batch['y_target'].shape[0] if 'y_target' in batch else batch['input_forecast'].shape[0]
        if vB != 1:
            print(f"Skipping Batch {idx} (vB != 1)")
            continue
            
        H, W = batch['y_target'].shape[-2:]
        
        # ── 1. Calculate True Physical Error Map (The Ideal Spread) ──
        true_gpcp = batch['target_raw'] # [1, 1, H, W]
        # Calculate raw GEOS ensemble mean for this specific lead time
        # GEOS is [1, M, L, H, W], we need lead index
        lead_idx = int(batch['lead_idx'][0].item())
        # Average over M (dim=0 since we take batch[0]), then select Lead
        geos_mean = batch['geos_ens_raw'][0].mean(dim=0)[lead_idx:lead_idx+1].unsqueeze(0) # [1, 1, H, W]
        
        # Absolute Error Map (Target Uncertainty)
        abs_error_map = torch.abs(true_gpcp - geos_mean).squeeze().to(device) # [H, W]
        
        # We structurally normalize the error map because we are comparing spatial structural similarity,
        # not the raw mm/day units against the N(0,1) noise output.
        target_spread_scaled = min_max_scale(abs_error_map)

        # Base dimensions
        E = args.num_ensemble
        
        # ── 2. Generate Spatially Structured Noise Prototypes ──
        # We want to see the 2D Variance/Spread map that each EOF method inherently injects
        def generate_and_get_spread_map(noise_gen_func, *args_list):
            with torch.no_grad():
                noise_ens = noise_gen_func(*args_list) # [E, 1, H, W]
                # Calculate standard deviation map
                std_map = noise_ens.std(dim=0).squeeze() # [H, W]
                # Scale structurally
                return min_max_scale(std_map)
                
        # 0. Pure Random
        spread_0 = generate_and_get_spread_map(lambda: torch.randn(E, 1, H, W, device=device))
        
        # 1. MJO LHS
        if mjo_bases:
            spread_1 = generate_and_get_spread_map(noise_utils.generate_dynamic_multimodal_noise, batch, E, device, mjo_bases, None, None, None, None, flow_matcher, args.start_year, True)
        else:
            spread_1 = torch.zeros_like(spread_0)
            
        # 2. NAO LHS
        if nao_bases:
            spread_2 = generate_and_get_spread_map(noise_utils.generate_dynamic_multimodal_noise, batch, E, device, None, nao_bases, nao_lookup, None, None, flow_matcher, args.start_year, True)
        else:
            spread_2 = torch.zeros_like(spread_0)
            
        # 3. ENSO LHS
        if enso_bases:
            spread_3 = generate_and_get_spread_map(noise_utils.generate_dynamic_multimodal_noise, batch, E, device, None, None, None, enso_bases, oni_lookup, None, flow_matcher, args.start_year, True)
        else:
            spread_3 = torch.zeros_like(spread_0)
            
        # ── 3. Score Maps via Structural Difference ──
        # We simply want the noise method whose 2D Variance structure most closely mimics the 2D Absolute Error structure
        # Lower score = better structural match
        score_0 = torch.abs(spread_0 - target_spread_scaled)
        score_1 = torch.abs(spread_1 - target_spread_scaled)
        score_2 = torch.abs(spread_2 - target_spread_scaled)
        score_3 = torch.abs(spread_3 - target_spread_scaled)
        
        # Stack Scores [4, H, W]
        score_tensor = torch.stack([score_0, score_1, score_2, score_3], dim=0)
        
        # Winner Map: The method that had the lowest absolute difference with the True Error Spread
        winner_map = torch.argmin(score_tensor, dim=0).to(torch.uint8) # [H, W]

        # ── 4. Save Network Inputs & Winner Map ──
        month = int(batch['month'][0].item())
        lead = int(batch['lead_idx'][0].item())
        
        # We assume start_year is the year since we'll run it year-by-year in the bash script
        out_file = os.path.join(args.out_dir, f"target_y{args.start_year}_{idx:05d}_m{month}_l{lead+1}.pt")
        
        # Extract MJO
        rmm1, rmm2 = 0.0, 0.0
        if 'mjo' in batch and len(batch['mjo'].shape) >= 2:
            rmm1 = batch['mjo'][0, 0].item()
            rmm2 = batch['mjo'][0, 1].item()
            
        f_month = batch['month'].to(device).float()
        fsin_month = torch.sin(2 * np.pi * (f_month - 1) / 12)
        fcos_month = torch.cos(2 * np.pi * (f_month - 1) / 12)
            
        fake_date = datetime.date(args.start_year, month, 15)
        nao_val = noise_utils.get_nao_value(fake_date, nao_lookup) if nao_lookup else 0.0
        enso_val = noise_utils.get_enso_value(month, args.start_year, oni_lookup) if oni_lookup else 0.0

        save_data = {
            'winner_map': winner_map.cpu(),
            'month': month,
            'lead': lead,
            'fsin_month': fsin_month[0].cpu().item(),
            'fcos_month': fcos_month[0].cpu().item(),
            'mjo_rmm1': rmm1,
            'mjo_rmm2': rmm2,
            'nao_val': nao_val,
            'enso_val': enso_val
        }
        
        torch.save(save_data, out_file)
        
        # Logging
        unique, counts = torch.unique(winner_map, return_counts=True)
        stats_dict = {int(k): int(v) for k, v in zip(unique, counts)}
        total = sum(stats_dict.values())
        p_rand = (stats_dict.get(0, 0) / total) * 100
        p_mjo = (stats_dict.get(1, 0) / total) * 100
        p_nao = (stats_dict.get(2, 0) / total) * 100
        p_enso = (stats_dict.get(3, 0) / total) * 100
        
        if idx % 50 == 0:
            print(f"[{idx}] y{args.start_year}.m{month:02d}.l{lead+1} | "
                  f"Wins: Rand({p_rand:.0f}%) MJO({p_mjo:.0f}%) NAO({p_nao:.0f}%) ENSO({p_enso:.0f}%)", flush=True)

if __name__ == "__main__":
    main()
