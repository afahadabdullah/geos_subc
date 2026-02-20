import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_model.dataset_hybrid import S2SHybridDataset

def calculate_stats():
    print("Initializing Dataset (normalize=False)...")
    # Load dataset without normalization to compute raw stats
    dataset = S2SHybridDataset(data_root="dataprocess", start_year=1999, end_year=2015, preload=False, normalize=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Calculating stats on {len(dataset)} samples...")
    
    # Welford's Algorithm or Two-Pass?
    # Two-pass is safer for 800 samples. 
    # Or just incremental sum.
    
    # Channels: 
    # x_geos: (B, M, 1, L, H, W). We want Mean/Std for 1 channel (Precip).
    # x_obs: (B, 16, H, W). We want Mean/Std per channel (16 channels).
    
    # 1. Calculate Sum and SumSq
    geos_sum = 0.0
    geos_sq_sum = 0.0
    geos_count = 0
    geos_min = float('inf')
    geos_max = float('-inf')
    
    obs_sum = torch.zeros(16).float()
    obs_sq_sum = torch.zeros(16).float()
    obs_count = 0
    obs_min = torch.full((16,), float('inf')).float()
    obs_max = torch.full((16,), float('-inf')).float()
    
    for batch in tqdm(loader):
        # GEOS: (B, M, 1, L, H, W)
        x_geos = batch['x_geos']
        # Obs: (B, 16, H, W)
        x_obs = batch['x_obs']
        
        # GEOS
        # Flatten to (N, 1)
        # We want global Precip mean/std
        gx = x_geos.view(-1, 1) # (B*M*L*H*W, 1)
        geos_sum += gx.sum(dim=0)
        geos_sq_sum += (gx ** 2).sum(dim=0)
        geos_count += gx.size(0)
        geos_min = min(geos_min, float(gx.min()))
        geos_max = max(geos_max, float(gx.max()))
        
        # Obs
        # We want stats per channel (16)
        # x_obs: (B, 16, H, W) -> Permute to (16, B*H*W)
        ox = x_obs.permute(1, 0, 2, 3).reshape(16, -1)
        obs_sum += ox.sum(dim=1)
        obs_sq_sum += (ox ** 2).sum(dim=1)
        obs_count += ox.size(1)
        
        # Batch max and min
        batch_min = ox.min(dim=1)[0]
        batch_max = ox.max(dim=1)[0]
        obs_min = torch.minimum(obs_min, batch_min)
        obs_max = torch.maximum(obs_max, batch_max)
        
    print("Computing Mean/Std...")
    
    # GEOS (Precip)
    geos_mean = geos_sum / geos_count
    geos_var = (geos_sq_sum / geos_count) - (geos_mean ** 2)
    geos_std = torch.sqrt(geos_var)
    
    # Obs (16 channels)
    obs_mean = obs_sum / obs_count
    obs_var = (obs_sq_sum / obs_count) - (obs_mean ** 2)
    obs_std = torch.sqrt(obs_var)
    
    print("GEOS Mean:", geos_mean)
    print("GEOS Std:", geos_std)
    print("GEOS Min:", geos_min)
    print("GEOS Max:", geos_max)
    print("Obs Mean:", obs_mean)
    print("Obs Std:", obs_std)
    print("Obs Min:", obs_min)
    print("Obs Max:", obs_max)
    
    # Save
    save_path = os.path.join(os.path.dirname(__file__), "global_stats.pt")
    torch.save({
        "geos_mean": geos_mean,
        "geos_std": geos_std,
        "geos_min": torch.tensor(geos_min),
        "geos_max": torch.tensor(geos_max),
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "obs_min": obs_min,
        "obs_max": obs_max
    }, save_path)
    print(f"Saved stats to {save_path}")

if __name__ == "__main__":
    calculate_stats()
