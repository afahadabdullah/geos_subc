import torch
import numpy as np
import yaml
import sys
import os
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_model.dataset_hybrid import S2SHybridDataset
from ml_model.train_diffusion import get_area_weights, load_topography

def analyze_batch():
    with open("ml_model/config_diffusion.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu')
    
    # Init dataset
    train_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["train_start_year"],
        end_year=config["train_end_year"],
        normalize=True,
        preload=False
    )
    
    loader = DataLoader(
        train_dataset, batch_size=4,
        shuffle=True, num_workers=0
    )
    
    topo_tensor = load_topography(config["data_dir"])
    TARGET_MEAN = 0.82
    TARGET_STD = 0.79

    for batch in loader:
        x_obs = batch['x_obs']           
        x_geos = batch['x_geos']         
        y_target = batch['y_target']     
        months = batch['month']          
        mjo = batch['mjo']              

        B, _, H, W = x_obs.shape

        x_geos_flat = x_geos.squeeze(2).reshape(B, 16, H, W)
        
        # Normalize GEOS condition
        if train_dataset.geos_mean is not None:
            x_geos_flat = (x_geos_flat - train_dataset.geos_mean.to(device)) / train_dataset.geos_std.to(device)
            print("GEOS Mean and Std applied:", train_dataset.geos_mean, train_dataset.geos_std)
        else:
            print("WARNING: GEOS Mean is None!")

        # Normalize Prev-GPCP
        x_obs[:, 12:16, :, :] = (x_obs[:, 12:16, :, :] - TARGET_MEAN) / TARGET_STD

        sin_month = torch.sin(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
        cos_month = torch.cos(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
        mjo_map = mjo.view(B, 2, 1, 1).expand(B, 2, H, W).to(device)
        topo_batch = topo_tensor.to(device).unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)

        condition = torch.cat([x_obs, x_geos_flat, sin_month, cos_month, mjo_map, topo_batch], dim=1)

        y_log = torch.log1p(y_target.clamp(min=0.0))
        target_norm = (y_log - TARGET_MEAN) / TARGET_STD

        print(f"Target Norm: mean={target_norm.mean().item():.3f}, std={target_norm.std().item():.3f}, min={target_norm.min().item():.3f}, max={target_norm.max().item():.3f}")
        
        print("====== Condition Stats by component ======")
        print(f"Condition overall   : mean={condition.mean():.3f}, std={condition.std():.3f}, min={condition.min():.3f}, max={condition.max():.3f}")
        print(f"x_obs (28 ch)     : mean={x_obs.mean():.3f}, std={x_obs.std():.3f}, min={x_obs.min():.3f}, max={x_obs.max():.3f}")
        print(f"  sst (0:4)       : mean={x_obs[:, 0:4].mean():.3f}, std={x_obs[:, 0:4].std():.3f}")
        print(f"  sss (4:8)       : mean={x_obs[:, 4:8].mean():.3f}, std={x_obs[:, 4:8].std():.3f}")
        print(f"  sm  (8:12)      : mean={x_obs[:, 8:12].mean():.3f}, std={x_obs[:, 8:12].std():.3f}")
        print(f"  prev (12:16)    : mean={x_obs[:, 12:16].mean():.3f}, std={x_obs[:, 12:16].std():.3f}")
        print(f"  ivt (16:20)     : mean={x_obs[:, 16:20].mean():.3f}, std={x_obs[:, 16:20].std():.3f}")
        print(f"  z500 (20:24)    : mean={x_obs[:, 20:24].mean():.3f}, std={x_obs[:, 20:24].std():.3f}")
        print(f"  u250 (24:28)    : mean={x_obs[:, 24:28].mean():.3f}, std={x_obs[:, 24:28].std():.3f}")
        print(f"x_geos (16 ch)    : mean={x_geos_flat.mean():.3f}, std={x_geos_flat.std():.3f}, min={x_geos_flat.min():.3f}, max={x_geos_flat.max():.3f}")
        print(f"sin/cos           : std={sin_month.std():.3f}, {cos_month.std():.3f}")
        print(f"mjo               : std={mjo_map.std():.3f}")
        print(f"topo              : std={topo_batch.std():.3f}")
        
        # Check if there's any huge variance causing NaN or blowing up group norm
        ch_vars = condition.var(dim=(2, 3)).mean(dim=0)
        print("Channel variances:", ch_vars.tolist()[:5], "...")
        print("Max channel variance:", ch_vars.max().item())
        
        break

if __name__ == '__main__':
    analyze_batch()
