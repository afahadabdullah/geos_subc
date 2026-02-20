import torch
import yaml
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml_model.dataset_hybrid import S2SHybridDataset
from ml_model.diffusion import ConditionalDiffusion
from ml_model.train_diffusion import load_topography

def print_stat(name, tensor):
    if isinstance(tensor, torch.Tensor):
        float_tensor = tensor.float()
        print(f"{name+':':<20} shape={list(tensor.shape)} | type={tensor.dtype} | "
              f"min={float_tensor.min().item():>8.4f} | max={float_tensor.max().item():>8.4f} | "
              f"mean={float_tensor.mean().item():>8.4f} | std={float_tensor.std().item():>8.4f}")
    else:
        print(f"{name+':':<20} {tensor}")

def run_diagnostics():
    with open("ml_model/config_diffusion.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # 1. Dataset
    val_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["val_start_year"],
        end_year=config["val_end_year"],
        normalize=True,
        preload=False
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    batch = next(iter(val_loader))

    topo_tensor = load_topography(config["data_dir"]).to(device)

    # 2. Build Condition
    x_obs = batch['x_obs'].to(device)
    x_geos = batch['x_geos'].to(device)
    y_target = batch['y_target'].to(device)
    months = batch['month'].to(device)
    mjo = batch['mjo'].to(device)
    
    B, _, H, W = x_obs.shape
    x_geos_flat = x_geos.squeeze(2).reshape(B, 16, H, W)
    sin_m = torch.sin(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W)
    cos_m = torch.cos(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W)
    mjo_map = mjo.view(B, 2, 1, 1).expand(B, 2, H, W)
    topo_batch = topo_tensor.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)

    condition = torch.cat([x_obs, x_geos_flat, sin_m, cos_m, mjo_map, topo_batch], dim=1)
    
    # 3. Target Norm
    y_log = torch.log1p(y_target.clamp(min=0.0))
    gm = val_dataset.geos_mean.to(device) if val_dataset.geos_mean is not None else 0.0
    gs = val_dataset.geos_std.to(device) if val_dataset.geos_std is not None else 1.0
    target_norm = (y_log - gm) / gs

    print("\n--- DATA INPUT STATS ---")
    print_stat("y_target (raw)", y_target)
    print_stat("y_log (log1p)", y_log)
    print_stat("target_norm", target_norm)
    print_stat("condition", condition)

    # 4. Model
    model = ConditionalDiffusion(
        in_channels=4, condition_channels=49, out_channels=4,
        block_out_channels=(64, 128, 256, 512), layers_per_block=2,
        num_train_timesteps=1000, cmde_ratio=0.0
    ).to(device)

    ckpt_path = os.path.join(config["output_dir"], "latest_diffusion_ckpt.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f"\nLoaded Checkpoint! Epoch: {ckpt['epoch']}")
    else:
        print("\nNO CHECKPOINT FOUND. Testing with untrained weights.")
    model.eval()

    # 5. Training Forward Pass Check
    print("\n--- TRAINING NOISE PREDICTION CHECK ---")
    t_train = torch.tensor([500], device=device).long()
    noise = torch.randn_like(target_norm)
    noisy_target = model.noise_scheduler.add_noise(target_norm, noise, t_train)
    
    print_stat("t_train", t_train)
    print_stat("noise (true)", noise)
    print_stat("noisy_target", noisy_target)

    with torch.no_grad():
        pred_noise = model(noisy_target, condition, t_train)
    print_stat("pred_noise", pred_noise)
    loss = torch.mean((pred_noise - noise)**2)
    print(f"Single-Batch Noise MSE Loss at t=500: {loss.item():.4f}")

    # 6. Sampling Check (Trace 5 steps)
    print("\n--- DDIM INFERENCE CHECK (First 5 steps of 50) ---")
    batch_size = condition.shape[0]
    orig_H, orig_W = condition.shape[2], condition.shape[3]
    
    latents = torch.randn((batch_size, 4, orig_H, orig_W), device=device)
    print_stat("latents (Initial)", latents)
    
    model.noise_scheduler.set_timesteps(50)
    timesteps = model.noise_scheduler.timesteps
    
    with torch.no_grad():
        for i, t in enumerate(timesteps[:5]):
            print(f"\n[Step {i+1}] t = {t.item()}")
            
            # Replicate forward() to log internals
            lat_padded, pH, pW = model._pad_to_multiple(latents)
            cond_padded, _, _ = model._pad_to_multiple(condition)
            
            model_input = torch.cat([lat_padded, cond_padded], dim=1)
            t_expand = t.expand(batch_size).to(device) # ensure vector
            
            unet_out = model.model(model_input, t_expand).sample
            unet_cropped = unet_out[..., :orig_H, :orig_W]
            
            print_stat("model_input (pad)", model_input)
            print_stat("unet_out (pad)", unet_out)
            print_stat("noise_pred (crop)", unet_cropped)
            
            latents = model.noise_scheduler.step(unet_cropped, t, latents).prev_sample
            print_stat("latents (Updated)", latents)

if __name__ == "__main__":
    run_diagnostics()
