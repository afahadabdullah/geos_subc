import os
import sys
import ctypes

# --- TACC/Remote Fix: Preload Conda libstdc++ ---
try:
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        libstd = os.path.join(conda_prefix, 'lib', 'libstdc++.so.6')
        if os.path.exists(libstd):
            ctypes.CDLL(libstd, mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass
# ------------------------------------------------
"""
GEOS S2S3 Training Script (Z-Score Normalization Experiment)

This script trains a Conditional UNet with Gaussian Diffusion (DDPM/DDIM) on Residuals.
Normalization: Per-Grid Z-Score (Mean/Std maps).

Usage:
    python ml_model/trainv2.py ...
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup


import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from ml_model.dataset import GeosSubCDataset
    from ml_model.model import ConditionalUNet, GaussianDiffusion
    from ml_model.utils import crps_ensemble, denormalize_zscore, denormalize_residual_zscore, plot_comparison
except ImportError:
    import dataset as GeosSubCDataset
    import model as model_module
    from model import ConditionalUNet, GaussianDiffusion
    from utils import crps_ensemble, denormalize_zscore, denormalize_residual_zscore, plot_comparison

def _ddim_sample_val(model, diffusion, forecast, observed, mjo_map, month_onehot,
                     n_steps=50, image_size=(181, 360), cmde_ratio=0.1):
    """
    DDIM sampling for validation.
    """
    device = forecast.device
    bs = forecast.shape[0]
    
    timestep_indices = np.linspace(diffusion.timesteps - 1, 0, n_steps, dtype=int)
    x = torch.randn(bs, 4, image_size[0], image_size[1], device=device)
    
    for i, t_curr in enumerate(timestep_indices):
        t_tensor = torch.tensor([t_curr], device=device).expand(bs)
        
        sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alpha_hats[t_tensor].view(-1, 1, 1, 1)
        cond_noise = torch.randn_like(forecast)
        noisy_forecast = forecast + (cmde_ratio * sqrt_one_minus_alpha * cond_noise)
        
        model_input = torch.cat([x, noisy_forecast, observed, mjo_map], dim=1)
        pred_noise = model(model_input, t_tensor, month_onehot)
        
        alpha_bar_t = diffusion.alpha_hats[t_curr]
        
        if i < len(timestep_indices) - 1:
            t_prev = timestep_indices[i + 1]
            alpha_bar_t_prev = diffusion.alpha_hats[t_prev]
        else:
            alpha_bar_t_prev = torch.tensor(1.0, device=device)
        
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        
        pred_x0 = (x - sqrt_one_minus_alpha_bar_t * pred_noise) / (sqrt_alpha_bar_t + 1e-8)
        # With ZSCORE_SCALE=3.0, data is in ~[-1, 1]. Clamp for stability.
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        term_1 = torch.sqrt(alpha_bar_t_prev) * pred_x0
        term_2 = torch.sqrt(1 - alpha_bar_t_prev) * pred_noise
        x = term_1 + term_2
    
    return x


def train_model():
    # Memory fragmentation management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    config = {
        "train_years": (1999, 2014),
        "val_years": (2015, 2016),
        "batch_size": 16,
        "num_epochs": 100,
        "lr": 1e-4,
        "image_size": (181, 360),
        "data_root": "dataprocess",
        "output_dir": "ml_output_zscore",
        "gradient_accumulation_steps": 1,
        "cmde_ratio": 0.1,
    }
    
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with="tensorboard",
        project_dir=config["output_dir"]
    )
    
    # --------------------------------------------------------------------------
    # Dataset (Z-Score)
    # --------------------------------------------------------------------------
    if accelerator.is_main_process:
        print(f"Loading datasets (Z-Score Mode)...")

    # Pass normalization="zscore"
    train_dataset = GeosSubCDataset(
        data_root=config["data_root"],
        start_year=config["train_years"][0], 
        end_year=config["train_years"][1],
        mjo_file="mjo_processed.csv",
        preload=True,
        normalization="zscore"
    )
    
    val_dataset = GeosSubCDataset(
        data_root=config["data_root"],
        start_year=config["val_years"][0], 
        end_year=config["val_years"][1],
        mjo_file="mjo_processed.csv",
        preload=True,
        normalization="zscore"
    )
    
    # Load Maps for Denormalization (keep on CPU or move to GPU later)
    # Maps are numpy arrays in dataset.maps
    maps = train_dataset.maps
    
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    
    # --------------------------------------------------------------------------
    # DIAGNOSTIC: Pre-training Data Check
    # --------------------------------------------------------------------------
    if accelerator.is_main_process:
        print("\n--- DIAGNOSTIC: Checking Data Distribution (Z-Score) ---")
        try:
            sample = train_dataset[0]
            f_norm = sample["input_forecast"].numpy()
            t_norm = sample["target_truth"].numpy()
            r_norm = sample["target_residual"].numpy()
            
            print(f"Norm Stats (Min/Max/Mean/Std):")
            print(f"  Forecast : {f_norm.min():.2f} / {f_norm.max():.2f} / {f_norm.mean():.2f} / {f_norm.std():.2f}")
            print(f"  Target   : {t_norm.min():.2f} / {t_norm.max():.2f} / {t_norm.mean():.2f} / {t_norm.std():.2f}")
            print(f"  Residual : {r_norm.min():.2f} / {r_norm.max():.2f} / {r_norm.mean():.2f} / {r_norm.std():.2f}")
            
            # Reconstruct Sample 0
            # Need Mean/Std maps for reconstruction
            # Maps are (H, W). Sample is (C, H, W).
            f_raw = denormalize_zscore(f_norm, maps["geos_mean"], maps["geos_std"])
            t_raw = denormalize_zscore(t_norm, maps["gpcp_mean"], maps["gpcp_std"])
            
            print(f"Reconstructed Raw Stats (mm/day):")
            print(f"  Forecast: {f_raw.min():.2f} / {f_raw.max():.2f} / {f_raw.mean():.2f} / {f_raw.std():.2f}")
            print(f"  Target  : {t_raw.min():.2f} / {t_raw.max():.2f} / {t_raw.mean():.2f} / {t_raw.std():.2f}")

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 3, figsize=(18, 10))
            ax[0,0].hist(f_norm.flatten(), bins=50, color='blue', alpha=0.7)
            ax[0,0].set_title("Z-Norm Forecast")
            ax[0,1].hist(t_norm.flatten(), bins=50, color='green', alpha=0.7)
            ax[0,1].set_title("Z-Norm Target")
            ax[0,2].hist(r_norm.flatten(), bins=50, color='red', alpha=0.7)
            ax[0,2].set_title("Z-Norm Residual")
            
            ax[1,0].hist(f_raw.flatten(), bins=50, color='blue', alpha=0.7)
            ax[1,0].set_title("Raw Forecast (mm/day)")
            ax[1,0].set_yscale('log')
            ax[1,1].hist(t_raw.flatten(), bins=50, color='green', alpha=0.7)
            ax[1,1].set_title("Raw Target (mm/day)")
            ax[1,1].set_yscale('log')
            
            plt.tight_layout()
            os.makedirs(f"{config['output_dir']}/plots", exist_ok=True)
            plt.savefig(f"{config['output_dir']}/plots/diagnostic_pretrain.png")
            plt.close()
            print("Diagnostic plot saved.")
            
        except Exception as e:
            print(f"Diagnostic check failed: {e}")
            import traceback
            traceback.print_exc()

    # --------------------------------------------------------------------------
    # Model Setup
    # --------------------------------------------------------------------------
    model = ConditionalUNet(
        in_channels=14, 
        out_channels=4, 
        base_filters=128
    )
    
    diffusion = GaussianDiffusion(timesteps=1000, device=accelerator.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=1000, 
        num_training_steps=len(train_dataloader) * config["num_epochs"]
    )
    
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    diffusion.device = accelerator.device
    diffusion.betas = diffusion.betas.to(accelerator.device)
    diffusion.alphas = diffusion.alphas.to(accelerator.device)
    diffusion.alpha_hats = diffusion.alpha_hats.to(accelerator.device)
    diffusion.sqrt_alpha_hats = diffusion.sqrt_alpha_hats.to(accelerator.device)
    diffusion.sqrt_one_minus_alpha_hats = diffusion.sqrt_one_minus_alpha_hats.to(accelerator.device)

    # Move maps to device for validation
    gpu_maps = {}
    if accelerator.is_main_process:
        for k, v in maps.items():
            gpu_maps[k] = torch.tensor(v, device=accelerator.device, dtype=torch.float32)

    # Auto-resume... (omitted for brevity standard code)
    latest_path = os.path.join(config["output_dir"], "latest_checkpoint")
    start_epoch = 0
    best_val_loss = float('inf')
    best_checkpoints = [] # For top-5
    
    if os.path.exists(latest_path):
        if accelerator.is_main_process:
            print(f"Loading checkpoint from: {latest_path}")
        accelerator.load_state(latest_path)
        epoch_file = os.path.join(latest_path, "epoch.json")
        if os.path.exists(epoch_file):
            with open(epoch_file, 'r') as f:
                meta = json.load(f)
                start_epoch = meta.get("epoch", 0) + 1
                best_val_loss = meta.get("best_val_loss", float('inf'))

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    if accelerator.is_main_process:
        print(f"Starting training. Samples: {len(train_dataset)}")
    
    global_step = start_epoch * len(train_dataloader)
    
    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                target_residual = batch["target_residual"] 
                forecast = batch["input_forecast"]    
                observed = batch["observed_state"]    
                mjo = batch["mjo_conditioning"]       
                month_onehot = batch["month_onehot"]  
                
                bs = target_residual.shape[0]
                timesteps = diffusion.sample_timesteps(bs)
                noisy_target, noise = diffusion.add_noise(target_residual, timesteps)
                
                sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alpha_hats[timesteps].view(-1, 1, 1, 1)
                cond_noise = torch.randn_like(forecast)
                noisy_forecast = forecast + (config["cmde_ratio"] * sqrt_one_minus_alpha * cond_noise)
                
                mjo_map = mjo.view(bs, 2, 1, 1).expand(-1, -1, config["image_size"][0], config["image_size"][1])
                model_input = torch.cat([noisy_target, noisy_forecast, observed, mjo_map], dim=1)
                
                noise_pred = model(model_input, timesteps, month_onehot)
                loss = F.mse_loss(noise_pred, noise)
                
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            train_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
            global_step += 1
            
        progress_bar.close()
               
        # Validation
        if epoch % 1 == 0:
            model.eval()
            val_loss = 0.0
            if accelerator.is_main_process:
                print("Validating...")
            
            first_batch_data = None
            
            with torch.no_grad():
                for idx, batch in enumerate(val_dataloader):
                    target_residual = batch["target_residual"]
                    forecast = batch["input_forecast"]
                    observed = batch["observed_state"]
                    mjo = batch["mjo_conditioning"]
                    month_onehot = batch["month_onehot"]
                    target_truth = batch["target_truth"]
                    
                    bs = target_residual.shape[0]
                    timesteps = diffusion.sample_timesteps(bs)
                    noisy_target, noise = diffusion.add_noise(target_residual, timesteps)
                    
                    sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alpha_hats[timesteps].view(-1, 1, 1, 1)
                    cond_noise = torch.randn_like(forecast)
                    noisy_forecast = forecast + (config["cmde_ratio"] * sqrt_one_minus_alpha * cond_noise)
                    mjo_map = mjo.view(bs, 2, 1, 1).expand(-1, -1, config["image_size"][0], config["image_size"][1])
                    model_input = torch.cat([noisy_target, noisy_forecast, observed, mjo_map], dim=1)
                    
                    noise_pred = model(model_input, timesteps, month_onehot)
                    # Validation loss in normalized space (unitless)
                    loss = F.mse_loss(noise_pred, noise)
                    val_loss += loss.item()

                    if idx == 0 and accelerator.is_main_process:
                        first_batch_data = (target_truth[0:1], forecast[0:1], observed[0:1], mjo_map[0:1], month_onehot[0:1])

            avg_val_loss = val_loss / len(val_dataloader)
            
            if accelerator.is_main_process:
                print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")
                is_best = avg_val_loss < best_val_loss
                
                if (epoch % 5 == 0 or is_best) and first_batch_data is not None:
                    target_s, forecast_s, observed_s, mjo_s, month_s = first_batch_data
                    
                    samples_list = []
                    with torch.no_grad():
                        for _ in range(3):
                            pred_residual = _ddim_sample_val(
                                model, diffusion, forecast_s, observed_s, mjo_s, month_s,
                                n_steps=50, image_size=config["image_size"],
                                cmde_ratio=config["cmde_ratio"]
                            )
                            # Reconstruct using GPU maps
                            pred_physical = denormalize_residual_zscore(
                                pred_residual[0], forecast_s[0], 
                                gpu_maps["resid_mean"], gpu_maps["resid_std"],
                                gpu_maps["geos_mean"], gpu_maps["geos_std"]
                            )
                            samples_list.append(pred_physical)
                    
                    ens_mean_recon = torch.stack(samples_list).mean(dim=0).detach().cpu().numpy()
                    input_raw = denormalize_zscore(forecast_s[0], gpu_maps["geos_mean"], gpu_maps["geos_std"]).detach().cpu().numpy()
                    target_raw = denormalize_zscore(target_s[0], gpu_maps["gpcp_mean"], gpu_maps["gpcp_std"]).detach().cpu().numpy()
                    
                    suffix = "_best" if is_best else ""
                    plot_save_path = f"{config['output_dir']}/plots/epoch_{epoch}{suffix}.png"
                    plot_comparison(
                        input_raw, target_raw, ens_mean_recon, 
                        plot_save_path,
                        title=f"Epoch {epoch} (Z-Score) - Residual Ensemble"
                    )
                    print(f"Plot saved to: {plot_save_path}")
                
                if is_best:
                    best_val_loss = avg_val_loss
                    accelerator.save_state(f"{config['output_dir']}/best_model_epoch_{epoch}")
        
        if (epoch + 1) % 1 == 0:
            latest_path = f"{config['output_dir']}/latest_checkpoint"
            accelerator.save_state(latest_path)
            if accelerator.is_main_process:
                with open(os.path.join(latest_path, "epoch.json"), "w") as f:
                    json.dump({"epoch": epoch, "best_val_loss": best_val_loss}, f)

if __name__ == "__main__":
    train_model()
