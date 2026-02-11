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
    from ml_model.utils import crps_ensemble, denormalize, plot_comparison
except ImportError:
    import dataset as GeosSubCDataset
    import model as model_module
    from model import ConditionalUNet, GaussianDiffusion
    from utils import crps_ensemble, denormalize, plot_comparison

def _ddim_sample_val(model, diffusion, forecast, observed, mjo_map, month_onehot,
                     n_steps=50, image_size=(181, 360), cmde_ratio=0.1):
    """
    Real DDIM sampling for validation plots.
    Starts from pure noise (not ground truth) for honest evaluation.
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
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0) # Clip to valid range for stability
        
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
        "batch_size": 32, # Increased back to 32 (4x the previous limit) for H200
        "num_epochs": 100,
        "lr": 1e-4,
        "image_size": (181, 360),
        "data_root": "dataprocess",
        "output_dir": "ml_output_cmde",
        "gradient_accumulation_steps": 1,
        "cmde_ratio": 0.1, # Ratio for noise on condition
    }
    
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with="tensorboard",
        project_dir=config["output_dir"]
    )

    if accelerator.is_main_process:
        print(f"Device: {accelerator.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # --------------------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------------------
    if accelerator.is_main_process:
        print(f"Loading datasets from {config['data_root']}...")

    train_dataset = GeosSubCDataset(
        data_root=config["data_root"],
        start_year=config["train_years"][0], 
        end_year=config["train_years"][1],
        mjo_file="mjo_processed.csv",
        preload=True  # Preload to RAM for faster I/O
    )
    
    val_dataset = GeosSubCDataset(
        data_root=config["data_root"],
        start_year=config["val_years"][0], 
        end_year=config["val_years"][1],
        mjo_file="mjo_processed.csv",
        preload=True
    )
    
    # pin_memory=True helps with CPU->GPU transfer
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    
    # --------------------------------------------------------------------------
    # Model Setup (CMDE Architecture)
    # --------------------------------------------------------------------------
    # Inputs:
    # 1. Noisy Residual (4 Channels for 4 weeks)
    # 2. Condition:
    #    - Noisy Forecast (4 Channels)
    #    - GPCP Observed State (4 Channels — weeks before init)
    #    - MJO Features (Broadcasted -> 2 Channels)
    # Total Input Channels to UNet = 4 + 4 + 4 + 2 = 14
    
    in_channels = 14 
    out_channels = 4 # Predicted Noise for Residual
    
    model = ConditionalUNet(
        in_channels=in_channels, 
        out_channels=out_channels, 
        base_filters=128
    )
    
    # Custom Gaussian Diffusion (defaults to cosine schedule now)
    diffusion = GaussianDiffusion(timesteps=1000, device=accelerator.device) # Will update device in loop
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=1000, 
        num_training_steps=len(train_dataloader) * config["num_epochs"]
    )
    
    # Prepare
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # Update diffusion device after accelerate prepare? 
    # Actually diffusion class is stateless regarding parameters, but stores tensors.
    # We should update its device.
    diffusion.device = accelerator.device
    diffusion.betas = diffusion.betas.to(accelerator.device)
    diffusion.alphas = diffusion.alphas.to(accelerator.device)
    diffusion.alpha_hats = diffusion.alpha_hats.to(accelerator.device)
    diffusion.sqrt_alpha_hats = diffusion.sqrt_alpha_hats.to(accelerator.device)
    diffusion.sqrt_one_minus_alpha_hats = diffusion.sqrt_one_minus_alpha_hats.to(accelerator.device)

    # Auto-resume Logic
    latest_path = os.path.join(config["output_dir"], "latest_checkpoint")
    start_epoch = 0
    best_val_loss = float('inf')  # Default for fresh start
    best_checkpoints = []
    
    if os.path.exists(latest_path):
        if accelerator.is_main_process:
            print(f"Loading checkpoint from: {latest_path}")
        accelerator.load_state(latest_path)
        
        # Load epoch metadata
        epoch_file = os.path.join(latest_path, "epoch.json")
        if os.path.exists(epoch_file):
            with open(epoch_file, 'r') as f:
                meta = json.load(f)
                start_epoch = meta.get("epoch", 0) + 1
                best_val_loss = meta.get("best_val_loss", float('inf'))
                if accelerator.is_main_process:
                    print(f"Resuming from epoch {start_epoch} (Best Val Loss: {best_val_loss:.4f})")

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    if accelerator.is_main_process:
        print(f"Starting training with CMDE architecture. Samples: {len(train_dataset)}")
    
    global_step = start_epoch * len(train_dataloader)
    
    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        train_loss = 0.0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # 1. Unpack & Normalize
                target_truth = batch["target_truth"]  # (B, 4, Y, X) - Already log-normalized
                forecast = batch["input_forecast"]    # (B, 4, Y, X)
                observed = batch["observed_state"]    # (B, 4, Y, X) - GPCP pre-init
                mjo = batch["mjo_conditioning"]       # (B, 2)
                month_onehot = batch["month_onehot"]  # (B, 12)
                
                # 2. Add Noise to Target
                target = target_truth
                bs = target.shape[0]
                timesteps = diffusion.sample_timesteps(bs)
                noisy_target, noise = diffusion.add_noise(target, timesteps)
                
                # 4. CMDE Logic
                sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alpha_hats[timesteps].view(-1, 1, 1, 1)
                cond_noise = torch.randn_like(forecast)
                noisy_forecast = forecast + (config["cmde_ratio"] * sqrt_one_minus_alpha * cond_noise)
                
                # 5. Broadcast MJO
                mjo_map = mjo.view(bs, 2, 1, 1).expand(-1, -1, config["image_size"][0], config["image_size"][1])
                
                # 6. Cat Inputs: [noisy_target, noisy_forecast, observed_state, mjo_map]
                model_input = torch.cat([noisy_target, noisy_forecast, observed, mjo_map], dim=1)
                
                # 7. Predict Noise
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
               
        # Validation Logic (Every Epoch)
        if epoch % 1 == 0:
            model.eval()
            val_loss = 0.0
            if accelerator.is_main_process:
                print("Validating...")
            
            first_batch_data = None # Storage for plotting
            
            with torch.no_grad():
                for idx, batch in enumerate(val_dataloader):
                    target_truth = batch["target_truth"]
                    forecast = batch["input_forecast"]
                    observed = batch["observed_state"]
                    mjo = batch["mjo_conditioning"]
                    month_onehot = batch["month_onehot"]
                    
                    bs = target_truth.shape[0]
                    timesteps = diffusion.sample_timesteps(bs)
                    noisy_target, noise = diffusion.add_noise(target_truth, timesteps)
                    
                    sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alpha_hats[timesteps].view(-1, 1, 1, 1)
                    cond_noise = torch.randn_like(forecast)
                    noisy_forecast = forecast + (config["cmde_ratio"] * sqrt_one_minus_alpha * cond_noise)
                    mjo_map = mjo.view(bs, 2, 1, 1).expand(-1, -1, config["image_size"][0], config["image_size"][1])
                    model_input = torch.cat([noisy_target, noisy_forecast, observed, mjo_map], dim=1)
                    
                    noise_pred = model(model_input, timesteps, month_onehot)
                    loss = F.mse_loss(noise_pred, noise)
                    val_loss += loss.item()

                    # Cache first batch for plotting
                    if idx == 0 and accelerator.is_main_process:
                        first_batch_data = (target_truth[0:1], forecast[0:1], observed[0:1], mjo_map[0:1], month_onehot[0:1])

            avg_val_loss = val_loss / len(val_dataloader)
            
            if accelerator.is_main_process:
                print(f"Epoch {epoch} Val Loss (MSE): {avg_val_loss:.4f}")
                
                is_best = avg_val_loss < best_val_loss
                
                # Plotting logic (Periodic OR Best) - Real DDIM sampling
                if (epoch % 5 == 0 or is_best) and first_batch_data is not None:
                    target_s, forecast_s, observed_s, mjo_s, month_s = first_batch_data
                    
                    samples_list = []
                    # Generate 3 ensemble members via real 50-step DDIM
                    with torch.no_grad():
                        for _ in range(3):
                            pred = _ddim_sample_val(
                                model, diffusion, forecast_s, observed_s, mjo_s, month_s,
                                n_steps=50, image_size=config["image_size"],
                                cmde_ratio=config["cmde_ratio"]
                            )
                            samples_list.append(denormalize(pred[0]))
                    
                    ens_mean_recon = torch.stack(samples_list).mean(dim=0).detach().cpu().numpy()
                    input_raw = denormalize(forecast_s[0]).detach().cpu().numpy()
                    target_raw = denormalize(target_s[0]).detach().cpu().numpy()
                    
                    suffix = "_best" if is_best else ""
                    plot_save_path = f"{config['output_dir']}/plots/epoch_{epoch}{suffix}.png"
                    plot_comparison(
                        input_raw, target_raw, ens_mean_recon, 
                        plot_save_path,
                        title=f"Epoch {epoch} - DDIM50 Ensemble {'(Best)' if is_best else ''}"
                    )
                    if accelerator.is_main_process:
                        print(f"Ensembled validation plot saved to: {plot_save_path}")
                
                # Check for Best Model
                if is_best:
                    print(f"New best model found at epoch {epoch} (Loss: {avg_val_loss:.4f})")
                    best_val_loss = avg_val_loss
                    checkpoint_path = f"{config['output_dir']}/best_model_epoch_{epoch}"
                    accelerator.save_state(checkpoint_path)
                    
                    best_checkpoints.append(checkpoint_path)
                    if len(best_checkpoints) > 4:
                        old_checkpoint = best_checkpoints.pop(0)
                        import shutil
                        if os.path.exists(old_checkpoint):
                            shutil.rmtree(old_checkpoint, ignore_errors=True)
        
        # Periodic Save for Resuming (Always keep latest)
        if (epoch + 1) % 1 == 0:
            latest_path = f"{config['output_dir']}/latest_checkpoint"
            accelerator.save_state(latest_path)
            if accelerator.is_main_process:
                with open(os.path.join(latest_path, "epoch.json"), "w") as f:
                    json.dump({"epoch": epoch, "best_val_loss": best_val_loss}, f)

    if accelerator.is_main_process:
        print("Training finished.")

if __name__ == "__main__":
    train_model()
