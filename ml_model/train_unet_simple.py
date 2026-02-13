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
GEOS S2S3 Bias Correction — Simple Residual CNN Baseline

This script trains a 3-Layer CNN to predict a RESIDUAL CORRECTION to the GEOS Forecast.
Output = GEOS_Forecast + CNN(GEOS_Forecast)

Goal: Verify if a trivial model can improve upon the baseline.
If this learns, the problem was the UNet architecture.
If this fails, the problem is data/loss.

Loss: PHYSICAL SPACE MSE (mm/day)^2
Input: GEOS Only (4 channels)
"""

import torch
import torch.nn as nn
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
    from ml_model.utils import denormalize, plot_comparison
    from ml_model.loss import WeightedL1Loss, SSIMLoss
except ImportError:
    from dataset import GeosSubCDataset
    from utils import denormalize, plot_comparison
    from loss import WeightedL1Loss, SSIMLoss

# ==============================================================================
# LOSS FUNCTIONS - MOVED TO ml_model/loss.py
# ==============================================================================
# (GradientLoss class removed as it's replaced by WeightedL1 + SSIM)

# ==============================================================================
# SIMPLE CNN MODEL
# ==============================================================================
class SimpleCNN(nn.Module):
    """
    3-Layer ResNet-like block.
    Learn correction: f(x) -> residual
    Output = x + f(x)
    """
    def __init__(self, in_channels=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, in_channels, 3, padding=1)
        )
        
        # Initialize last layer to near-zero so we start close to Identity
        nn.init.uniform_(self.net[-1].weight, -0.001, 0.001)
        nn.init.constant_(self.net[-1].bias, 0)
        
    def forward(self, x, emb=None):
        # x: (B, 4, H, W)
        correction = self.net(x)
        return x + correction

# ==============================================================================
# TRAINING SCRIPT
# ==============================================================================
def train_model():
    # Memory fragmentation management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    config = {
        "train_years": (1999, 2014),
        "val_years": (2015, 2016),
        "batch_size": 32, # Increased for simple model
        "num_epochs": 100,
        "lr": 1e-4,
        "image_size": (181, 360),
        "data_root": "dataprocess",
        "output_dir": "ml_output_simple_cnn", 
        "gradient_accumulation_steps": 1,
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
    train_dataset = GeosSubCDataset(
        data_root=config["data_root"],
        start_year=config["train_years"][0], 
        end_year=config["train_years"][1],
        mjo_file="mjo_processed.csv",
        preload=True,
        ocean_vars=True # Loaded but not used
    )
    
    val_dataset = GeosSubCDataset(
        data_root=config["data_root"],
        start_year=config["val_years"][0], 
        end_year=config["val_years"][1],
        mjo_file="mjo_processed.csv",
        preload=True,
        ocean_vars=True
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    # Capture Norm Stats for Loss Function
    norm_min = train_dataset.norm_min
    norm_max = train_dataset.norm_max
    
    # Helper to denormalize batch in-graph
    def denormalize_batch(x_norm):
        # x_norm: [-1, 1]
        x = (x_norm + 1.0) / 2.0
        denom = norm_max - norm_min if norm_max != norm_min else 1.0
        x = x * denom + norm_min
        # Expm1 to Physical
        # Safe clamp for exp
        return torch.expm1(torch.clamp(x, max=10.0))

    # --------------------------------------------------------------------------
    # Model Setup
    # --------------------------------------------------------------------------
    model = SimpleCNN()
    
    # ADVANCED LOSSES
    # weighted_l1: Penalize heavy rain more (scale=5.0)
    # ssim: Penalize blurriness
    loss_weighted_l1 = WeightedL1Loss(scale=5.0)
    loss_ssim_fn = SSIMLoss()
    
    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: SimpleCNN (Residual) | Params: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=1000, 
        num_training_steps=len(train_dataloader) * config["num_epochs"]
    )
    
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    if accelerator.is_main_process:
        print(f"Starting training. Optimization Target: WEIGHTED_L1 + 0.2*SSIM")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                target_truth_norm = batch["target_truth"]
                forecast_norm = batch["input_forecast"]
                month_onehot = batch["month_onehot"] # Unused but available
                
                # Model predicts corrected forecast (norm space)
                pred_norm = model(forecast_norm)
                
                # Denormalize to Physical
                pred_mm = denormalize_batch(pred_norm)
                target_mm = denormalize_batch(target_truth_norm)
                
                # Loss Calculation
                l_wl1 = loss_weighted_l1(pred_mm, target_mm)
                l_ssim = loss_ssim_fn(pred_mm, target_mm) # Returns (1 - SSIM)
                
                loss = l_wl1 + 0.2 * l_ssim
                
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            train_loss += loss.item()
            pred_mean = pred_mm.mean().item()
            
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.2f}", "p_mu": f"{pred_mean:.2f}"})
            
            if global_step % 100 == 0:
                accelerator.log({"train_loss": loss.item()}, step=global_step)
            global_step += 1
            
        progress_bar.close()
        avg_train_loss = train_loss / len(train_dataloader)
        
        # ======================================================================
        # Validation
        # ======================================================================
        model.eval()
        val_loss = 0.0
        first_batch_data = None
        
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                target_truth_norm = batch["target_truth"]
                forecast_norm = batch["input_forecast"]
                
                pred_norm = model(forecast_norm)
                
                pred_mm = denormalize_batch(pred_norm)
                target_mm = denormalize_batch(target_truth_norm)
                
                # Loss Calculation
                l_wl1 = loss_weighted_l1(pred_mm, target_mm)
                l_ssim = loss_ssim_fn(pred_mm, target_mm)
                
                loss = l_wl1 + 0.2 * l_ssim
                val_loss += loss.item()

                # Capture first batch for plotting (on main process)
                if first_batch_data is None and accelerator.is_main_process:
                    first_batch_data = (
                        target_truth_norm[0:1].detach().cpu(), 
                        forecast_norm[0:1].detach().cpu(), 
                        pred_norm[0:1].detach().cpu()
                    )

        avg_val_loss = val_loss / len(val_dataloader)
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | Best: {best_val_loss:.4f}")
            accelerator.log({"val_loss": avg_val_loss}, step=global_step)
            
            is_best = avg_val_loss < best_val_loss
            
            # Plot ONLY if new best model found
            if is_best and first_batch_data is not None:
                try:
                    target_s, forecast_s, pred_precip_s = first_batch_data
                    
                    # Move to CPU for plotting (if not already)
                    # Use utils.denormalize for consistency with plots
                    pred_raw = denormalize(pred_precip_s[0]).numpy()
                    input_raw = denormalize(forecast_s[0]).numpy()
                    target_raw = denormalize(target_s[0]).numpy()
                    
                    plot_save_path = f"{config['output_dir']}/plots/epoch_{epoch}.png"
                    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
                    
                    plot_comparison(
                        input_raw, target_raw, pred_raw, 
                        plot_save_path,
                        title=f"Epoch {epoch} — CNN (Loss={avg_val_loss:.2f})"
                    )
                    print(f"  Plot saved to {plot_save_path}")
                except Exception as e:
                    print(f"  Plotting failed: {e}")
                
            if is_best:
                best_val_loss = avg_val_loss
                accelerator.save_state(f"{config['output_dir']}/best_model")
        
        # Save latest checkpoint for resuming
        accelerator.save_state(f"{config['output_dir']}/latest_checkpoint")
        if accelerator.is_main_process:
            with open(os.path.join(config['output_dir'], "latest_checkpoint", "epoch.json"), "w") as f:
                json.dump({"epoch": epoch, "best_val_loss": best_val_loss}, f)

if __name__ == "__main__":
    train_model()
