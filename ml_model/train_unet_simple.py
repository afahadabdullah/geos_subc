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
GEOS S2S3 Bias Correction — Simple UNet Baseline

This script trains a TemporalAttentionUNet to DIRECTLY PREDICT GPCP precipitation
using ONLY GEOS FORECAST as input.

Architecture: Conv2D UNet with temporal self-attention at the bottleneck.

Input (4 channels):
    1. GEOS Forecast (4 channels)
    - NO OTHER INPUTS (No SST, SSS, MJO, Observed)

Loss:
    - Standard MSE Loss (Mean Squared Error)

Usage:
    python ml_model/train_unet_simple.py
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
    from ml_model.model_unet import TemporalAttentionUNet
    from ml_model.utils import denormalize, plot_comparison
except ImportError:
    from dataset import GeosSubCDataset
    from model_unet import TemporalAttentionUNet
    from utils import denormalize, plot_comparison


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
        "output_dir": "ml_output_unet_simple", # Changed output dir
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
    if accelerator.is_main_process:
        print(f"Loading datasets from {config['data_root']}...")

    train_dataset = GeosSubCDataset(
        data_root=config["data_root"],
        start_year=config["train_years"][0], 
        end_year=config["train_years"][1],
        mjo_file="mjo_processed.csv",
        preload=True,
        ocean_vars=True # Loaded but not used in model input
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
    
    # --------------------------------------------------------------------------
    # DIAGNOSTIC
    # --------------------------------------------------------------------------
    if accelerator.is_main_process:
        print("\n--- DIAGNOSTIC: Checking Data Distribution ---")
        try:
            sample = train_dataset[0]
            f_norm = sample["input_forecast"].numpy()
            t_norm = sample["target_truth"].numpy()
            print(f"  Forecast : {f_norm.min():.2f} / {f_norm.max():.2f} / {f_norm.mean():.2f} / {f_norm.std():.2f}")
            print(f"  Target   : {t_norm.min():.2f} / {t_norm.max():.2f} / {t_norm.mean():.2f} / {t_norm.std():.2f}")
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # Model Setup (Deterministic UNet + Temporal Attention)
    # --------------------------------------------------------------------------
    # Inputs (SIMPLE BASELINE):
    #    - GEOS Forecast (4 Channels)
    # Total Input Channels = 4
    
    in_channels = 4
    out_channels = 4
    
    model = TemporalAttentionUNet(
        in_channels=in_channels, 
        out_channels=out_channels, 
        base_filters=128,
        emb_dim=256,
        n_weeks=4,
        temporal_heads=4
    )
    
    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: Simple UNet (GEOS Only) | Params: {n_params:,}")
    
    # Loss: Standard MSE
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=1000, 
        num_training_steps=len(train_dataloader) * config["num_epochs"]
    )
    
    # Prepare with Accelerate
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # Auto-resume Logic
    latest_path = os.path.join(config["output_dir"], "latest_checkpoint")
    start_epoch = 0
    best_val_loss = float('inf')
    best_checkpoints = []
    
    if os.path.exists(latest_path):
        if accelerator.is_main_process:
            print(f"Loading checkpoint from: {latest_path}")
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        accelerator.load_state(latest_path)
        
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
        print(f"Starting training with Simple UNet. Samples: {len(train_dataset)}")
    
    global_step = start_epoch * len(train_dataloader)
    
    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # 1. Unpack
                target_truth = batch["target_truth"]        # (B, 4, Y, X)
                forecast = batch["input_forecast"]          # (B, 4, Y, X)
                month_onehot = batch["month_onehot"]        # (B, 12)
                
                # 2. Input = Forecast Only
                model_input = forecast
                
                # 3. Forward pass
                pred_precip = model(model_input, month_onehot)
                
                # 4. Loss (MSE on normalized values)
                loss = criterion(pred_precip, target_truth)
                
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            train_loss += loss.item()
            
            # Logging
            pred_mean = pred_precip.mean().item()
            pred_std = pred_precip.std().item()
            
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "p_mu": f"{pred_mean:.3f}", "p_std": f"{pred_std:.3f}"})
            
            if global_step % 100 == 0:
                accelerator.log({
                    "train_loss": loss.item(), 
                    "pred_mean": pred_mean, 
                    "pred_std": pred_std
                }, step=global_step)
            
            global_step += 1
            
        progress_bar.close()
        avg_train_loss = train_loss / len(train_dataloader)
        
        # ======================================================================
        # Validation (Every Epoch)
        # ======================================================================
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        first_batch_data = None
        
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                target_truth = batch["target_truth"]
                forecast = batch["input_forecast"]
                month_onehot = batch["month_onehot"]
                
                model_input = forecast
                pred_precip = model(model_input, month_onehot)
                
                loss = criterion(pred_precip, target_truth)
                val_loss += loss.item()
                n_val_batches += 1

                if idx == 0 and accelerator.is_main_process:
                    first_batch_data = (
                        target_truth[0:1], forecast[0:1], 
                        pred_precip[0:1].detach()
                    )

        avg_val_loss = val_loss / len(val_dataloader)
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch} | Val Loss (MSE): {avg_val_loss:.4f}")
            accelerator.log({"val_loss": avg_val_loss}, step=global_step)
            
            is_best = avg_val_loss < best_val_loss
            
            # ------------------------------------------------------------------
            # Validation Plot
            # ------------------------------------------------------------------
            if (epoch % 5 == 0 or is_best) and first_batch_data is not None:
                target_s, forecast_s, pred_precip_s = first_batch_data
                
                # Reconstruct physical precip from predicted precip directly
                pred_raw = denormalize(pred_precip_s[0]).detach().cpu().numpy()
                input_raw = denormalize(forecast_s[0]).detach().cpu().numpy()
                target_raw = denormalize(target_s[0]).detach().cpu().numpy()
                
                suffix = "_best" if is_best else ""
                plot_save_path = f"{config['output_dir']}/plots/epoch_{epoch}{suffix}.png"
                plot_comparison(
                    input_raw, target_raw, pred_raw, 
                    plot_save_path,
                    title=f"Epoch {epoch} — Simple UNet (MSE) {'(Best)' if is_best else ''}"
                )
                print(f"  Validation plot saved to: {plot_save_path}")
            
            # ------------------------------------------------------------------
            # Checkpoint
            # ------------------------------------------------------------------
            if is_best:
                print(f"  ★ New best model at epoch {epoch}")
                best_val_loss = avg_val_loss
                checkpoint_path = f"{config['output_dir']}/best_model_epoch_{epoch}"
                accelerator.save_state(checkpoint_path)
                
                best_checkpoints.append(checkpoint_path)
                if len(best_checkpoints) > 4:
                    old_checkpoint = best_checkpoints.pop(0)
                    import shutil
                    if os.path.exists(old_checkpoint):
                        shutil.rmtree(old_checkpoint, ignore_errors=True)
        
        latest_path = f"{config['output_dir']}/latest_checkpoint"
        accelerator.save_state(latest_path)
        if accelerator.is_main_process:
            with open(os.path.join(latest_path, "epoch.json"), "w") as f:
                json.dump({"epoch": epoch, "best_val_loss": best_val_loss}, f)

    if accelerator.is_main_process:
        print("Training finished.")

if __name__ == "__main__":
    train_model()
