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
GEOS S2S3 Bias Correction — Deterministic UNet + Temporal Attention (Ocean)

This script trains a TemporalAttentionUNet to directly predict the precipitation
residual (GPCP − GEOS) in a SINGLE forward pass. No diffusion process.

Architecture: Conv2D UNet with temporal self-attention at the bottleneck.
The model learns cross-week dependencies between the 4 lead weeks.

Key Differences vs train_ocean.py (Diffusion):
    - No noise schedule, no iterative denoising, no DDIM sampling
    - Direct regression: MSE + Huber loss on residual
    - Single forward pass at inference (fast evaluation)
    - Input: 18 channels (no noisy target needed)

Conditioning:
    1. GEOS Forecast (4 channels)
    2. Observed State — previous GPCP (4 channels)
    3. SST (4 channels)
    4. SSS (4 channels)
    5. MJO (RMM1, RMM2 broadcast) (2 channels)
    6. Month one-hot → FiLM embedding (scalar conditioning)

Usage:
    python ml_model/train_ocean_unet.py
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
    from ml_model.model_unet import TemporalAttentionUNet, AreaWeightedLoss
    from ml_model.utils import denormalize, denormalize_residual, plot_comparison
except ImportError:
    from dataset import GeosSubCDataset
    from model_unet import TemporalAttentionUNet, AreaWeightedLoss
    from utils import denormalize, denormalize_residual, plot_comparison


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
        "output_dir": "ml_output_ocean_unet",
        "gradient_accumulation_steps": 1,
        # Loss config
        "loss_alpha": 0.5,        # MSE weight (1-alpha = Huber weight)
        "huber_delta": 1.0,       # Huber loss delta
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
        ocean_vars=True
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
    # DIAGNOSTIC: Pre-training Data Check
    # --------------------------------------------------------------------------
    if accelerator.is_main_process:
        print("\n--- DIAGNOSTIC: Checking Data Distribution Before Training ---")
        try:
            sample = train_dataset[0]
            f_norm = sample["input_forecast"].numpy()
            t_norm = sample["target_truth"].numpy()
            r_norm = sample["target_residual"].numpy()
            
            vmin, vmax = train_dataset.norm_min, train_dataset.norm_max
            denom = vmax - vmin if vmax != vmin else 1.0
            
            def inv_func(x):
                val = (x + 1.0) / 2.0
                val = val * denom + vmin
                return np.expm1(val)
            
            f_raw = inv_func(f_norm)
            t_raw = inv_func(t_norm)
            
            print(f"Norm Stats (Min/Max/Mean/Std):")
            print(f"  Forecast : {f_norm.min():.2f} / {f_norm.max():.2f} / {f_norm.mean():.2f} / {f_norm.std():.2f}")
            print(f"  Target   : {t_norm.min():.2f} / {t_norm.max():.2f} / {t_norm.mean():.2f} / {t_norm.std():.2f}")
            print(f"  Residual : {r_norm.min():.2f} / {r_norm.max():.2f} / {r_norm.mean():.2f} / {r_norm.std():.2f}")
            
            print(f"Reconstructed Raw Stats (Min/Max/Mean/Std) [mm/day]:")
            print(f"  Forecast: {f_raw.min():.2f} / {f_raw.max():.2f} / {f_raw.mean():.2f} / {f_raw.std():.2f}")
            print(f"  Target  : {t_raw.min():.2f} / {t_raw.max():.2f} / {t_raw.mean():.2f} / {t_raw.std():.2f}")
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 3, figsize=(18, 10))
            
            ax[0,0].hist(f_norm.flatten(), bins=50, color='blue', alpha=0.7)
            ax[0,0].set_title("Normalized Forecast [-1, 1]")
            ax[0,1].hist(t_norm.flatten(), bins=50, color='green', alpha=0.7)
            ax[0,1].set_title("Normalized Target [-1, 1]")
            ax[0,2].hist(r_norm.flatten(), bins=50, color='red', alpha=0.7)
            ax[0,2].set_title("Normalized Residual [-1, 1]")
            ax[0,2].axvline(0, color='k', linestyle='--')
            
            ax[1,0].hist(f_raw.flatten(), bins=50, color='blue', alpha=0.7)
            ax[1,0].set_title("Raw Forecast (mm/day)")
            ax[1,0].set_yscale('log')
            ax[1,1].hist(t_raw.flatten(), bins=50, color='green', alpha=0.7)
            ax[1,1].set_title("Raw Target (mm/day)")
            ax[1,1].set_yscale('log')
            ax[1,2].text(0.5, 0.5, f"Residual\nShould be centered near 0\nMean: {r_norm.mean():.3f}", 
                        ha='center', va='center', fontsize=14, transform=ax[1,2].transAxes)
            ax[1,2].set_title("Residual Info")
            
            plt.tight_layout()
            os.makedirs(f"{config['output_dir']}/plots", exist_ok=True)
            plt.savefig(f"{config['output_dir']}/plots/diagnostic_pretrain.png")
            plt.close()
            print(f"Diagnostic plot saved to {config['output_dir']}/plots/diagnostic_pretrain.png")
            print("----------------------------------------------------------\n")
            
        except Exception as e:
            print(f"Diagnostic check failed: {e}")
            
    # --------------------------------------------------------------------------
    # Model Setup (Deterministic UNet + Temporal Attention)
    # --------------------------------------------------------------------------
    # Inputs (NO noisy target — no diffusion):
    #    - Forecast (4 Channels)
    #    - GPCP Observed State (4 Channels — weeks before init)
    #    - SST Observed (4 Channels)
    #    - SSS Observed (4 Channels)
    #    - MJO Features (Broadcasted → 2 Channels)
    # Total Input Channels = 4 + 4 + 4 + 4 + 2 = 18
    
    in_channels = 18
    out_channels = 4  # Predicted residual for 4 lead weeks
    
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
        print(f"Model: TemporalAttentionUNet | Params: {n_params:,}")
    
    # Loss: Combined MSE + Huber
    criterion = AreaWeightedLoss(
        n_lat=config["image_size"][0],
        n_lon=config["image_size"][1],
        lat_range=(90, -90),
        alpha=config["loss_alpha"],
        huber_delta=config["huber_delta"]
    ).to(accelerator.device)
    
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
        print(f"Starting training with TemporalAttentionUNet. Samples: {len(train_dataset)}")
    
    global_step = start_epoch * len(train_dataloader)
    
    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # 1. Unpack
                target_residual = batch["target_residual"]  # (B, 4, Y, X)
                forecast = batch["input_forecast"]          # (B, 4, Y, X)
                observed = batch["observed_state"]          # (B, 4, Y, X)
                ocean_sst = batch["ocean_sst"]              # (B, 4, Y, X)
                ocean_sss = batch["ocean_sss"]              # (B, 4, Y, X)
                mjo = batch["mjo_conditioning"]             # (B, 2)
                month_onehot = batch["month_onehot"]        # (B, 12)
                
                bs = forecast.shape[0]
                
                # DIAGNOSTIC: Check range periodically
                if global_step % 500 == 0 and accelerator.is_main_process:
                    print(f"\n[Step {global_step}] Input Stats (should be in [-1, 1]):")
                    print(f"  Residual: min={target_residual.min():.2f}, max={target_residual.max():.2f}, mean={target_residual.mean():.2f}")
                    print(f"  Forecast: min={forecast.min():.2f}, max={forecast.max():.2f}, mean={forecast.mean():.2f}")
                    print(f"  SST: min={ocean_sst.min():.2f}, max={ocean_sst.max():.2f}, mean={ocean_sst.mean():.2f}")
                    print(f"  SSS: min={ocean_sss.min():.2f}, max={ocean_sss.max():.2f}, mean={ocean_sss.mean():.2f}")
                
                # 2. Broadcast MJO
                mjo_map = mjo.view(bs, 2, 1, 1).expand(-1, -1, config["image_size"][0], config["image_size"][1])
                
                # 3. Concatenate all conditioning inputs (NO noisy target)
                model_input = torch.cat([forecast, observed, ocean_sst, ocean_sss, mjo_map], dim=1)
                
                # 4. Forward pass — predict residual directly
                pred_residual = model(model_input, month_onehot)
                
                # 5. Loss on residual
                loss = criterion(pred_residual, target_residual)
                
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            train_loss += loss.item()
            
            # Logging
            pred_mean = pred_residual.mean().item()
            pred_std = pred_residual.std().item()
            
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
        
        # Per-week RMSE tracking
        week_mse = [0.0, 0.0, 0.0, 0.0]
        n_val_batches = 0
        
        if accelerator.is_main_process:
            print(f"Validating... (Train Loss: {avg_train_loss:.4f})")
        
        first_batch_data = None
        
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                target_residual = batch["target_residual"]
                target_truth = batch["target_truth"]
                forecast = batch["input_forecast"]
                observed = batch["observed_state"]
                ocean_sst = batch["ocean_sst"]
                ocean_sss = batch["ocean_sss"]
                mjo = batch["mjo_conditioning"]
                month_onehot = batch["month_onehot"]
                
                bs = forecast.shape[0]
                mjo_map = mjo.view(bs, 2, 1, 1).expand(-1, -1, config["image_size"][0], config["image_size"][1])
                
                model_input = torch.cat([forecast, observed, ocean_sst, ocean_sss, mjo_map], dim=1)
                
                # Single forward pass — no DDIM sampling!
                pred_residual = model(model_input, month_onehot)
                
                loss = criterion(pred_residual, target_residual)
                val_loss += loss.item()
                
                # Per-week MSE
                for w in range(4):
                    week_mse[w] += F.mse_loss(pred_residual[:, w], target_residual[:, w]).item()
                n_val_batches += 1

                # Cache first batch for plotting
                if idx == 0 and accelerator.is_main_process:
                    first_batch_data = (
                        target_truth[0:1], forecast[0:1], 
                        pred_residual[0:1].detach()
                    )

        avg_val_loss = val_loss / len(val_dataloader)
        
        if accelerator.is_main_process:
            week_rmse_str = " | ".join([f"LW{w+1}={np.sqrt(week_mse[w]/n_val_batches):.4f}" for w in range(4)])
            print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | RMSE by week: {week_rmse_str}")
            
            accelerator.log({
                "val_loss": avg_val_loss,
                "val_rmse_lw1": np.sqrt(week_mse[0] / n_val_batches),
                "val_rmse_lw2": np.sqrt(week_mse[1] / n_val_batches),
                "val_rmse_lw3": np.sqrt(week_mse[2] / n_val_batches),
                "val_rmse_lw4": np.sqrt(week_mse[3] / n_val_batches),
            }, step=global_step)
            
            is_best = avg_val_loss < best_val_loss
            
            # ------------------------------------------------------------------
            # Validation Plot (Periodic OR Best)
            # ------------------------------------------------------------------
            if (epoch % 5 == 0 or is_best) and first_batch_data is not None:
                target_s, forecast_s, pred_res_s = first_batch_data
                
                # Reconstruct physical precip from predicted residual
                pred_physical = denormalize_residual(pred_res_s[0], forecast_s[0])
                input_raw = denormalize(forecast_s[0]).detach().cpu().numpy()
                target_raw = denormalize(target_s[0]).detach().cpu().numpy()
                pred_raw = pred_physical.detach().cpu().numpy()
                
                if accelerator.is_main_process:
                    print(f"  Pred residual stats: mean={pred_res_s.mean().item():.3f}, "
                          f"std={pred_res_s.std().item():.3f}, "
                          f"min={pred_res_s.min().item():.3f}, "
                          f"max={pred_res_s.max().item():.3f}")
                    print(f"  Pred physical stats: mean={pred_raw.mean():.2f}, "
                          f"max={pred_raw.max():.2f} mm/day")
                
                suffix = "_best" if is_best else ""
                plot_save_path = f"{config['output_dir']}/plots/epoch_{epoch}{suffix}.png"
                plot_comparison(
                    input_raw, target_raw, pred_raw, 
                    plot_save_path,
                    title=f"Epoch {epoch} — UNet+TempAttn Direct {'(Best)' if is_best else ''}"
                )
                if accelerator.is_main_process:
                    print(f"  Validation plot saved to: {plot_save_path}")
            
            # ------------------------------------------------------------------
            # Checkpoint Logic
            # ------------------------------------------------------------------
            if is_best:
                print(f"  ★ New best model at epoch {epoch} (Loss: {avg_val_loss:.4f})")
                best_val_loss = avg_val_loss
                checkpoint_path = f"{config['output_dir']}/best_model_epoch_{epoch}"
                accelerator.save_state(checkpoint_path)
                
                best_checkpoints.append(checkpoint_path)
                if len(best_checkpoints) > 4:
                    old_checkpoint = best_checkpoints.pop(0)
                    import shutil
                    if os.path.exists(old_checkpoint):
                        shutil.rmtree(old_checkpoint, ignore_errors=True)
        
        # Periodic Save for Resuming
        latest_path = f"{config['output_dir']}/latest_checkpoint"
        accelerator.save_state(latest_path)
        if accelerator.is_main_process:
            with open(os.path.join(latest_path, "epoch.json"), "w") as f:
                json.dump({"epoch": epoch, "best_val_loss": best_val_loss}, f)

    if accelerator.is_main_process:
        print("Training finished.")

if __name__ == "__main__":
    train_model()
