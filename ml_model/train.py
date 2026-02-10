import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
import os
import numpy as np
import pandas as pd
import tqdm

try:
    from ml_model.dataset import GeosSubCDataset
    from ml_model.model import ConditionalUNet, GaussianDiffusion
    from ml_model.utils import crps_ensemble
except ImportError:
    from dataset import GeosSubCDataset
    from model import ConditionalUNet, GaussianDiffusion
    from utils import crps_ensemble

def train_model():
    # --------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------
    config = {
        "train_years": (1999, 2014),
        "val_years": (2015, 2016),
        "batch_size": 4, # Reduced from 16 due to OOM
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
    
    # --------------------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------------------
    train_dataset = GeosSubCDataset(
        data_root=config["data_root"],
        start_year=config["train_years"][0], 
        end_year=config["train_years"][1],
        mjo_file="mjo_processed.csv"
    )
    
    val_dataset = GeosSubCDataset(
        data_root=config["data_root"],
        start_year=config["val_years"][0], 
        end_year=config["val_years"][1],
        mjo_file="mjo_processed.csv"
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
    
    # --------------------------------------------------------------------------
    # Model Setup (CMDE Architecture)
    # --------------------------------------------------------------------------
    # Inputs:
    # 1. Noisy Residual (4 Channels for 4 weeks)
    # 2. Condition:
    #    - Noisy Forecast (4 Channels)
    #    - MJO Features (Broadcasted -> 2 Channels)
    # Total Input Channels to UNet = 4 + 4 + 2 = 10
    
    in_channels = 10 
    out_channels = 4 # Predicted Noise for Residual
    
    model = ConditionalUNet(
        in_channels=in_channels, 
        out_channels=out_channels, 
        base_filters=64,
        num_months=12
    )
    
    # Custom Gaussian Diffusion
    diffusion = GaussianDiffusion(timesteps=1000, device=accelerator.device) # Will update device in loop
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=500, 
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
    diffusion.alpha_hats = diffusion.alpha_hats.to(accelerator.device)
    diffusion.sqrt_alpha_hats = diffusion.sqrt_alpha_hats.to(accelerator.device)
    diffusion.sqrt_one_minus_alpha_hats = diffusion.sqrt_one_minus_alpha_hats.to(accelerator.device)

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    if accelerator.is_main_process:
        print("Starting training with CMDE architecture...")
    
    global_step = 0
    
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # 1. Unpack & Normalize
                # Scale Factor (50mm -> 1.0) approx for raw mm/day
                scale = 0.02
                
                target_truth = batch["target_truth"] * scale  # (B, 4, Y, X)
                forecast = batch["input_forecast"] * scale    # (B, 4, Y, X)
                mjo = batch["mjo_conditioning"]               # (B, 2)
                s_date_str = batch["S"] # List of strings
                
                # Derive Month
                # Pandify
                dates = pd.to_datetime(s_date_str)
                months = torch.tensor(dates.month, device=accelerator.device).long()
                
                # 2. Add Noise to Target (Raw Truth, NOT Residual)
                # User requested raw target instead of residual.
                target = target_truth
                
                bs = target.shape[0]
                timesteps = diffusion.sample_timesteps(bs)
                noisy_target, noise = diffusion.add_noise(target, timesteps)
                
                # 4. CMDE Logic: Add Reduced Noise to Forecast Condition
                # Get sqrt(1-alpha_bar) for current t
                sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alpha_hats[timesteps].view(-1, 1, 1, 1)
                
                # Noise for condition (scaled by cmde_ratio)
                cond_noise = torch.randn_like(forecast)
                noisy_forecast = forecast + (config["cmde_ratio"] * sqrt_one_minus_alpha * cond_noise)
                
                # 5. Broadcast MJO
                mjo_map = mjo.view(bs, 2, 1, 1).expand(-1, -1, config["image_size"][0], config["image_size"][1])
                
                # 6. Cat Inputs
                # [NoisyTarget (4), NoisyForecast (4), MJO (2)]
                model_input = torch.cat([noisy_target, noisy_forecast, mjo_map], dim=1)
                
                # 7. Predict Noise
                noise_pred = model(model_input, timesteps, months)
                
                loss = F.mse_loss(noise_pred, noise)
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            train_loss += loss.item()
            global_step += 1
            
            if step % 50 == 0 and accelerator.is_main_process:
               print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")
               
        # Validation Logic (Simple MSE on noise for now)
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    # Scale Factor (50mm -> 1.0) approx for raw mm/day
                    scale = 0.02

                    target_truth = batch["target_truth"] * scale  # (B, 4, Y, X)
                    forecast = batch["input_forecast"] * scale    # (B, 4, Y, X)
                    mjo = batch["mjo_conditioning"]               # (B, 2)
                    s_date_str = batch["S"] # List of strings
                    
                    # Generate Sample (Reverse Diffusion)
                    # Derive Month
                    dates = pd.to_datetime(s_date_str)
                    months = torch.tensor(dates.month, device=accelerator.device).long()
                    
                    # Validation on Raw Target
                    target = target_truth
                    bs = target.shape[0]
                    timesteps = diffusion.sample_timesteps(bs)
                    noisy_target, noise = diffusion.add_noise(target, timesteps)
                    
                    sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alpha_hats[timesteps].view(-1, 1, 1, 1)
                    cond_noise = torch.randn_like(forecast)
                    noisy_forecast = forecast + (config["cmde_ratio"] * sqrt_one_minus_alpha * cond_noise)
                    mjo_map = mjo.view(bs, 2, 1, 1).expand(-1, -1, config["image_size"][0], config["image_size"][1])
                    model_input = torch.cat([noisy_target, noisy_forecast, mjo_map], dim=1)
                    
                    noise_pred = model(model_input, timesteps, months)
                    loss = F.mse_loss(noise_pred, noise)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            if accelerator.is_main_process:
                print(f"Epoch {epoch} Val Loss (MSE): {avg_val_loss:.4f}")
        
        # Save Model
        if epoch % 10 == 0:
            accelerator.save_state(f"{config['output_dir']}/checkpoint-{epoch}")

    if accelerator.is_main_process:
        print("Training finished.")

if __name__ == "__main__":
    train_model()
