import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import os
import numpy as np
import tqdm
try:
    from ml_model.dataset import GeosSubCDataset
    from ml_model.utils import crps_ensemble
except ImportError:
    from dataset import GeosSubCDataset
    from utils import crps_ensemble

def train_model():
    # --------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------
    config = {
        "train_years": (1999, 2014),
        "val_years": (2015, 2016),
        "batch_size": 4, 
        "num_epochs": 100,
        "lr": 1e-4,
        "image_size": (181, 360),
        "data_root": "dataprocess",
        "output_dir": "ml_output",
        "gradient_accumulation_steps": 2,
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
    # Dataset & Dataloaders
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
    
    # ... (Rest of setup)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)
    
    # Model Setup (Same as before)
    in_channels = 4 + 4 + 2 # Noisy Truth (4) + Condition Forecast (4) + MJO (2)
    out_channels = 4       # Pred Noise
    
    model = UNet2DModel(
        sample_size=config["image_size"],
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"
        ),
    )
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
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
    
    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    if accelerator.is_main_process:
        print("Starting training...")
    
    global_step = 0
    
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # 1. Unpack
                clean_images = batch["target_truth"]  # (B, 4, Y, X)
                forecast = batch["input_forecast"]    # (B, 4, Y, X)
                mjo = batch["mjo_conditioning"]       # (B, 2)
                
                # Scale Factor (50mm -> 1.0)
                scale_factor = 0.02 
                clean_images = clean_images * scale_factor
                forecast = forecast * scale_factor
                
                # 2. Add Noise
                bs = clean_images.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
                noise = torch.randn_like(clean_images)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                
                # 3. Construct Input
                # Broadcast MJO
                mjo_map = mjo.view(bs, 2, 1, 1).expand(-1, -1, config["image_size"][0], config["image_size"][1])
                net_input = torch.cat([noisy_images, forecast, mjo_map], dim=1) # (B, 10, Y, X)
                
                # 4. Predict Noise
                noise_pred = model(net_input, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            train_loss += loss.item()
            global_step += 1
            
            if step % 100 == 0 and accelerator.is_main_process:
               print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")
               
        # Validation Loop (CRPS)
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            # For diffusion, CRPS requires generating an ensemble.
            # This is slow. We might just check MSE on noise first for validation.
            # Or generate 1 sample.
            
            with torch.no_read_grad():
                for batch in val_dataloader:
                    clean_images = batch["target_truth"] * 0.02
                    forecast = batch["input_forecast"] * 0.02
                    mjo = batch["mjo_conditioning"]
                    bs = clean_images.shape[0]
                    
                    # Generate Sample (Reverse Diffusion)
                    # We start from random noise and denoise conditioned on Forecast+MJO
                    # This is just one sample (Member 1).
                    # To calculate proper CRPS, we need multiple members.
                    # For quick validation, let's just do MSE on noise (Validation Loss).
                    
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
                    noise = torch.randn_like(clean_images)
                    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                    mjo_map = mjo.view(bs, 2, 1, 1).expand(-1, -1, config["image_size"][0], config["image_size"][1])
                    net_input = torch.cat([noisy_images, forecast, mjo_map], dim=1)
                    
                    noise_pred = model(net_input, timesteps).sample
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
