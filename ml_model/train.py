import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import os
import numpy as np
import tqdm
from dataset import GeosSubCDataset
from utils import crps_ensemble

def train_model():
    # --------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------
    config = {
        "train_years": (1999, 2014), # e.g.
        "val_years": (2015, 2016),
        "batch_size": 4, # Small batch for 2D time sequences
        "num_epochs": 100,
        "lr": 1e-4,
        "image_size": (181, 360), # (Lat, Lon)
        "leads": 4, # 4 weeks
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
        data_root="dataprocess",
        start_year=config["train_years"][0], 
        end_year=config["train_years"][1],
        mjo_file="mjo_processed.csv"
    )
    
    val_dataset = GeosSubCDataset(
        data_root="dataprocess",
        start_year=config["val_years"][0], 
        end_year=config["val_years"][1],
        mjo_file="mjo_processed.csv"
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)
    
    # --------------------------------------------------------------------------
    # Model & Scheduler
    # --------------------------------------------------------------------------
    # We treat Leads (4) as Channels? OR use 3D U-Net?
    # Simple approach: Channel = Lead * Variables.
    # Here we have 1 variable (Precip) for now. Dataset loads only 'pr' into input/target.
    # Dataset returns (4, Y, X). So 4 channels.
    
    # Conditioning: 
    # MJO (2) + Forecast (4, Y, X).
    # Standard UNet2DModel in diffusers is unconditional or class-conditional.
    # For Modulation by MJO vector, we can use 'class_labels' input if we map vector to embedding,
    # OR we can just concatenate MJO as extra channels (broadcasted).
    # Let's try concat approach in forward pass wrapper or custom model.
    # BUT, to use standard diffusers UNet2DModel, we can use `class_embed_type="identity"`?
    # No, that expects integer class labels.
    
    # Simplest for now: Train a residual correction model?
    # Diffusion target: The ERROR (Truth - Forecast)?
    # Or generating Truth conditioned on Forecast?
    # Let's say we generate Truth.
    # Input channels to UNet = Noisy Truth (4) + Condition Forecast (4). Total 8 channels.
    # MJO (2) can be injected to Time Embedding or added as global channels (broadcast).
    # Let's add MJO as broadcasted channels. 2 more channels. Total 10 channels.
    
    in_channels = 4 + 4 + 2 # Noisy Truth + Forecast + MJO
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
    
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # 1. Unpack
            clean_images = batch["target_truth"]  # (B, 4, Y, X)
            forecast = batch["input_forecast"]    # (B, 4, Y, X)
            mjo = batch["mjo_conditioning"]       # (B, 2)
            
            # Normalize? (Assume data is raw mm/day). 
            # Diffusion works best with [-1, 1] or Gaussian.
            # Simple log transform or standardization might be needed.
            # For this MVP, let's proceed with raw (maybe scale by 1/50?).
            scale_factor = 0.02 # 50mm/day -> 1.0
            clean_images = clean_images * scale_factor
            forecast = forecast * scale_factor
            
            # 2. Add Noise
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
            noise = torch.randn_like(clean_images)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # 3. Construct Input
            # Concatenate Noisy Target + Forecast + MJO (Broadcast)
            mjo_map = mjo.view(bs, 2, 1, 1).expand(-1, -1, config["image_size"][0], config["image_size"][1])
            net_input = torch.cat([noisy_images, forecast, mjo_map], dim=1) # (B, 10, Y, X)
            
            # 4. Predict Noise
            with accelerator.accumulate(model):
                noise_pred = model(net_input, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            train_loss += loss.item()
            
            if step % 100 == 0:
               accelerator.print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")
               
        # Validation
        avg_train_loss = train_loss / len(train_dataloader)
        accelerator.print(f"Epoch {epoch} Average Train Loss: {avg_train_loss:.4f}")
        
        # Save Model
        if epoch % 10 == 0:
            accelerator.save_state(f"{config['output_dir']}/checkpoint-{epoch}")

    accelerator.print("Training finished.")

if __name__ == "__main__":
    train_model()
