import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import os

from ml_model.dataset import GeosSubCDataset

def train_loop(config):
    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
    )
    
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("geos_subc_diffusion")

    # Load Dataset
    # Placeholder: We need the ground truth group name to fully initialize this
    train_dataset = GeosSubCDataset(
        forecast_store_path=config.train_data_path,
        obs_group_name=config.obs_group_name # This will need to be provided
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)

    # Initialize Model
    # Since we are predicting 32 lead times * 2 variables = 64 channels (placeholder logic)
    # We might need to adjust channel counts
    model = UNet2DModel(
        sample_size=config.image_size,  # The target image resolution
        in_channels=config.in_channels, # Noisy input channels
        out_channels=config.out_channels, # Predicted noise channels (or data)
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", 
            "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
    )

    # Scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # LR Scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs)
    )

    # Prepare everything with Accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["pixel_values"]
            
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # Predict the noise residual
            # Note: We need to inject conditioning here (Forecast + Initial Obs)
            # For now, this is standard unconditional DDM training skeleton.
            # Condition injection strategy needs to be finalized in model architecture or input concatenation.
            noise_pred = model(noisy_images, timesteps).sample

            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # Validation Loop
        if (epoch + 1) % config.save_model_epochs == 0: # Verify at save frequency
            model.eval()
            val_loss = 0.0
            val_crps = 0.0
            num_val_batches = 0
            
            # Use train_dataloader for now as placeholder for validation split
            # Ideally we split dataset into train/val
            print("Running validation...")
            with torch.no_grad():
                for batch in train_dataloader: 
                    clean_images = batch["pixel_values"]
                    bs = clean_images.shape[0]
                    
                    # Generate samples for CRPS (Ensemble)
                    # For diffusion, we generate from noise
                    n_members = 5 # Small ensemble for validation
                    ensemble_preds = []
                    
                    for _ in range(n_members):
                         # Standard sampling loop (simplified)
                         # Real implementation needs full pipeline or manual scheduler loop
                         # For speed, we might just denoising one step or use few-step scheduler
                         noise = torch.randn_like(clean_images)
                         # This is just a placeholder for actual generation logic
                         # actual generation requires iterating scheduler.step()
                         # We will implement proper generation when model structure is finalized
                         ensemble_preds.append(noise) # Placeholder
                    
                    # Stack: (B, M, C, H, W)
                    ensemble_tensor = torch.stack(ensemble_preds, dim=1)
                    
                    # Compute CRPS (Placeholder)
                    # target: clean_images (B, C, H, W)
                    # crps = crps_ensemble(clean_images, ensemble_tensor)
                    # val_crps += crps.item()
                    num_val_batches += 1
                    if num_val_batches > 5: break # validation on small subset
            
            model.train()

        # Save model checkpoint
        if accelerator.is_main_process and (epoch + 1) % config.save_model_epochs == 0:
             pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
             pipeline.save_pretrained(config.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="dataprocess/geos_subc_2000.zarr")
    parser.add_argument("--obs_group_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="ml_model/output")
    parser.add_argument("--image_size", type=int, default=64) # Placeholder resolution
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    
    # Placeholder channels
    parser.add_argument("--in_channels", type=int, default=3) 
    parser.add_argument("--out_channels", type=int, default=3)

    args = parser.parse_args()
    train_loop(args)
