import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import os
import argparse
import yaml # pyyaml
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml_model.dataset_hybrid import S2SHybridDataset
from ml_model.model_committee import CommitteeModel
from ml_model.loss_committee import ZeroInflatedGammaLoss

def train(config_path="config.yaml"): # Or hardcoded defaults
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    
    # Config Defaults
    BATCH_SIZE = 4
    LR = 1e-4
    EPOCHS = 20
    
    # 1. Dataset
    print("Initializing Dataset...")
    # Adjust root path as needed. Assuming running from project root.
    dataset = S2SHybridDataset(data_root="dataprocess", start_year=1999, end_year=2015, preload=False)
    # Val set? Using last year 2015-2016 from split?
    # Or simplified: Train 1999-2015.
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    # 2. Model
    # Obs channels: SST (4) + SSS (4) + SM (4) + PrevGPCP (4) = 16
    # GEOS channels: Pr -> 1
    model = CommitteeModel(channels_obs=16, channels_geos=1, num_members=4)
    
    # 3. Loss & Optimizer
    criterion = ZeroInflatedGammaLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Prepare
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    
    # Training Loop
    if accelerator.is_main_process:
        print(f"Starting Training on {device}...")
        
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(loader, disable=not accelerator.is_main_process)
        for batch in pbar:
            # Inputs
            x_obs = batch['x_obs']   # (B, 3, H, W)
            x_geos = batch['x_geos'] # (B, 4, 1, L, H, W)
            y_target = batch['y_target'] # (B, L, H, W)
            
            # Forward
            # Returns (B, M, L, H, W) tuples for p, alpha, beta
            p_stack, alpha_stack, beta_stack = model(x_obs, x_geos)
            
            # Loss Calculation (Committee Average)
            # p_stack: (B, 4, L, H, W)
            loss = 0.0
            M = p_stack.shape[1]
            
            # Broadcast target to members? Or strict per-member loss?
            # Target is same for all members.
            for m in range(M):
                lm = criterion(p_stack[:,m], alpha_stack[:,m], beta_stack[:,m], y_target)
                loss += lm
            
            loss = loss / M
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            
        avg_loss = epoch_loss / len(loader)
        if accelerator.is_main_process:
            print(f"Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f}")
            
            # Save Checkpoint
            os.makedirs("checkpoints_committee", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints_committee/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    train()
