import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import glob
import argparse
from tqdm import tqdm
from accelerate import Accelerator
import matplotlib.pyplot as plt
import numpy as np

from model_ssg import SpatialSpreadGenerator

class SSGDataset(Dataset):
    def __init__(self, data_dir, years):
        self.files = []
        for y in years:
            self.files.extend(glob.glob(os.path.join(data_dir, f"target_y{y}_*.pt")))
            
        if not self.files:
            print(f"WARNING: No target files found for years {years} in {data_dir}")
            
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        path = self.files[idx]
        data = torch.load(path, map_location="cpu")
        
        # Build 7-channel input tensor [7, H, W]
        H, W = data['winner_map'].shape
        x = torch.zeros((7, H, W), dtype=torch.float32)
        
        x[0] = data['fsin_month']
        x[1] = data['fcos_month']
        x[2] = data['lead'] / 4.0  # Normalize lead [0, 1]
        x[3] = data['mjo_rmm1']
        x[4] = data['mjo_rmm2']
        x[5] = data['nao_val']
        x[6] = data['enso_val']
        
        y = data['winner_map'].long() # [H, W] Integer targets {0,1,2,3}
        
        return x, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/scratch/11353/afahad/geossub/geos_subc/dataprocess/noise")
    parser.add_argument("--out_dir", type=str, default="ml_output_ssg")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    os.makedirs(args.out_dir, exist_ok=True)
    
    # ── Dataset ──
    # User requested 1999-2020 for Training, 2021 for Validation
    train_years = list(range(1999, 2021))
    val_years = [2021]
    
    train_dataset = SSGDataset(args.data_dir, years=train_years)
    val_dataset = SSGDataset(args.data_dir, years=val_years)
    
    if len(train_dataset) == 0:
        print("ERROR: Train dataset empty. Run target generation first!")
        return
        
    if len(val_dataset) == 0:
        print("WARNING: Val dataset empty. Proceeding without validation tracking.")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # ── Model & Optim ──
    model = SpatialSpreadGenerator(in_channels=7, out_channels=4, hidden_dim=64)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    best_val_loss = float('inf')
    history_train = []
    history_val = []
    history_acc = []
    
    print(f"Starting SSG Training: {len(train_dataset)} Train | {len(val_dataset)} Val")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x) # [B, 4, H, W]
            
            loss = criterion(logits, y)
            accelerator.backward(loss)
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        
        first_batch = None
        first_logits = None
        
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                
                preds = torch.argmax(logits, dim=1) # [B, H, W]
                correct_pixels += (preds == y).sum().item()
                total_pixels += y.numel()
                
                if i == 0:
                    first_batch = (x[0].clone(), y[0].clone())
                    first_logits = logits[0].clone()
                
        val_loss /= len(val_dataset)
        val_acc = (correct_pixels / total_pixels) * 100.0
        
        history_train.append(train_loss)
        history_val.append(val_loss)
        history_acc.append(val_acc)
        
        if accelerator.is_main_process:
            print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.out_dir, "BEST_ssg.pt")
                unwrapped = accelerator.unwrap_model(model)
                torch.save({
                    'model_state_dict': unwrapped.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, save_path)
                print(f"  --> Saved new best model to {save_path}")
                
            # Plot Diagnostics every 5 epochs or last epoch
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                if first_batch is not None and first_logits is not None:
                    probs = torch.softmax(first_logits, dim=0).cpu().numpy()
                    target = first_batch[1].cpu().numpy()
                    pred = torch.argmax(first_logits, dim=0).cpu().numpy()
                    
                    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                    
                    # Colors: 0=Rand (Gray), 1=MJO (Blue), 2=NAO (Orange), 3=ENSO (Green)
                    from matplotlib.colors import ListedColormap
                    cmap = ListedColormap(['#AAAAAA', '#1f77b4', '#ff7f0e', '#2ca02c'])
                    
                    axes[0,0].imshow(target, cmap=cmap, vmin=0, vmax=3)
                    axes[0,0].set_title("True Winner Map")
                    
                    axes[0,1].imshow(pred, cmap=cmap, vmin=0, vmax=3)
                    axes[0,1].set_title("Predicted Winner Map")
                    
                    axes[0,2].imshow(probs[0], cmap='Greys', vmin=0, vmax=1)
                    axes[0,2].set_title(f"Rand Weight [{probs[0].mean():.2f}]")
                    
                    axes[1,0].imshow(probs[1], cmap='Blues', vmin=0, vmax=1)
                    axes[1,0].set_title(f"MJO Weight [{probs[1].mean():.2f}]")
                    
                    axes[1,1].imshow(probs[2], cmap='Oranges', vmin=0, vmax=1)
                    axes[1,1].set_title(f"NAO Weight [{probs[2].mean():.2f}]")
                    
                    axes[1,2].imshow(probs[3], cmap='Greens', vmin=0, vmax=1)
                    axes[1,2].set_title(f"ENSO Weight [{probs[3].mean():.2f}]")
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.out_dir, f"diagnostic_ep{epoch:03d}.png"))
                    plt.close()
                    
    if accelerator.is_main_process:
        # Final Loss Curve
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history_train, label='Train Loss')
        plt.plot(history_val, label='Val Loss')
        plt.legend()
        plt.title('Loss Curve')
        
        plt.subplot(1, 2, 2)
        plt.plot(history_acc, label='Val Accuracy', color='green')
        plt.legend()
        plt.title('Validation Accuracy (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "training_curve.png"))
        plt.close()

if __name__ == "__main__":
    main()
