import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import glob
import argparse
from tqdm import tqdm
from accelerate import Accelerator

from model_ssg import SpatialSpreadGenerator

class SSGDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "target_*.pt"))
        if not self.files:
            print(f"WARNING: No target files found in {data_dir}")
            
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
    dataset = SSGDataset(args.data_dir)
    if len(dataset) == 0:
        return
        
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
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
        
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                
                preds = torch.argmax(logits, dim=1) # [B, H, W]
                correct_pixels += (preds == y).sum().item()
                total_pixels += y.numel()
                
        val_loss /= len(val_dataset)
        val_acc = (correct_pixels / total_pixels) * 100.0
        
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

if __name__ == "__main__":
    main()
