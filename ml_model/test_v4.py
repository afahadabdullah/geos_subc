import torch
from diffusion_v4 import DiffusionModelV4, CustomDiffusionScheduler

def test_v4():
    print("Initializing Diffusion V4...")
    device = "cpu"
    
    # 1. Model setup
    model = DiffusionModelV4(in_channels=49, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params:,}")
    
    # 2. Scheduler setup
    scheduler = CustomDiffusionScheduler(num_timesteps=1000, device=device)
    
    # 3. Dummy Inputs (1 Batch)
    # Target shape is [B, 49, H, W] - 1 noisy GPCP target + 48 conditionals
    # (H=181, W=360) based on typical GEOS structure output
    B, C, H, W = 1, 49, 181, 360
    
    print(f"Simulating Batch [B={B}, C={C}, H={H}, W={W}]...")
    
    x_noisy = torch.randn((B, 1, H, W)).to(device)
    x_cond = torch.randn((B, 48, H, W)).to(device)
    
    # Random Timesteps
    t = torch.randint(0, 1000, (B,)).long()
    
    # 4. Forward Pass Prediction
    try:
        noise_pred = model(x_noisy, x_cond, t)
        print(f"Forward Pass Success! Output shape: {noise_pred.shape}")
        assert noise_pred.shape == (B, 1, H, W)
    except Exception as e:
        print(f"Forward Pass FAILED: {e}")
        return False
        
    return True

if __name__ == "__main__":
    test_v4()
