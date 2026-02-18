import torch
import numpy as np

def get_area_weights(lats):
    weights = np.cos(np.deg2rad(lats))
    weights = weights / weights.mean()
    return torch.from_numpy(weights).float().view(1, 1, -1, 1)

def test_weighted_loss():
    H, W = 181, 360
    lats = np.linspace(-90, 90, H)
    weights = get_area_weights(lats)
    
    # Test 1: Uniform error should result in loss = error^2
    error = 2.0
    noise = torch.zeros((1, 1, H, W))
    noise_pred = torch.full((1, 1, H, W), error)
    
    weighted_loss = (weights * (noise_pred - noise)**2).mean().item()
    print(f"Uniform Error (2.0): Loss = {weighted_loss:.4f} (Expected: 4.0)")
    
    # Test 2: Error only in poles vs Error only in tropics
    # Poles: indices near 0 and H-1
    # Tropics: indices near H/2
    
    # Pole error
    noise_pred_pole = torch.zeros((1, 1, H, W))
    noise_pred_pole[:, :, 0, :] = 10.0 # North Pole
    noise_pred_pole[:, :, -1, :] = 10.0 # South Pole
    loss_pole = (weights * (noise_pred_pole - noise)**2).mean().item()
    
    # Tropic error
    noise_pred_tropic = torch.zeros((1, 1, H, W))
    noise_pred_tropic[:, :, H//2, :] = 10.0 # Equator
    # We add two lines to match the poles count
    noise_pred_tropic[:, :, H//2 + 1, :] = 10.0 
    loss_tropic = (weights * (noise_pred_tropic - noise)**2).mean().item()
    
    print(f"Pole Error (10.0): Loss = {loss_pole:.6f}")
    print(f"Tropic Error (10.0): Loss = {loss_tropic:.6f}")
    
    if loss_tropic > loss_pole:
        print("Success: Tropic error is weighted more heavily than Pole error.")
    else:
        print("Failure: Weights not applied correctly.")

if __name__ == "__main__":
    test_weighted_loss()
