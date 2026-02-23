import torch

# Approximate TACC dictionary values:
# If GEOS is roughly [0, 40]
g_min = 0.0
g_max = 50.0

# If Residuals bounded roughly [-50, 50]
r_min = -60.0
r_max = 60.0

# 1. Simulate DataLoader Behavior
pure_geos = torch.tensor([5.0]) # Forecast says 5.0
pure_target = torch.tensor([15.0]) # Reality was 15.0

residual_raw = pure_target - pure_geos # Residual is 10.0
# Normalize residual to [-1, 1]
normalized_residual = 2.0 * (torch.clamp(residual_raw, r_min, r_max) - r_min) / (r_max - r_min) - 1.0

# Normalize GEOS to [-1, 1]
normalized_geos = 2.0 * (torch.clamp(pure_geos, g_min, g_max) - g_min) / (g_max - g_min) - 1.0

print(f"Dataset Output Normalized Residual: {normalized_residual.item()}")
print(f"Dataset Output Normalized GEOS: {normalized_geos.item()}")

# 2. Simulate train_diffusion_v4.py Validation Plot Math
fb_target_norm = normalized_residual
fx_geos_norm = normalized_geos

denorm_residual_raw = ((fb_target_norm + 1.0) / 2.0) * (r_max - r_min) + r_min
fx_geos_raw = ((fx_geos_norm + 1.0) / 2.0) * (g_max - g_min) + g_min

print(f"Script Demaps Residual back to: {denorm_residual_raw.item()} (Should be 10.0)")
print(f"Script Demaps GEOS back to: {fx_geos_raw.item()} (Should be 5.0)")

true_target_raw = fx_geos_raw + denorm_residual_raw
true_target_precip = torch.clamp(true_target_raw, min=0.0)

print(f"Validation Outputs True Target Precip: {true_target_precip.item()} (Should be 15.0)")
