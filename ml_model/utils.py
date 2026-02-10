import torch
import numpy as np

def crps_ensemble(observations, forecasts):
    """
    Compute Continuous Ranked Probability Score (CRPS) for an ensemble forecast.
    
    Args:
        observations (torch.Tensor): Shape (Batch, ...) - The ground truth.
        forecasts (torch.Tensor): Shape (Batch, Members, ...) - The ensemble predictions.
    
    Returns:
        torch.Tensor: Scalar CRPS value (averaged over batch).
    """
    # CRPS_ensemble = E_x|y| - 0.5 * E_x,x'|x - x'|
    # Where x, x' are independent draws from the forecast distribution (members)
    # y is the observation
    
    # forecasts: (B, M, ...)
    # observations: (B, ...) -> unsqueeze to (B, 1, ...)
    obs = observations.unsqueeze(1)
    
    # Term 1: Mean absolute error of members wrt observation
    # Shape: (B, M, ...) -> Mean over M -> (B, ...)
    term_1 = torch.abs(forecasts - obs).mean(dim=1)
    
    # Term 2: Mean absolute difference between ensemble members
    # This is O(M^2) which is fine for small ensembles (e.g. M=10-50)
    # Expand to (B, M, 1, ...) and (B, 1, M, ...)
    f1 = forecasts.unsqueeze(2)
    f2 = forecasts.unsqueeze(1)
    
    # Shape: (B, M, M, ...) -> Mean over M, M -> (B, ...)
    term_2 = torch.abs(f1 - f2).mean(dim=(1, 2))
    
    # CRPS = Term 1 - 0.5 * Term 2
    crps = term_1 - 0.5 * term_2
    
    return crps.mean()

def compute_mse(pred, target):
    """
    Standard MSE for validation.
    """
    return torch.nn.functional.mse_loss(pred, target)

# Placeholder for MJO metric
def compute_mjo_rmse(pred_rmm, target_rmm):
    return torch.sqrt(torch.mean((pred_rmm - target_rmm)**2))
