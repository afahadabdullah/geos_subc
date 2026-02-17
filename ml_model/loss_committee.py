import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroInflatedGammaLoss(nn.Module):
    """
    Zero-Inflated Gamma Loss (Negative Log-Likelihood).
    
    Likelihood:
      P(Y=0) = 1 - p
      P(Y=y | y>0) = p * Gamma(y; alpha, beta)
      
    Log-Likelihood:
      If y=0: log(1-p)
      If y>0: log(p) + log Gamma(y; alpha, beta)
      
    Loss = -Log-Likelihood
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, p, alpha, beta, target):
        """
        p: (B, L, H, W) - Probability of rain
        alpha: (B, L, H, W) - Shape (>0)
        beta: (B, L, H, W) - Rate (>0)
        target: (B, L, H, W) - Observed Precip (>=0)
        """
        # Ensure positive params
        p = torch.clamp(p, min=self.eps, max=1.0 - self.eps)
        alpha = torch.clamp(alpha, min=self.eps)
        beta = torch.clamp(beta, min=self.eps)
        target = torch.clamp(target, min=0.0) # Ensure no negative precip
        
        # Masks
        rain_mask = (target > self.eps)
        no_rain_mask = (target <= self.eps)
        
        loss = torch.zeros_like(target)
        
        # 1. No Rain Case (y=0)
        # Loss = -log(1-p)
        if no_rain_mask.any():
            loss[no_rain_mask] = -torch.log(1.0 - p[no_rain_mask])
            
        # 2. Rain Case (y>0)
        # Loss = -log(p) - log Gamma(y)
        # Log Gamma(y) = alpha*log(beta) - lgamma(alpha) + (alpha-1)*log(y) - beta*y
        if rain_mask.any():
             y = target[rain_mask]
             a = alpha[rain_mask]
             b = beta[rain_mask]
             p_rain = p[rain_mask]
             
             log_gamma_pdf = (a * torch.log(b)) - torch.lgamma(a) + ((a - 1.0) * torch.log(y)) - (b * y)
             
             # NLL
             loss[rain_mask] = - (torch.log(p_rain) + log_gamma_pdf)
             
        return loss.mean()
