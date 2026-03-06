import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialSpreadGenerator(nn.Module):
    """
    Spatial Spread Generator (SSG)
    A lightweight CNN that predicts spatial mixing weights for 4 noise modes
    (Pure Random, MJO, NAO, ENSO) based on atmospheric indices and time.
    """
    def __init__(self, in_channels=7, out_channels=4, hidden_dim=64):
        super().__init__()
        
        # We use a very shallow CNN. Since inputs are mostly spatially uniform scalars,
        # the convolutions act like a dense layer applied per-pixel, but preserving
        # the ability to accept spatial fields (like Z500 or topography) in the future.
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            
            nn.Conv2d(hidden_dim // 2, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        """
        x: [B, C, H, W] - Conditioning tensor
           Channels (in order):
           0. fsin_month
           1. fcos_month
           2. lead_time (normalized)
           3. mjo_rmm1
           4. mjo_rmm2
           5. nao_val
           6. enso_val
           
        returns:
           logits: [B, 4, H, W] - Unnormalized logits for the 4 noise strategies.
        """
        return self.net(x)

    def get_blending_weights(self, x):
        """
        Returns the Softmax probabilities.
        weights: [B, 4, H, W] summing to 1.0 along dim 1.
        """
        logits = self(x)
        return F.softmax(logits, dim=1)
