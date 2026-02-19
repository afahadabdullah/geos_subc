"""
PatchGAN Discriminator for Precipitation Sharpness
===================================================

Conditional PatchGAN discriminator that classifies local spatial patches
of precipitation as real (GPCP) or fake (UNet prediction).

Input:  (B, C_cond + 1, H, W)  where C_cond = conditioning channels (GEOS)
        and 1 = precipitation channel (real or fake)
Output: (B, 1, H/8, W/8)  per-patch real/fake logits

Uses spectral normalization on all conv layers for stable GAN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator conditioned on GEOS ensemble mean.
    
    For each lead week, receives:
      - Precipitation map (1 channel): either real GPCP or fake UNet E[rain]
      - Condition (4 channels): GEOS ensemble mean for all 4 leads
    
    Total input: 5 channels per forward pass.
    Processes each lead week independently.
    
    Architecture: 4-layer strided convolutions → ~70×70 receptive field
    Output: (B, 1, H/8, W/8) real/fake patch scores
    """
    
    def __init__(self, in_channels=5, ndf=64):
        """
        Args:
            in_channels: 1 (precip) + 4 (GEOS condition) = 5
            ndf: Base number of discriminator filters
        """
        super().__init__()
        
        # Layer 1: no normalization (standard PatchGAN practice)
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 2
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 3
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Layer 4: stride=1 for larger receptive field
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1)),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final: 1-channel output (real/fake logit per patch)
        self.final = spectral_norm(nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1))
    
    def forward(self, precip, condition):
        """
        Args:
            precip:    (B, 1, H, W) - single lead's precipitation (real or fake)
            condition: (B, 4, H, W) - GEOS ensemble mean for all 4 leads
            
        Returns:
            (B, 1, H', W') - per-patch real/fake logits
        """
        x = torch.cat([precip, condition], dim=1)  # (B, 5, H, W)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        return x
    
    def forward_all_leads(self, precip_4leads, condition):
        """
        Process all 4 lead weeks and return combined adversarial score.
        
        Args:
            precip_4leads: (B, 4, H, W) - all 4 leads of precipitation
            condition:     (B, 4, H, W) - GEOS ensemble mean
            
        Returns:
            List of 4 tensors, each (B, 1, H', W') per-patch logits
        """
        outputs = []
        for lead in range(4):
            precip_lead = precip_4leads[:, lead:lead+1, :, :]  # (B, 1, H, W)
            out = self.forward(precip_lead, condition)
            outputs.append(out)
        return outputs


def discriminator_loss(disc_real_outputs, disc_fake_outputs):
    """
    LSGAN discriminator loss: MSE toward 1 for real, 0 for fake.
    More stable than vanilla GAN BCE loss.
    
    Args:
        disc_real_outputs: list of 4 (B, 1, H', W') from real GPCP
        disc_fake_outputs: list of 4 (B, 1, H', W') from fake E[rain]
    
    Returns:
        Scalar discriminator loss
    """
    loss = 0.0
    for real_out, fake_out in zip(disc_real_outputs, disc_fake_outputs):
        loss += torch.mean((real_out - 1.0) ** 2)  # real → 1
        loss += torch.mean(fake_out ** 2)            # fake → 0
    return loss / (2 * len(disc_real_outputs))


def generator_adversarial_loss(disc_fake_outputs):
    """
    LSGAN generator loss: MSE toward 1 for fake (fool discriminator).
    
    Args:
        disc_fake_outputs: list of 4 (B, 1, H', W') from fake E[rain]
    
    Returns:
        Scalar generator adversarial loss
    """
    loss = 0.0
    for fake_out in disc_fake_outputs:
        loss += torch.mean((fake_out - 1.0) ** 2)  # fake → 1 (fool disc)
    return loss / len(disc_fake_outputs)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing PatchGANDiscriminator on {device}...")
    
    disc = PatchGANDiscriminator(in_channels=5, ndf=64).to(device)
    n_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"Discriminator parameters: {n_params:,}")
    
    # Test forward
    precip = torch.randn(2, 4, 181, 360).to(device)
    condition = torch.randn(2, 4, 181, 360).to(device)
    outputs = disc.forward_all_leads(precip, condition)
    print(f"Output shapes: {[o.shape for o in outputs]}")
