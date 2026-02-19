"""
UNet-Based Precipitation Prediction with Zero-Inflated Log-Normal (ZILN) Distribution

Architecture: TemporalAttentionUNet from model_unet.py
Output: 12 channels (3 ZILN params x 4 lead weeks) -> p, mu, sigma per grid cell
Loss: CRPS (Continuous Ranked Probability Score) for ZILN distribution, area-weighted
Target: Raw GPCP precipitation (mm/day), no normalization

The ZILN parameterization handles precipitation's extreme dynamic range:
  - p (rain probability):  sigmoid(raw_p)  -> [0, 1]
  - mu (log-space mean):   raw_mu          -> unbounded
  - sigma (log-space std): softplus(raw_s) + 1e-4 -> positive

Expected Value: E[Rain] = p * exp(mu + sigma^2/2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import os
import argparse
import yaml
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ml_model.dataset_hybrid import S2SHybridDataset
from ml_model.model_unet import TemporalAttentionUNet
from ml_model.model_discriminator import PatchGANDiscriminator, discriminator_loss, generator_adversarial_loss


# ==============================================================================
# ZILN PARAMETERIZATION & CRPS LOSS
# ==============================================================================

def parameterize_ziln(raw_output):
    """
    Convert raw UNet output (B, 3*leads, H, W) to ZILN parameters.
    
    Args:
        raw_output: (B, 3*leads, H, W) - Raw UNet output
        
    Returns:
        p:     (B, leads, H, W) - Rain probability [0, 1]
        mu:    (B, leads, H, W) - Log-space mean (unbounded)
        sigma: (B, leads, H, W) - Log-space std (positive)
    """
    B, C, H, W = raw_output.shape
    n_leads = C // 3
    
    # Reshape: (B, C, H, W) -> (B, n_leads, 3, H, W)
    params = raw_output.view(B, n_leads, 3, H, W)
    
    raw_p = params[:, :, 0, :, :]     
    raw_mu = params[:, :, 1, :, :]    
    raw_sigma = params[:, :, 2, :, :] 
    
    p = torch.sigmoid(raw_p)
    mu = raw_mu.clamp(min=-10.0, max=10.0)  # Moderate clamping for stability
    sigma = F.softplus(raw_sigma).clamp(max=3.0) + 1e-4  # Prevent explosion in exp()
    
    return p, mu, sigma


def ziln_expected_value(p, mu, sigma):
    """
    Compute E[Rain] = p * exp(mu + sigma^2 / 2)
    
    This is the mean of the ZILN distribution, used for deterministic validation.
    """
    return p * torch.exp(mu + 0.5 * sigma**2)


def ziln_sample(p, mu, sigma):
    """
    Draw a reparameterized sample from the Zero-Inflated Log-Normal.
    
    Uses the reparameterization trick so gradients flow through p, mu, sigma.
    Produces SHARP fields (unlike E[rain] which is inherently smooth).
    Used as discriminator input for meaningful adversarial training.
    """
    z = torch.randn_like(mu)
    rain_amount = torch.exp((mu + sigma * z).clamp(max=10.0))  # prevent overflow
    return p * rain_amount  # soft zero-inflation (differentiable)


def crps_ziln_loss(p, mu, sigma, target, lead_indices=None, area_weights=None):
    """
    Args:
        lead_indices: (B,) int tensor [0-3] indicating which lead each sample corresponds to.
                      If None, assumes multi-lead batch (legacy/validation).
    """
    eps = 1e-6
    y = target.clamp(min=0.0)
    
    Phi = lambda x: 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    
    # --- Log-Normal CRPS component ---
    y_safe = y.clamp(min=eps)
    z = (torch.log(y_safe) - mu) / sigma
    E_LN = torch.exp(mu + 0.5 * sigma**2)
    
    crps_ln_wet = y_safe * (2 * Phi(z) - 1) \
                  - 2 * E_LN * (Phi(z - sigma) + Phi(-sigma / math.sqrt(2)) - 1)
    crps_ln_dry = 2 * E_LN * (1 - Phi(-sigma / math.sqrt(2)))
    
    is_wet = (y > eps).float()
    crps_ln = is_wet * crps_ln_wet + (1 - is_wet) * crps_ln_dry
    
    # --- Zero-Inflated CRPS ---
    ln_spread = 2 * E_LN * (2 * Phi(sigma / math.sqrt(2)) - 1)
    
    crps = (1 - p)**2 * y \
         + p * crps_ln \
         + p * (1 - p) * E_LN \
         - 0.5 * p**2 * ln_spread
    
    # --- FIX 1: Lead-dependent weighting ---
    # Longer leads get higher weight.
    # Weights table: [1.0, 1.5, 2.0, 2.5]
    lead_weights_table = torch.tensor([1.0, 1.5, 2.0, 2.5], device=crps.device)
    
    if lead_indices is not None:
        # Single-Lead Case: (B, 1, H, W) -> weights (B, 1, 1, 1)
        lead_weights = lead_weights_table[lead_indices].view(-1, 1, 1, 1)
    else:
        # Multi-Lead Case: (B, 4, H, W) -> weights (1, 4, 1, 1)
        lead_weights = lead_weights_table.view(1, 4, 1, 1)
        
    crps = crps * lead_weights
    
    # --- FIX 2: Focal-style upweighting of wet cells (8x) ---
    WET_THRESHOLD = 1.0  # mm/day
    intensity_weight = 1.0 + 7.0 * (y > WET_THRESHOLD).float()  # 1x dry, 8x wet
    crps = crps * intensity_weight
    
    # Area weighting
    if area_weights is not None:
        crps = crps * area_weights
    
    crps_loss = crps.mean()
    
    # --- FIX 3: BCE auxiliary with 5x false-negative penalty ---
    wet_label = (target > WET_THRESHOLD).float()
    bce = F.binary_cross_entropy(p, wet_label, reduction='none')
    bce = bce * (1.0 + 4.0 * wet_label)  # 1x false-pos, 5x false-neg
    bce = bce * lead_weights  # Lead-dependent too
    if area_weights is not None:
        bce = bce * area_weights
    bce_loss = bce.mean()
    
    # --- FIX 4: Mean-bias penalty ---
    # Penalizes systematic underprediction: if E[rain] < target on average,
    # add a squared-bias term. This is asymmetric — only penalizes underprediction.
    pred_mean = p * E_LN  # Expected value of ZILN
    bias_per_lead = (y - pred_mean).mean(dim=(0, 2, 3))  # (4,) positive = underprediction
    underpred_bias = torch.clamp(bias_per_lead, min=0.0)
    bias_loss = (underpred_bias ** 2).mean()
    
    # --- FIX 5: Direct L1 intensity matching ---
    l1_err = torch.abs(pred_mean - y)
    l1_weight = 1.0 + torch.sqrt(y)
    l1_err = l1_err * l1_weight * lead_weights
    if area_weights is not None:
        l1_err = l1_err * area_weights
    l1_loss = l1_err.mean()
    
    # --- FIX 6: Sobel gradient sharpness loss ---
    # Penalizes the model for producing spatially blurry predictions.
    # Computes Sobel edge-detection on both prediction and target, then
    # matches them with L1 to force the model to preserve sharp spatial features.
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=pred_mean.dtype, device=pred_mean.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=pred_mean.dtype, device=pred_mean.device).view(1, 1, 3, 3)
    
    # Reshape (B, 4, H, W) -> (B*4, 1, H, W) for conv2d
    B_lead = pred_mean.shape[0] * pred_mean.shape[1]
    pred_2d = pred_mean.reshape(B_lead, 1, pred_mean.shape[2], pred_mean.shape[3])
    targ_2d = y.reshape(B_lead, 1, y.shape[2], y.shape[3])
    
    pred_gx = F.conv2d(pred_2d, sobel_x, padding=1)
    pred_gy = F.conv2d(pred_2d, sobel_y, padding=1)
    targ_gx = F.conv2d(targ_2d, sobel_x, padding=1)
    targ_gy = F.conv2d(targ_2d, sobel_y, padding=1)
    
    # Gradient magnitude
    pred_grad = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-8)
    targ_grad = torch.sqrt(targ_gx**2 + targ_gy**2 + 1e-8)
    
    grad_loss = F.l1_loss(pred_grad, targ_grad)
    
    # Combined: CRPS + 0.3*BCE + 0.2*bias + 0.5*L1 + 1.0*gradient (increased for sharpness)
    return crps_loss + 0.3 * bce_loss + 0.2 * bias_loss + 0.5 * l1_loss + 1.0 * grad_loss


def spatial_gradient_loss(pred, target):
    """
    L1 loss on spatial gradients to encourage edge/texture sharpness.
    Operates on raw pixel differences (simpler than Sobel, complementary).
    """
    # Horizontal gradients
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    # Vertical gradients
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def spectral_loss(pred, target):
    """
    FFT-based spectral loss: penalizes missing high-frequency content.
    
    Computes 2D FFT of pred and target, then matches log-magnitude spectra.
    This directly targets blurriness (smooth = missing high frequencies).
    More stable than GAN, no trainable discriminator needed.
    """
    # 2D Real FFT on spatial dims
    pred_fft = torch.fft.rfft2(pred, norm='ortho')
    target_fft = torch.fft.rfft2(target, norm='ortho')
    
    # Log-magnitude spectrum (log1p puts all frequency bands on equal footing)
    pred_mag = torch.log1p(torch.abs(pred_fft))
    target_mag = torch.log1p(torch.abs(target_fft))
    
    return F.l1_loss(pred_mag, target_mag)


def get_area_weights(lats, device):
    """
    Calculates area weights based on cosine of latitude.
    Normalizes weights to have a mean of 1.
    """
    weights = np.cos(np.deg2rad(lats))
    weights = weights / weights.mean()
    # Shape: (1, 1, H, 1) for broadcasting with (B, 4, H, W)
    weights_tensor = torch.from_numpy(weights).float().to(device)
    return weights_tensor.view(1, 1, -1, 1)


# ==============================================================================
# TRAIN
# ==============================================================================

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config_unet.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Accelerator
    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    device = accelerator.device

    # Dataset
    preload = config.get("preload", False)
    train_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["train_start_year"],
        end_year=config["train_end_year"],
        normalize=True,
        preload=preload
    )
    
    loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # Validation Dataset
    val_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["val_start_year"],
        end_year=config["val_end_year"],
        normalize=True,
        preload=preload
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    # Validation Fixed Batch for consistent plotting
    fixed_val_batch = next(iter(val_loader))
    
    # Model: TemporalAttentionUNet
    # Input:  46 channels (28 Obs + 16 GEOS flat + 2 Month sin/cos)
    #   Obs: SST(4)+SSS(4)+SM(4)+PrevGPCP(4)+IVT(4)+Z500(4)+U250(4) = 28
    # Output: 3 channels (p, mu, sigma) for ONE lead week (conditioned on lead_idx)
    in_channels = 48  # 28 Obs + 16 GEOS + 2 Seasonality + 2 MJO (RMM1, RMM2)
    out_channels = 3   # 3 params * 1 lead

    model = TemporalAttentionUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=128,
        emb_dim=256,
        n_weeks=4,
        temporal_heads=4
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    
    # Scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=float(config["learning_rate"]), 
        steps_per_epoch=len(loader), 
        epochs=config["epochs"]
    )
    
    # --- PatchGAN Discriminator ---
    disc = PatchGANDiscriminator(in_channels=2, ndf=64)
    disc_optimizer = torch.optim.AdamW(disc.parameters(), lr=2e-5, betas=(0.5, 0.999))
    GAN_WARMUP_START = 3
    GAN_WARMUP_END = 20
    GAN_WEIGHT = 0.0       # DISABLED: D collapses to d_loss=0.003, provides no signal
    SHARP_WEIGHT = 20.0    # Spatial gradient sharpness loss
    SPECTRAL_WEIGHT = 1.0  # FFT spectral loss (replaces GAN for sharpness)
    D_TRAIN_RATIO = 5
    D_NOISE_STD = 0.1
    global_step = 0

    # Prepare
    model, optimizer, loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, loader, val_loader, lr_scheduler
    )
    disc, disc_optimizer = accelerator.prepare(disc, disc_optimizer)

    # Area Weights for Loss
    lats = np.linspace(-90, 90, 181)
    area_weights = get_area_weights(lats, device)

    # Output Dir
    os.makedirs(config["output_dir"], exist_ok=True)
    log_file = os.path.join(config["output_dir"], "training_log_unet.csv")
    if accelerator.is_main_process:
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train_CRPS", "Val_CRPS", "Val_RMSE"])

    # Load Checkpoint?
    start_epoch = 0
    latest_ckpt = os.path.join(config["output_dir"], "latest_unet_ckpt.pt")
    
    # Top K Checkpoints
    top_k_ckpts = []  # List of (rmse, epoch, path)
    save_top_k = config.get("save_top_k", 4)
    
    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'disc' in checkpoint:
            disc.load_state_dict(checkpoint['disc'])
            disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        if 'top_k_ckpts' in checkpoint:
            top_k_ckpts = checkpoint['top_k_ckpts']
    best_val_rmse = float('inf')
    if top_k_ckpts:
        best_val_rmse = top_k_ckpts[0][0]
        print(f"Resumed Best Val RMSE: {best_val_rmse:.4f}")

    # Print Model Info
    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"TemporalAttentionUNet Parameters: {n_params:,}")
        print(f"Input Channels: {in_channels}, Output Channels: {out_channels}")
        print(f"ZILN Parameterization: 3 params (p, mu, sigma) x 4 leads = 12 output channels")

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        disc.train()
        train_loss = 0.0
        train_disc_loss = 0.0
        
        # GAN weight warmup: 0 for epochs < 3, ramps to GAN_WEIGHT by epoch 6
        if epoch < GAN_WARMUP_START:
            gan_w = 0.0
        elif epoch < GAN_WARMUP_END:
            gan_w = GAN_WEIGHT * (epoch - GAN_WARMUP_START) / (GAN_WARMUP_END - GAN_WARMUP_START)
        else:
            gan_w = GAN_WEIGHT
        
        pbar = tqdm(loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch} (GAN w={gan_w:.4f})")

        for batch in pbar:
            x_obs = batch['x_obs']
            x_geos = batch['x_geos']
            y_target = batch['y_target']
            months = batch['month']
            
            B, _, H, W = x_obs.shape
            
            # GEOS: (B, 4, 1, 4, H, W) -> (B, 16, H, W)
            x_geos_flat = x_geos.squeeze(2).reshape(B, 16, H, W)
            
            y_target = y_target.clamp(min=0.0)
            
            # Month Embeddings
            sin_month = torch.sin(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
            cos_month = torch.cos(2 * np.pi * (months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W).to(device)
            
            # MJO: broadcast (B, 2) -> (B, 2, H, W)
            mjo = batch['mjo']  # (B, 2)
            mjo_map = mjo.view(B, 2, 1, 1).expand(B, 2, H, W).to(device)
            
            x_input = torch.cat([x_obs, x_geos_flat, sin_month, cos_month, mjo_map], dim=1)
            month_onehot = F.one_hot(months.long() - 1, num_classes=12).float().to(device)
            
            # --- SINGLE LEAD TRAINING ---
            # Randomly select one lead (0-3) for this batch
            target_lead = torch.randint(0, 4, (1,)).item()
            lead_indices = torch.full((B,), target_lead, dtype=torch.long, device=device)
            
            # Slice Target for this lead: (B, 1, H, W)
            y_target_lead = y_target[:, target_lead:target_lead+1, :, :]
            
            # GEOS condition for discriminator: (B, 1, H, W) corresponding to target lead
            # x_geos_flat (B, 16, H, W) -> view (B, 4, 4, H, W) -> slice lead -> mean members -> (B, 1, H, W)
            geos_cond_lead = x_geos_flat.view(B, 4, 4, H, W)[:, :, target_lead, :, :].mean(dim=1, keepdim=True)
            
            # --- Generator Forward ---
            # Predict only one lead (3 channels output)
            raw_output = model(x_input, month_onehot, lead_indices)
            p, mu, sigma = parameterize_ziln(raw_output)
            pred_mean = ziln_expected_value(p, mu, sigma)  # (B, 1, H, W)
            
            # --- Step D: Update Discriminator (every D_TRAIN_RATIO steps) ---
            if gan_w > 0 and global_step % D_TRAIN_RATIO == 0:
                disc_optimizer.zero_grad()
                
                with torch.no_grad():
                    # Use SAMPLE (sharp) instead of E[rain] (smooth)
                    fake_sample_d = ziln_sample(p, mu, sigma).detach()
                
                # Add instance noise to D inputs (prevents D from memorizing pixel-level differences)
                noise_real = torch.randn_like(y_target_lead) * D_NOISE_STD
                noise_fake = torch.randn_like(fake_sample_d) * D_NOISE_STD
                
                # Discriminator sees (Precip + noise, GEOS_Cond) pair
                disc_real_out = disc(y_target_lead + noise_real, geos_cond_lead)
                disc_fake_out = disc(fake_sample_d + noise_fake, geos_cond_lead)
                
                # LSGAN Loss with label smoothing
                d_loss = discriminator_loss([disc_real_out], [disc_fake_out], target_real=0.9, target_fake=0.0)
                
                accelerator.backward(d_loss)
                accelerator.clip_grad_norm_(disc.parameters(), max_norm=1.0)
                disc_optimizer.step()
                train_disc_loss += d_loss.item()
            
            # --- Step G: Update Generator ---
            optimizer.zero_grad()
            
            # Base losses on SINGLE LEAD (includes CRPS + BCE + Sobel gradient)
            g_loss = crps_ziln_loss(p, mu, sigma, y_target_lead, lead_indices=lead_indices, area_weights=area_weights)
            
            # Spatial gradient sharpness loss on E[rain]
            sharp_loss = spatial_gradient_loss(pred_mean, y_target_lead)
            g_loss = g_loss + SHARP_WEIGHT * sharp_loss
            
            # Spectral (FFT) sharpness loss on E[rain] — matches frequency content
            spec_loss = spectral_loss(pred_mean, y_target_lead)
            g_loss = g_loss + SPECTRAL_WEIGHT * spec_loss
            
            # Adversarial loss (fool discriminator with SAMPLE, not E[rain])
            if gan_w > 0:
                fake_sample_g = ziln_sample(p, mu, sigma)  # new sample, with gradients
                disc_fake_out_g = disc(fake_sample_g, geos_cond_lead)
                g_adv_loss = generator_adversarial_loss([disc_fake_out_g])
                g_loss = g_loss + gan_w * g_adv_loss
            
            accelerator.backward(g_loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            global_step += 1
            
            train_loss += g_loss.item()
            d_str = f"{train_disc_loss / max(1, pbar.n+1):.3f}" if gan_w > 0 else "off"
            pbar.set_postfix({"g_loss": f"{g_loss.item():.4f}", "d_loss": d_str})
            
        avg_train_loss = train_loss / len(loader)
        
        # --- VALIDATION LOOP ---
        model.eval()
        val_crps_sum = 0
        val_count = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                vx_obs = val_batch['x_obs']
                vx_geos = val_batch['x_geos']
                vy_target = val_batch['y_target'].clamp(min=0.0)
                v_months = val_batch['month']
                
                vB, _, vH, vW = vx_obs.shape
                
                # Reshape GEOS
                vx_geos_flat = vx_geos.squeeze(2).reshape(vB, 16, vH, vW)

                # Month Embeddings
                v_sin_month = torch.sin(2 * np.pi * (v_months - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, vH, vW).to(device)
                v_cos_month = torch.cos(2 * np.pi * (v_months - 1) / 12).view(vB, 1, 1, 1).expand(vB, 1, vH, vW).to(device)
                
                # MJO: broadcast (B, 2) -> (B, 2, H, W)
                v_mjo = val_batch['mjo']
                v_mjo_map = v_mjo.view(vB, 2, 1, 1).expand(vB, 2, vH, vW).to(device)
                
                # Model Input
                v_input = torch.cat([vx_obs, vx_geos_flat, v_sin_month, v_cos_month, v_mjo_map], dim=1)
                v_month_onehot = F.one_hot(v_months.long() - 1, num_classes=12).float().to(device)
                
                # Forward - Reconstruct full 4 leads iteratively
                v_p_list, v_mu_list, v_sigma_list = [], [], []
                
                for lead in range(4):
                    lead_indices = torch.full((vB,), lead, dtype=torch.long, device=device)
                    # Generator forward for single lead
                    v_raw_lead = model(v_input, v_month_onehot, lead_indices)
                    vp, vmu, vsigma = parameterize_ziln(v_raw_lead)
                    
                    v_p_list.append(vp)
                    v_mu_list.append(vmu)
                    v_sigma_list.append(vsigma)
                
                # Stack to reconstruct (B, 4, H, W)
                v_p = torch.cat(v_p_list, dim=1)
                v_mu = torch.cat(v_mu_list, dim=1)
                v_sigma = torch.cat(v_sigma_list, dim=1)
                
                # Validation CRPS (on full forecast)
                # Pass lead_indices=None to use multi-lead weighting
                v_crps = crps_ziln_loss(v_p, v_mu, v_sigma, vy_target, lead_indices=None, area_weights=area_weights)
                val_crps_sum += v_crps.item()
                val_count += 1

        avg_val_crps = val_crps_sum / val_count if val_count > 0 else 0
        
        # --- FIXED BATCH VISUALIZATION & RMSE ---
        fb_obs = fixed_val_batch['x_obs'].to(device)
        fb_geos = fixed_val_batch['x_geos'].to(device)
        fb_target = fixed_val_batch['y_target'].to(device).clamp(min=0.0)
        fb_months = fixed_val_batch['month'].to(device)
        
        fb_B = fb_obs.shape[0]
        _, _, H, W = fb_obs.shape
        fb_geos_flat = fb_geos.squeeze(2).reshape(fb_B, 16, H, W)
        
        # Month
        fb_sin_month = torch.sin(2 * np.pi * (fb_months - 1) / 12).view(fb_B, 1, 1, 1).expand(fb_B, 1, H, W).to(device)
        fb_cos_month = torch.cos(2 * np.pi * (fb_months - 1) / 12).view(fb_B, 1, 1, 1).expand(fb_B, 1, H, W).to(device)
        
        # MJO
        fb_mjo = fixed_val_batch['mjo']
        fb_mjo_map = fb_mjo.view(fb_B, 2, 1, 1).expand(fb_B, 2, H, W).to(device)
        
        fb_input = torch.cat([fb_obs, fb_geos_flat, fb_sin_month, fb_cos_month, fb_mjo_map], dim=1)
        fb_month_onehot = F.one_hot(fb_months.long() - 1, num_classes=12).float().to(device)
        
        unwrapped_model = accelerator.unwrap_model(model)
        
        with torch.no_grad():
            # Iterative Reconstruction
            fb_p_list, fb_mu_list, fb_sigma_list = [], [], []
            for lead in range(4):
                ld_idx = torch.full((fb_B,), lead, dtype=torch.long, device=device)
                fb_raw_lead = unwrapped_model(fb_input, fb_month_onehot, ld_idx)
                fbp, fbmu, fbsigma = parameterize_ziln(fb_raw_lead)
                fb_p_list.append(fbp); fb_mu_list.append(fbmu); fb_sigma_list.append(fbsigma)
            
            fb_p = torch.cat(fb_p_list, dim=1)
            fb_mu = torch.cat(fb_mu_list, dim=1)
            fb_sigma = torch.cat(fb_sigma_list, dim=1)
            
            # Expected value: E[Rain] = p * exp(mu + sigma^2/2)
            fb_pred = ziln_expected_value(fb_p, fb_mu, fb_sigma)  # (B, 4, H, W)
        
        # Compute RMSE on Fixed Batch (All Leads)
        val_rmse = torch.sqrt(torch.mean((fb_pred - fb_target)**2)).item()
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch} | CRPS: {avg_train_loss:.4f} | Val CRPS: {avg_val_crps:.4f} | Val RMSE (Fixed): {val_rmse:.4f}")
            
            with open(log_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_train_loss, avg_val_crps, val_rmse])
            
            if val_rmse < best_val_rmse:
                print(f"New Best Model Found! RMSE improved from {best_val_rmse:.4f} to {val_rmse:.4f}. Plotting...")
                best_val_rmse = val_rmse
                
                # Plot First Sample, All 4 Leads
                s_img_all = fb_pred[0].cpu().numpy()    # (4, H, W)
                t_img_all = fb_target[0].cpu().numpy()  # (4, H, W)
                p_img_all = fb_p[0].cpu().numpy()       # (4, H, W) rain prob
                
                # GEOS Mean (4 leads) - Denormalize
                g_flat = fb_geos_flat[0]  # (16, H, W)
                g_ens = g_flat.view(4, 4, H, W)  # (Members, Leads, H, W)
                g_mean_norm = g_ens.mean(dim=0)   # (Leads, H, W) -> (4, H, W)
                
                # Denormalize GEOS Mean: inverse of log1p is expm1
                g_img_all = np.expm1(np.maximum(g_mean_norm.cpu().numpy(), 0.0))

                fig, axes = plt.subplots(4, 5, figsize=(25, 20))
                
                for l_idx in range(4):
                    g_img = g_img_all[l_idx]
                    t_img = t_img_all[l_idx]
                    s_img = s_img_all[l_idx]
                    diff_img = s_img - t_img
                    geos_bias = g_img - t_img
                    
                    rmse_l = np.sqrt(np.mean((s_img - t_img)**2))
                    
                    if l_idx == 0: axes[l_idx, 0].set_title("GEOS Mean")
                    axes[l_idx, 0].imshow(g_img, cmap='Blues', vmin=0, vmax=50)
                    
                    if l_idx == 0: axes[l_idx, 1].set_title("Target GPCP")
                    axes[l_idx, 1].imshow(t_img, cmap='Blues', vmin=0, vmax=50)
                    
                    if l_idx == 0: axes[l_idx, 2].set_title("UNet E[Rain]")
                    axes[l_idx, 2].imshow(s_img, cmap='Blues', vmin=0, vmax=50)
                    axes[l_idx, 2].set_ylabel(f"Week {l_idx+1}\nRMSE: {rmse_l:.2f}")
                    
                    if l_idx == 0: axes[l_idx, 3].set_title("UNet Bias")
                    axes[l_idx, 3].imshow(diff_img, cmap='RdBu_r', vmin=-20, vmax=20)
                    
                    if l_idx == 0: axes[l_idx, 4].set_title("GEOS Bias")
                    axes[l_idx, 4].imshow(geos_bias, cmap='RdBu_r', vmin=-20, vmax=20)

                os.makedirs(os.path.join(config["output_dir"], "plots_unet"), exist_ok=True)
                plt.suptitle(f"UNet ZILN - Epoch {epoch} | RMSE: {val_rmse:.2f} | CRPS: {avg_val_crps:.4f}", fontsize=14)
                plt.savefig(os.path.join(config["output_dir"], f"plots_unet/epoch_{epoch}_rmse_{val_rmse:.2f}.png"))
                plt.close()
            else:
                print(f"Validation RMSE ({val_rmse:.4f}) did not improve over current best ({best_val_rmse:.4f}). Skipping plot.")
            
            # SAVE LATEST
            ckpt_state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'disc': disc.state_dict(),
                'disc_optimizer': disc_optimizer.state_dict(),
                'epoch': epoch,
                'top_k_ckpts': top_k_ckpts
            }
            torch.save(ckpt_state, latest_ckpt)
            
            # SAVE TOP K logic
            current_path = os.path.join(config["output_dir"], f"unet_epoch_{epoch}_rmse_{val_rmse:.4f}.pt")
            top_k_ckpts.append((val_rmse, epoch, current_path))
            top_k_ckpts.sort(key=lambda x: x[0]) 
            
            if len(top_k_ckpts) > save_top_k:
                worst = top_k_ckpts.pop()
                if worst[2] != current_path and os.path.exists(worst[2]):
                        os.remove(worst[2])
                        print(f"Removed worse checkpoint: {worst[2]}")
            
            is_in_top = any(x[2] == current_path for x in top_k_ckpts)
            if is_in_top:
                print(f"New Top Model! RMSE: {val_rmse:.4f}")
                torch.save(ckpt_state, current_path)
                
            ckpt_state['top_k_ckpts'] = top_k_ckpts
            torch.save(ckpt_state, latest_ckpt)


# ==============================================================================
# TEST
# ==============================================================================

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config_unet.yaml", help="Path to config file")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--n_samples", type=int, default=12, help="Number of test samples")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    device = accelerator.device

    # Validation Dataset
    val_dataset = S2SHybridDataset(
        data_root=config["data_dir"],
        start_year=config["val_start_year"],
        end_year=config["val_end_year"],
        normalize=True,
        preload=config.get("preload", False)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # Model
    model = TemporalAttentionUNet(
        in_channels=48,  # 28 Obs + 16 GEOS + 2 Seasonality + 2 MJO
        out_channels=12,
        base_filters=128,
        emb_dim=256,
        n_weeks=4,
        temporal_heads=4
    )

    # Load Best or Latest Model
    latest_ckpt = os.path.join(config["output_dir"], "latest_unet_ckpt.pt")
    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location='cpu')
        top_k = checkpoint.get('top_k_ckpts', [])
        
        if top_k:
            best_ckpt_path = top_k[0][2]
            print(f"Loading best model from {best_ckpt_path} (RMSE: {top_k[0][0]:.4f})")
            if os.path.exists(best_ckpt_path):
                ckpt = torch.load(best_ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['model'])
            else:
                print(f"Best model file missing, loading latest instead.")
                model.load_state_dict(checkpoint['model'])
        else:
            print("No top_k info, loading latest.")
            model.load_state_dict(checkpoint['model'])
    else:
        print("No checkpoint found.")
        return

    model, val_loader = accelerator.prepare(model, val_loader)
    model.eval()

    # Pick N evenly-spaced indices from validation set
    n_val = len(val_dataset)
    n_samples = min(args.n_samples, n_val)
    test_indices = set(np.linspace(0, n_val - 1, n_samples, dtype=int).tolist())
    
    output_dir = os.path.join(config["output_dir"], "plots_test_suite")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running Test Suite: {n_samples} samples from {n_val} validation samples...")
    print(f"Indices: {sorted(test_indices)}")

    import cartopy.crs as ccrs
    from scipy import stats as sp_stats
    
    # Accumulators for correlation plot
    # Per lead: list of (pred_flat, target_flat) arrays
    all_pred = {lead: [] for lead in range(4)}
    all_target = {lead: [] for lead in range(4)}
    all_geos = {lead: [] for lead in range(4)}
    rmse_unet = {lead: [] for lead in range(4)}
    rmse_geos = {lead: [] for lead in range(4)}
    # Spatial accumulators: list of 2D arrays (H, W) per lead per sample
    spatial_target = {lead: [] for lead in range(4)}
    spatial_pred = {lead: [] for lead in range(4)}
    spatial_geos = {lead: [] for lead in range(4)}
    crps_unet_acc = {lead: [] for lead in range(4)}
    crps_geos_acc = {lead: [] for lead in range(4)}
    
    current_idx = 0
    samples_processed = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if current_idx not in test_indices:
                current_idx += 1
                continue
                
            print(f"  [{samples_processed+1}/{n_samples}] Sample {current_idx}...")
            
            x_obs = batch['x_obs'].to(device)
            x_geos = batch['x_geos'].to(device)
            y_target = batch['y_target'].to(device).clamp(min=0.0)
            t_months = batch['month'].to(device)
            
            B = x_obs.shape[0]
            _, _, H, W = x_obs.shape
            
            # Reshape GEOS
            x_geos_flat = x_geos.squeeze(2).reshape(B, 16, H, W)
            
            # Month Embeddings
            t_sin = torch.sin(2 * np.pi * (t_months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W)
            t_cos = torch.cos(2 * np.pi * (t_months - 1) / 12).view(B, 1, 1, 1).expand(B, 1, H, W)
            
            # MJO: broadcast (B, 2) -> (B, 2, H, W)
            t_mjo = batch['mjo'].to(device)
            t_mjo_map = t_mjo.view(B, 2, 1, 1).expand(B, 2, H, W)
            
            x_input = torch.cat([x_obs, x_geos_flat, t_sin, t_cos, t_mjo_map], dim=1)
            month_onehot = F.one_hot(t_months.long() - 1, num_classes=12).float().to(device)
            
            # Forward
            unwrapped = accelerator.unwrap_model(model)
            raw_out = unwrapped(x_input, month_onehot)
            pred_p, pred_mu, pred_sigma = parameterize_ziln(raw_out)
            pred_mean = ziln_expected_value(pred_p, pred_mu, pred_sigma)  # (1, 4, H, W)
            
            target_np = y_target.squeeze(0).cpu().numpy()    # (4, H, W)
            pred_np = pred_mean.squeeze(0).cpu().numpy()      # (4, H, W)
            
            # GEOS denormalize (inverse log1p = expm1)
            geos_ens = x_geos_flat.view(4, 4, H, W)
            geos_mean_log = geos_ens.mean(dim=0)  # (4, H, W)
            geos_np = torch.expm1(geos_mean_log.clamp(min=0.0)).cpu().numpy()
            
            # Accumulate for correlation
            for lead in range(4):
                t_2d = target_np[lead]  # (H, W)
                p_2d = pred_np[lead]
                g_2d = geos_np[lead]
                
                all_target[lead].append(t_2d.flatten())
                all_pred[lead].append(p_2d.flatten())
                all_geos[lead].append(g_2d.flatten())
                
                # Spatial accumulators (keep 2D)
                spatial_target[lead].append(t_2d.copy())
                spatial_pred[lead].append(p_2d.copy())
                spatial_geos[lead].append(g_2d.copy())
                
                rmse_unet[lead].append(np.sqrt(np.mean((p_2d - t_2d)**2)))
                rmse_geos[lead].append(np.sqrt(np.mean((g_2d - t_2d)**2)))
            
            # --- CRPS computation ---
            # UNet CRPS: pure ZILN CRPS (no focal/BCE) for evaluation
            Phi_np = lambda x: 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
            p_param = pred_p.squeeze(0).cpu().numpy()      # (4, H, W)
            mu_param = pred_mu.squeeze(0).cpu().numpy()
            sigma_param = pred_sigma.squeeze(0).cpu().numpy()
            
            for lead in range(4):
                y = target_np[lead].clip(min=0.0)
                pp = p_param[lead]
                mm = mu_param[lead]
                ss = sigma_param[lead]
                
                y_safe = np.maximum(y, 1e-6)
                z = (np.log(y_safe) - mm) / ss
                E_LN = np.exp(mm + 0.5 * ss**2)
                
                # CRPS LN for wet obs
                crps_ln_wet = y_safe * (2 * Phi_np(z) - 1) \
                    - 2 * E_LN * (Phi_np(z - ss) + Phi_np(-ss / math.sqrt(2)) - 1)
                # CRPS LN for dry obs
                crps_ln_dry = 2 * E_LN * (1 - Phi_np(-ss / math.sqrt(2)))
                
                is_wet = (y > 1e-6).astype(np.float32)
                crps_ln = is_wet * crps_ln_wet + (1 - is_wet) * crps_ln_dry
                
                ln_spread = 2 * E_LN * (2 * Phi_np(ss / math.sqrt(2)) - 1)
                crps_pixel = (1 - pp)**2 * y + pp * crps_ln + pp * (1 - pp) * E_LN - 0.5 * pp**2 * ln_spread
                crps_unet_acc[lead].append(float(np.mean(crps_pixel)))
            
            # GEOS Ensemble CRPS: CRPS = (1/M)Σ|x_m-y| - (1/2M²)ΣΣ|x_m-x_n|
            # geos_ens: (4_members, 4_leads, H, W) — denormalize each member
            geos_members = torch.expm1(geos_ens.clamp(min=0.0)).cpu().numpy()  # (4, 4, H, W)
            for lead in range(4):
                y = target_np[lead]
                members = geos_members[:, lead, :, :]  # (4, H, W)
                M = members.shape[0]
                mae_term = np.mean([np.abs(members[m] - y) for m in range(M)], axis=0)
                spread_term = 0.0
                for m in range(M):
                    for n in range(M):
                        spread_term += np.abs(members[m] - members[n])
                spread_term = spread_term / (2 * M * M)
                crps_geos_acc[lead].append(float(np.mean(mae_term - spread_term)))
            
            # --- Per-Sample Map Plot ---
            lats = np.linspace(-90, 90, H)
            lons = np.linspace(0, 360, W)
            
            fig = plt.figure(figsize=(25, 20))
            for lead in range(4):
                g_img = geos_np[lead]
                t_img = target_np[lead]
                d_img = pred_np[lead]
                
                u_rmse = rmse_unet[lead][-1]
                g_rmse = rmse_geos[lead][-1]
                
                for col, (data, title, cmap, vmin, vmax) in enumerate([
                    (g_img,         f"W{lead+1}: GEOS Mean\nRMSE: {g_rmse:.2f}",   'Blues',  0, 50),
                    (t_img,         f"W{lead+1}: Target GPCP",                       'Blues',  0, 50),
                    (d_img,         f"W{lead+1}: UNet E[Rain]\nRMSE: {u_rmse:.2f}", 'Blues',  0, 50),
                    (d_img - t_img, f"W{lead+1}: UNet Bias",                         'RdBu_r', -20, 20),
                    (g_img - t_img, f"W{lead+1}: GEOS Bias",                         'RdBu_r', -20, 20),
                ]):
                    ax = fig.add_subplot(4, 5, lead * 5 + col + 1, projection=ccrs.PlateCarree())
                    im = ax.imshow(data, origin='lower',
                                   extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                                   transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.coastlines()
                    ax.set_title(title, fontsize=10)
                    
                    if lead == 0 and col == 0:
                        cax = fig.add_axes([0.92, 0.55, 0.015, 0.30])
                        fig.colorbar(im, cax=cax, label='mm/day')
                    if lead == 0 and col == 3:
                        cax = fig.add_axes([0.92, 0.12, 0.015, 0.30])
                        fig.colorbar(im, cax=cax, label='mm/day')

            plt.suptitle(f"UNet ZILN — Sample {current_idx} (Val Set)", fontsize=16)
            plt.savefig(os.path.join(output_dir, f"test_sample_{current_idx}.png"),
                        bbox_inches='tight', dpi=150)
            plt.close()
            
            samples_processed += 1
            current_idx += 1
    
    # ==========================================================================
    # SUMMARY: Correlation Scatter Plot (All Samples Combined)
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Generating Correlation Plots ({samples_processed} samples)...")
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    for lead in range(4):
        t_all = np.concatenate(all_target[lead])
        p_all = np.concatenate(all_pred[lead])
        g_all = np.concatenate(all_geos[lead])
        
        # Mean RMSE across samples
        mean_rmse_u = np.mean(rmse_unet[lead])
        mean_rmse_g = np.mean(rmse_geos[lead])
        
        # Correlation (Pearson r)
        r_unet, _ = sp_stats.pearsonr(t_all, p_all)
        r_geos, _ = sp_stats.pearsonr(t_all, g_all)
        
        # --- Row 1: UNet vs Target ---
        ax = axes[0, lead]
        # Subsample for plotting clarity (max 50k points)
        n_pts = len(t_all)
        if n_pts > 50000:
            idx = np.random.choice(n_pts, 50000, replace=False)
            t_sub, p_sub = t_all[idx], p_all[idx]
        else:
            t_sub, p_sub = t_all, p_all
            
        ax.scatter(t_sub, p_sub, s=0.5, alpha=0.15, c='steelblue', rasterized=True)
        ax.plot([0, 50], [0, 50], 'r--', lw=1.5, label='1:1')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.set_aspect('equal')
        ax.set_title(f"Week {lead+1}: UNet\nr={r_unet:.3f}  RMSE={mean_rmse_u:.2f}", fontsize=12)
        ax.set_xlabel("Target GPCP (mm/day)")
        ax.set_ylabel("UNet E[Rain] (mm/day)")
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # --- Row 2: GEOS vs Target ---
        ax2 = axes[1, lead]
        if n_pts > 50000:
            g_sub = g_all[idx]
        else:
            g_sub = g_all
            
        ax2.scatter(t_sub, g_sub, s=0.5, alpha=0.15, c='darkorange', rasterized=True)
        ax2.plot([0, 50], [0, 50], 'r--', lw=1.5, label='1:1')
        ax2.set_xlim(0, 50)
        ax2.set_ylim(0, 50)
        ax2.set_aspect('equal')
        ax2.set_title(f"Week {lead+1}: GEOS\nr={r_geos:.3f}  RMSE={mean_rmse_g:.2f}", fontsize=12)
        ax2.set_xlabel("Target GPCP (mm/day)")
        ax2.set_ylabel("GEOS Ens Mean (mm/day)")
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Correlation: UNet (top) vs GEOS (bottom) — {samples_processed} Samples", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    corr_path = os.path.join(output_dir, "correlation_all_samples.png")
    plt.savefig(corr_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation plot: {corr_path}")
    
    # ==========================================================================
    # SUMMARY TABLE
    # ==========================================================================
    print(f"\n{'='*90}")
    print(f"{'Lead':<8} {'UNet RMSE':<11} {'GEOS RMSE':<11} {'UNet r':<9} {'GEOS r':<9} {'UNet CRPS':<11} {'GEOS CRPS':<11} {'RMSE Imp':<10}")
    print(f"{'-'*90}")
    for lead in range(4):
        t_all = np.concatenate(all_target[lead])
        p_all = np.concatenate(all_pred[lead])
        g_all = np.concatenate(all_geos[lead])
        
        mu = np.mean(rmse_unet[lead])
        mg = np.mean(rmse_geos[lead])
        ru, _ = sp_stats.pearsonr(t_all, p_all)
        rg, _ = sp_stats.pearsonr(t_all, g_all)
        cu = np.mean(crps_unet_acc[lead])
        cg = np.mean(crps_geos_acc[lead])
        imp = (mg - mu) / mg * 100
        
        print(f"Week {lead+1:<3} {mu:<11.3f} {mg:<11.3f} {ru:<9.3f} {rg:<9.3f} {cu:<11.3f} {cg:<11.3f} {imp:>+.1f}%")
    
    overall_u = np.mean([np.mean(rmse_unet[l]) for l in range(4)])
    overall_g = np.mean([np.mean(rmse_geos[l]) for l in range(4)])
    overall_cu = np.mean([np.mean(crps_unet_acc[l]) for l in range(4)])
    overall_cg = np.mean([np.mean(crps_geos_acc[l]) for l in range(4)])
    overall_imp = (overall_g - overall_u) / overall_g * 100
    print(f"{'-'*90}")
    print(f"{'Mean':<8} {overall_u:<11.3f} {overall_g:<11.3f} {'':9} {'':9} {overall_cu:<11.3f} {overall_cg:<11.3f} {overall_imp:>+.1f}%")
    print(f"{'='*90}")
    
    # ==========================================================================
    # SPATIAL CORRELATION MAPS
    # ==========================================================================
    print(f"\nGenerating Spatial Correlation Maps...")
    
    lats_plot = np.linspace(-90, 90, spatial_target[0][0].shape[0])
    lons_plot = np.linspace(0, 360, spatial_target[0][0].shape[1])
    
    fig, axes = plt.subplots(3, 4, figsize=(24, 15),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    
    for lead in range(4):
        # Stack: (N_samples, H, W)
        t_stack = np.stack(spatial_target[lead], axis=0)
        p_stack = np.stack(spatial_pred[lead], axis=0)
        g_stack = np.stack(spatial_geos[lead], axis=0)
        
        N = t_stack.shape[0]
        H, W = t_stack.shape[1], t_stack.shape[2]
        
        # Per-pixel Pearson r across samples
        # r = cov(x,y) / (std(x)*std(y))
        def pixel_corr(a, b):
            """Compute per-pixel correlation between (N,H,W) arrays."""
            a_mean = a.mean(axis=0, keepdims=True)
            b_mean = b.mean(axis=0, keepdims=True)
            a_dev = a - a_mean
            b_dev = b - b_mean
            cov = (a_dev * b_dev).mean(axis=0)
            std_a = a_dev.std(axis=0) + 1e-8
            std_b = b_dev.std(axis=0) + 1e-8
            return cov / (std_a * std_b)
        
        r_unet_map = pixel_corr(t_stack, p_stack)
        r_geos_map = pixel_corr(t_stack, g_stack)
        r_diff_map = r_unet_map - r_geos_map  # positive = UNet better
        
        # Row 0: UNet spatial r
        ax0 = axes[0, lead]
        im0 = ax0.imshow(r_unet_map, origin='lower',
                         extent=[lons_plot.min(), lons_plot.max(), lats_plot.min(), lats_plot.max()],
                         transform=ccrs.PlateCarree(), cmap='RdYlGn', vmin=-0.2, vmax=1.0)
        ax0.coastlines(linewidth=0.5)
        ax0.set_title(f"Week {lead+1}: UNet r\nmean={np.nanmean(r_unet_map):.3f}", fontsize=11)
        
        # Row 1: GEOS spatial r
        ax1 = axes[1, lead]
        im1 = ax1.imshow(r_geos_map, origin='lower',
                         extent=[lons_plot.min(), lons_plot.max(), lats_plot.min(), lats_plot.max()],
                         transform=ccrs.PlateCarree(), cmap='RdYlGn', vmin=-0.2, vmax=1.0)
        ax1.coastlines(linewidth=0.5)
        ax1.set_title(f"Week {lead+1}: GEOS r\nmean={np.nanmean(r_geos_map):.3f}", fontsize=11)
        
        # Row 2: UNet - GEOS (improvement)
        ax2 = axes[2, lead]
        im2 = ax2.imshow(r_diff_map, origin='lower',
                         extent=[lons_plot.min(), lons_plot.max(), lats_plot.min(), lats_plot.max()],
                         transform=ccrs.PlateCarree(), cmap='RdBu', vmin=-0.3, vmax=0.3)
        ax2.coastlines(linewidth=0.5)
        ax2.set_title(f"Week {lead+1}: UNet−GEOS\nmean={np.nanmean(r_diff_map):.3f}", fontsize=11)
    
    # Colorbars
    cax1 = fig.add_axes([0.92, 0.40, 0.015, 0.48])
    fig.colorbar(im0, cax=cax1, label='Pearson r')
    cax2 = fig.add_axes([0.92, 0.08, 0.015, 0.25])
    fig.colorbar(im2, cax=cax2, label='Δr (UNet−GEOS)')
    
    plt.suptitle(f"Spatial Correlation Maps — {samples_processed} Samples", fontsize=16)
    spatial_path = os.path.join(output_dir, "spatial_correlation_maps.png")
    plt.savefig(spatial_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial correlation map: {spatial_path}")
    
    print("Test Suite Completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ml_model/config_unet.yaml", help="Path to config file")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()
    
    if args.test:
        test()
    else:
        train()
