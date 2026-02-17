import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderPhysics(nn.Module):
    """
    Encodes static/initial state (SST, SSS, Soil Moisture).
    Input: (B, C_obs, H, W) -> (B, EmbedDim, H_small, W_small)
    """
    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        # Simple ResNet-like or ConvNet
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2) # /2
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2) # /4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, embed_dim, kernel_size=3, padding=1, stride=2) # /8
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.conv4 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=2) # /16
        self.bn4 = nn.BatchNorm2d(embed_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x

class EncoderDynamic(nn.Module):
    """
    Encodes dynamic forecast trajectory (GEOS Pr).
    Input: (B, C_geos, L, H, W) -> (B, EmbedDim, H_small, W_small)
    Processes L as depth in Conv3D or channels?
    Using Conv3D to capture spatiotemporal evolution.
    """
    def __init__(self, in_channels=1, lead_time=4, embed_dim=128):
        super().__init__()
        # (B, C, L, H, W)
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,2,2)) # /2 spatial
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,2,2)) # /2 time, /4 spatial
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, embed_dim, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,2,2)) # /4 time (L=1), /8 spatial
        self.bn3 = nn.BatchNorm3d(embed_dim)
        # Final layer to align dimensions?
        # If L=4 -> /2 -> 2 -> /2 -> 1. Perfect.
        self.conv4 = nn.Conv3d(embed_dim, embed_dim, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,2,2)) # /16 spatial
        self.bn4 = nn.BatchNorm3d(embed_dim)
        
    def forward(self, x):
        # x: (B, C, L, H, W). If C_geos=1 (pr)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # Output should be (B, Embed, 1, H', W')
        return x.squeeze(2) # Remove Time dim if it's 1

class FusionAttention(nn.Module):
    """
    Cross-Attention Fusion.
    Query: Dynamic (Flow)
    Key/Value: Physics (State)
    """
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, z_dynamic, z_physics):
        # Inputs: (B, C, H, W)
        B, C, H, W = z_dynamic.shape
        
        # Flatten spatial: (B, H*W, C)
        q = z_dynamic.flatten(2).transpose(1, 2)
        k = z_physics.flatten(2).transpose(1, 2)
        v = k 
        
        # Self-Attention or Cross? Plan says Cross.
        # Q=Dynamic looking up K=Physics.
        attn_out, _ = self.attn(q, k, v)
        
        x = self.norm(q + attn_out)
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)
        
        # Reshape to (B, C, H, W)
        return x.transpose(1, 2).reshape(B, C, H, W)

class DecoderProbability(nn.Module):
    """
    Decodes to Zero-Inflated Gamma Parameters (One per member).
    Input: (B, Embed, H_small, W_small) -> (B, 3*L, H_orig, W_orig) ??
    Or (B, 3, L, H, W)?
    We need L=4 outputs?
    The EncoderDynamic collapsed L.
    We need to reconstruct L?
    Or predicts Weekly Mean?
    The Plan says "Output parameters per lead time".
    If we collapsed L, we need to expand it back.
    Better: Use EmbedDim to carry L info or Upsample time?
    Alternative: Predict L channels in output conv.
    3 params * 4 leads = 12 channels.
    """
    def __init__(self, embed_dim=128, out_leads=4, scale_factor=16):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(embed_dim, 64, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1)
        
        # Final head: 3 params * L output channels
        self.head = nn.Conv2d(16, out_leads * 3, kernel_size=3, padding=1)
        
        self.out_leads = out_leads
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x = F.relu(self.up3(x))
        # Resize to Exact target size?? (181, 360)
        # Current logic: 16x upsample.
        # If input 11x22 -> 176x352. Close but not exact.
        # Interpolate to 181x360
        x = F.interpolate(x, size=(181, 360), mode='bilinear', align_corners=False)
        x = F.relu(self.up4(x)) # Wait, up4 was part of 16x.
        
        # Let's fix upsampling logic.
        # Just interpolate at end.
        
        out = self.head(x) # (B, 12, 181, 360)
        
        # Reshape to (B, L, 3, H, W)
        B, _, H, W = out.shape
        out = out.view(B, self.out_leads, 3, H, W)
        
        # Activations
        # p (prob rain): Sigmoid
        p_logit = out[:, :, 0, :, :]
        p = torch.sigmoid(p_logit)
        
        # alpha, beta: Softplus + epsilon
        alpha_logit = out[:, :, 1, :, :]
        beta_logit = out[:, :, 2, :, :]
        
        alpha = F.softplus(alpha_logit) + 1e-6
        beta = F.softplus(beta_logit) + 1e-6
        
        return p, alpha, beta

class CommitteeModel(nn.Module):
    def __init__(self, channels_obs=3, channels_geos=1, num_members=4):
        super().__init__()
        self.encoder_phys = EncoderPhysics(in_channels=channels_obs)
        self.encoder_dyn = EncoderDynamic(in_channels=channels_geos)
        self.fusion = FusionAttention()
        self.decoder = DecoderProbability()
        self.num_members = num_members
        
    def forward(self, x_obs, x_geos):
        """
        x_obs: (B, C_obs, H, W)
        x_geos: (B, M, C_geos, L, H, W)
        """
        B, M, Cg, L, H, W = x_geos.shape
        
        # 1. Encode Physics (Shared) - Runs ONCE
        z_phys = self.encoder_phys(x_obs)
        
        outputs = []
        
        # 2. Committee Loop
        for m in range(M):
            # Extract member: (B, C, L, H, W)
            x_mem = x_geos[:, m, :, :, :, :]
            
            # Encode Dynamic
            z_dyn = self.encoder_dyn(x_mem)
            
            # Fuse
            z_fused = self.fusion(z_dyn, z_phys)
            # Concatenate original dynamic to fused? Or residual?
            # Fusion block has residual already.
            
            # Decode
            p, alpha, beta = self.decoder(z_fused)
            
            # Stack results: (B, L, H, W)
            outputs.append((p, alpha, beta))
            
        # Stack members
        # Structure: List of tuples.
        # We want (B, M, L, H, W) for each param
        p_stack = torch.stack([o[0] for o in outputs], dim=1)
        alpha_stack = torch.stack([o[1] for o in outputs], dim=1)
        beta_stack = torch.stack([o[2] for o in outputs], dim=1)
        
        return p_stack, alpha_stack, beta_stack
