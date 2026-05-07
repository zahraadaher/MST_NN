import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Residual Block: Conv3d -> BN -> LeakyReLU -> Conv3d -> BN -> Sum -> LeakyReLU
    Helps prevents vanishing gradients in deeper networks.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.act(out)

class ProbUNet3D(nn.Module):
    """
    Probabilistic 3D U-Net for muography.
    Can operate in "Residual" mode (Physics-Grounded, LeakyReLU) or "Standard" mode (ReLU).
    
    Args:
        in_channels (int): Input channels (default: 3)
        out_channels (int): Output channels (default: 2 -> log-density, log-sigma)
        base_features (int): Number of features in first layer (default: 32)
        depth (int): Number of downsampling steps (default: 3 for 24^3 grid)
        use_resblock (bool): If True, uses ResBlocks + LeakyReLU. If False, uses Standard Conv + ReLU.
    """
    def __init__(self, in_channels=3, out_channels=2, base_features=16, depth=3, use_resblock=False):
        super().__init__()
        self.depth = depth
        self.use_resblock = use_resblock
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        
        feats = base_features
        in_ch = in_channels
        
        # Encoder
        for i in range(depth):
            block = self._make_block(in_ch, feats)
            self.downs.append(block)
            in_ch = feats
            feats *= 2

        # Bottleneck
        if use_resblock:
            self.bottleneck = nn.Sequential(
                nn.Conv3d(in_ch, feats, 3, padding=1, bias=False),
                nn.BatchNorm3d(feats),
                nn.LeakyReLU(0.1, inplace=True),
                ResBlock(feats),
                ResBlock(feats)
            )
        else:
            # Standard bottleneck: Double Conv (LeakyReLU)
            self.bottleneck = nn.Sequential(
                nn.Conv3d(in_ch, feats, 3, padding=1),
                nn.BatchNorm3d(feats),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv3d(feats, feats, 3, padding=1),
                nn.BatchNorm3d(feats),
                nn.LeakyReLU(0.1, inplace=True)
            )

        # Decoder
        self.ups = nn.ModuleList()
        
        for i in range(depth):
            enc_feats = feats // 2
            out_ch = enc_feats if i < depth - 1 else base_features
            
            # Input to block will be `feats + enc_feats`.
            self.ups.append(nn.ModuleDict({
                "conv": self._make_block(feats + enc_feats, out_ch)
            }))
            feats = enc_feats 

        self.final_conv = nn.Conv3d(base_features, out_channels, 1)
        
        #nn.init.constant_(self.final_conv.bias[0], 1.7) # -2.0 worked 
        with torch.no_grad():
            # Set the 'mean' head bias to 1.7 (the average of 0.1, 1.8, 3.2 targets)
            self.final_conv.bias[0] = 1.7
            self.final_conv.bias[1] = -0.3 

    def _make_block(self, in_c, out_c):
        if self.use_resblock:
            # ResBlock strategy
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.LeakyReLU(0.1, inplace=True),
                ResBlock(out_c)
            )
        else:
            # Standard U-Net strategy
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.LeakyReLU(0.1, inplace=True)
            )

    def forward(self, x):
        enc_feats = []
        out = x
        
        # Encoder
        for down in self.downs:
            out = down(out)
            enc_feats.append(out)
            out = self.pool(out)
            
        # Bottleneck
        out = self.bottleneck(out)
        
        # Decoder
        for i, up_modules in enumerate(self.ups):
            # 1. Upsample
            out = F.interpolate(out, scale_factor=2, mode="trilinear", align_corners=False)
            
            # 2. Skip Connection
            skip = enc_feats[-(i + 1)]
            
            if skip.shape[-3:] != out.shape[-3:]:
                diffZ = skip.size(2) - out.size(2)
                diffY = skip.size(3) - out.size(3)
                diffX = skip.size(4) - out.size(4)
                out = F.pad(out, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2,
                                  diffZ // 2, diffZ - diffZ // 2])
            
            # 3. Concatenate
            out = torch.cat([out, skip], dim=1)
            
            # 4. Conv Block
            out = up_modules["conv"](out)
            
        preds = self.final_conv(out)
        
        predicted_density = preds[:, 0:1]
        predicted_log_sigma2   = preds[:, 1:2]
        
        return torch.cat([predicted_density, predicted_log_sigma2], dim=1)

def nll_loss_masked(pred, target, mask):
    """
    pred:   (B,2,D,D,D) -> [mu, log_sigma]
    target: (B,1,D,D,D)
    mask:   (B,1,D,D,D) in {0,1}
    We assume Y|X ~ N(mu, sigma^2) and maximize log-likelihood
    (i.e. minimize negative log-likelihood).
    """
    mu        =pred[:, 0:1]
    log_sigma_2 = pred[:, 1:2]

    sigma_2     = torch.exp(log_sigma_2)

    diff = target - mu

    # per-voxel Gaussian NLL
    nll = 0.5 * np.log(2 * math.pi) + 0.5 * log_sigma_2 + 0.5 * diff**2  * torch.exp(-log_sigma_2)

    # weight target voxels 
    weight = torch.where(target > 0.1, 10.0, 0.0)

    # beta-variance scale
    scale = sigma_2.detach()  ** 0.5

    # apply exposure mask
    masked_nll = nll * mask * weight * scale
    monitor_loss =  nll * mask * weight 
    n_masked   = (weight*mask).sum()
    
    # Avoid division by 0 if mask is empty
    if n_masked == 0:
        base_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
    else:
        base_loss = masked_nll.sum() / n_masked
        monitor_loss = monitor_loss.sum() / n_masked

    return base_loss, monitor_loss


