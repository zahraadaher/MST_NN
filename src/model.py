import os
import glob
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ProbUNet3D(nn.Module):
    """
    Probabilistic 3D U-Net for muography
    with MuonDataset (S, N_log, S_sigma_log) inputs and
    Gaussian heteroskedastic regression p(Y|X) = N(mu, sigma^2).
    """
    def __init__(self, in_channels=3, out_channels=2, base_features=32, depth=4):
        super().__init__()
        self.depth = depth
        self.downs = nn.ModuleList()
        feats = base_features
        in_ch = in_channels
        for _ in range(depth):
            self.downs.append(nn.Sequential(
                nn.Conv3d(in_ch, feats, 3, padding=1),
                nn.BatchNorm3d(feats),
                nn.ReLU(),
                nn.Conv3d(feats, feats, 3, padding=1),
                nn.BatchNorm3d(feats),
                nn.ReLU()
            ))
            in_ch = feats
            feats *= 2

        self.pool = nn.MaxPool3d(2)

        self.ups = nn.ModuleList()
        feats = feats // 2
        for i in range(depth):
            in_ch = feats * 2
            out_ch = feats // 2 if i < depth - 1 else base_features
            self.ups.append(nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU()
            ))
            feats = out_ch

        self.final_conv = nn.Conv3d(base_features, out_channels, 1)

    def forward(self, x):
        enc_feats = []
        out = x
        for down in self.downs:
            out = down(out)
            enc_feats.append(out)
            out = self.pool(out)
        for i, up in enumerate(self.ups):
            out = F.interpolate(out, scale_factor=2, mode="trilinear", align_corners=False)
            skip = enc_feats[-(i + 1)]
            if skip.shape[-3:] != out.shape[-3:]:
                diffZ = skip.size(2) - out.size(2)
                diffY = skip.size(3) - out.size(3)
                diffX = skip.size(4) - out.size(4)
                out = F.pad(out, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2,
                                  diffZ // 2, diffZ - diffZ // 2])
            out = torch.cat([out, skip], dim=1)
            out = up(out)
        return self.final_conv(out)

def nll_loss_masked(pred, target, mask, min_sigma=1e-3, sigma_prior_strength=1e-4):
    """
    pred:   (B,2,D,D,D) -> [mu, log_sigma]
    target: (B,1,D,D,D)
    mask:   (B,1,D,D,D) in {0,1}
    We assume Y|X ~ N(mu, sigma^2) and maximize log-likelihood
    (i.e. minimize negative log-likelihood).
    """
    mu        = pred[:, 0:1]
    log_sigma = pred[:, 1:2]

    # clamp log_sigma to avoid numerical blow-up
    log_sigma = torch.clamp(log_sigma, -10.0, 5.0)
    sigma     = torch.exp(log_sigma) + min_sigma

    diff = target - mu

    # per-voxel Gaussian NLL
    nll = 0.5 * torch.log(2 * math.pi * sigma**2) + 0.5 * (diff**2) / (sigma**2)

    # apply exposure mask
    masked_nll = nll * mask
    n_masked   = mask.sum() + 1e-8
    base_loss  = masked_nll.sum() / n_masked

    # optional weak prior: penalize too-large sigma
    # encourages log_sigma ~ 0 (i.e. sigma ~ 1 in normalized units)
    sigma_prior = sigma_prior_strength * (log_sigma**2).mean()

    return base_loss + sigma_prior



