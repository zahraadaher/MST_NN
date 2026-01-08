import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class MuonDataset(Dataset):
    """
    Dataset for loading precomputed PoCA voxel grids (S, N, S_sigma)
    saved as `*_poca_voxels.npz` files, along with the corresponding
    ground-truth density volumes.

    Each .npz file is expected to contain:
        - S       : (D, D, D) scattering-density grid
        - N       : (D, D, D) hit-count grid
        - S_sigma : (D, D, D) uncertainty kernel accumulation
        - voi_min : (3,) lower bounds of volume of interest (physical units)
        - voi_max : (3,) upper bounds of volume of interest (physical units)
        - D       : integer grid size

    Returned tensors:
        x     : (3, D, D, D)
                 channels = [ S_norm , log(1+N)_norm , log(1+S_sigma)_norm ]
        tgt   : (1, D, D, D) ground truth density map
        mask  : (1, D, D, D) exposure mask (N > 0)
        name  : str, basename of the sample (without suffix)

    Args:
        poca_dir (str): directory containing *_poca_voxels.npz files.
        target_dir (str): directory with *_density.npy ground-truth files.
        normalize_stats (dict or None):
            If provided, must contain:
                {"S_max", "N_log_max", "S_sigma_log_max"}
            Otherwise normalization is computed per-sample.

    Notes:
        - All arrays are loaded as NumPy, then converted to float32.
        - Returned arrays are PyTorch-convertible (no PyTorch dependency inside).
    """

    def __init__(self, poca_dir: str, target_dir: str, normalize_stats=None):
        self.files = sorted(glob.glob(os.path.join(poca_dir, "*_poca_voxels.npz")))
        if len(self.files) == 0:
            raise RuntimeError(f"No precomputed PoCA files found in {poca_dir}")

        self.target_dir = target_dir
        self.normalize_stats = normalize_stats

        # Load metadata (D, VOI bounds) from first file
        first = np.load(self.files[0])
        self.D = int(first["D"])
        self.voi_min = first["voi_min"].astype(np.float32)
        self.voi_max = first["voi_max"].astype(np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # -------- Load precomputed PoCA voxel grids --------
        path = self.files[idx]
        name = os.path.basename(path).replace("_poca_voxels.npz", "")
        data = np.load(path)

        S       = data["S"].astype(np.float32)       # scattering density
        N       = data["N"].astype(np.float32)       # hit count
        S_sigma = data["S_sigma"].astype(np.float32) # angular sigma accumulator

        # -------- Log transforms --------
        N_log = np.log1p(N)
        S_sigma_log = np.log1p(S_sigma)

        # -------- Normalization --------
        if self.normalize_stats:
            S           = S           / (self.normalize_stats["S_max"] + 1e-9)
            N_log       = N_log       / (self.normalize_stats["N_log_max"] + 1e-9)
            S_sigma_log = S_sigma_log / (self.normalize_stats["S_sigma_log_max"] + 1e-9)
        else:
            S           /= (S.max()           + 1e-9)
            N_log       /= (N_log.max()       + 1e-9)
            S_sigma_log /= (S_sigma_log.max() + 1e-9)

        # -------- Input tensor (C, D, D, D) --------
        x = np.stack([S, N_log, S_sigma_log], axis=0).astype(np.float32)

        # -------- Ground truth density volume --------
        tgt_path = os.path.join(self.target_dir, name + "_X0.npy")
        if not os.path.exists(tgt_path):
            raise RuntimeError(f"Ground truth not found: {tgt_path}")

        tgt = np.load(tgt_path).astype(np.float32)
        
        # Avoid division by 0 in void
        eps = 1e-9
        tgt = 1.0 / (tgt + eps)
        
        # Scale to [0,1] or global normalization
        # if self.normalize_stats:
        #     tgt = tgt / (self.normalize_stats["tgt_max"] + 1e-9)
        # else:
        #     tgt = tgt / (tgt.max() + 1e-9)
            
        tgt = tgt[None, ...]  # add channel dimension: (1, D, D, D)

        # -------- Exposure mask (1, D, D, D) --------
        mask = (N > 0).astype(np.float32)#[None, ...]

        # additional target threshold mask
        target_mask = (tgt[0] > 0.1).astype(np.float32)  # tgt[0] because tgt has shape (1,D,D,D)

        # Combine masks
        mask = (mask * target_mask)[None, ...]  # add channel dimension

        # Return NumPy arrays (PyTorch DataLoader converts them to tensors automatically)
        return x, tgt, mask, name



