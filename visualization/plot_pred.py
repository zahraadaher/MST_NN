import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_slices(mu, sigma, target, batch_idx=0, axis="z", n_slices=4):
    """
    mu, sigma, target: tensors of shape [B, 1, D, D, D] or [B, D, D, D]
    """

    # --- Select sample ---
    mu_b     = mu[batch_idx].detach().cpu().squeeze()
    sigma_b  = sigma[batch_idx].detach().cpu().squeeze()
    target_b = target[batch_idx].detach().cpu().squeeze()

    if mu_b.ndim != 3:
        raise ValueError(f"Expected (D,D,D), got {mu_b.shape}")

    D = mu_b.shape[0]

    # --- Choose slice indices ---
    idxs = np.linspace(0, D-1, n_slices).astype(int)

    # --- Compute global vmin/vmax for each channel ---
    mu_min,     mu_max     = float(mu_b.min()),     float(mu_b.max())
    sig_min,    sig_max    = float(sigma_b.min()),  float(sigma_b.max())
    tgt_min,    tgt_max    = float(target_b.min()), float(target_b.max())

    # --- Prepare figure ---
    fig, axs = plt.subplots(n_slices, 3, figsize=(13, 4*n_slices))

    for row, idx in enumerate(idxs):

        # --- Extract slice ---
        if axis == "z":
            s_mu     = mu_b[:, :, idx]
            s_sigma  = sigma_b[:, :, idx]
            s_target = target_b[:, :, idx]
        elif axis == "y":
            s_mu     = mu_b[:, idx, :]
            s_sigma  = sigma_b[:, idx, :]
            s_target = target_b[:, idx, :]
        elif axis == "x":
            s_mu     = mu_b[idx, :, :]
            s_sigma  = sigma_b[idx, :, :]
            s_target = target_b[idx, :, :]
        else:
            raise ValueError("axis must be x, y, or z")

        # --- μ ---
        im0 = axs[row, 0].imshow(s_mu.numpy(), cmap="viridis",
                                 vmin=mu_min, vmax=mu_max)
        axs[row, 0].set_title(f"μ (slice {idx})")
        fig.colorbar(im0, ax=axs[row, 0], fraction=0.046, pad=0.04)

        # --- σ ---
        im1 = axs[row, 1].imshow(s_sigma.numpy(), cmap="viridis",
                                 vmin=sig_min, vmax=sig_max)
        axs[row, 1].set_title("σ")
        fig.colorbar(im1, ax=axs[row, 1], fraction=0.046, pad=0.04)

        # --- target ---
        im2 = axs[row, 2].imshow(s_target.numpy(), cmap="viridis",
                                 vmin=tgt_min, vmax=tgt_max)
        axs[row, 2].set_title("target")
        fig.colorbar(im2, ax=axs[row, 2], fraction=0.046, pad=0.04)

        for col in range(3):
            axs[row, col].axis("off")

    plt.tight_layout()
    return fig
