#!/usr/bin/env python3
import os
import sys
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import logging

use_wandb = False

if use_wandb:
    import wandb
else:
    wandb = None

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..")))

from src.data import MuonDataset
from src.model import ProbUNet3D, nll_loss_masked
from visualization.plot_pred import plot_slices
from utils.params import Params
from utils.logging import setup_logging

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def center_crop(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Crop pred to match target spatial dimensions, symmetrically.
    Assumes pred and target are [B, C, D, H, W].
    """
    _, _, Dp, Hp, Wp = pred.shape
    _, _, Dt, Ht, Wt = target.shape

    # Compute cropping indices
    d1 = (Dp - Dt) // 2
    d2 = d1 + Dt

    h1 = (Hp - Ht) // 2
    h2 = h1 + Ht

    w1 = (Wp - Wt) // 2
    w2 = w1 + Wt

    return pred[:, :, d1:d2, h1:h2, w1:w2]

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for x, y, mask, _ in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        optimizer.zero_grad()
        pred = model(x)

        # crop pred to match y
        pred_cropped = center_crop(pred, y)
        mask = center_crop(mask, y)

        if epoch <= -10:
            mse = F.mse_loss(pred_cropped[:, 0], y.log()[:, 0], reduction="none")
            loss = (mse * mask[:, 0]).sum() / (mask.sum() + 1e-8)
        else:
            loss = nll_loss_masked(pred_cropped, y, mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y, mask, _ in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            pred = model(x)
            pred= center_crop(pred, y)
            mask = center_crop(mask, y)
            loss = nll_loss_masked(pred, y, mask)
            total_loss += loss.item()

    return total_loss / len(loader)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON config file")
    args = parser.parse_args()

    params = Params.from_json(args.config)

    # Create output directory
    params.out_dir.mkdir(parents=True, exist_ok=True)

    # Save copy of config used in this run
    params.save(params.out_dir / "used_config.json")

    # Fix seed for reproducibility
    seed_everything(params.seed) 

    # setup loggig
    setup_logging(params.out_dir)
    logging.info(f"Loaded config from {args.config}")
    logging.info(f"Parameters: {params}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #torch.backends.cudnn.benchmark = True

    # Datasets
    train_dataset = MuonDataset(
        poca_dir=params.data_path / "train/poca_voxels",
        target_dir=params.data_path / "train/density_maps",
    )

    val_dataset = MuonDataset(
        poca_dir=params.data_path / "val/poca_voxels",
        target_dir=params.data_path / "val/density_maps",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size_train,
        shuffle=True,
        num_workers=params.num_workers_train,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=params.batch_size_val,
        shuffle=False,
        num_workers=params.num_workers_val,
        pin_memory=True,
    )

    # Model
    model = ProbUNet3D(
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        base_features=params.base_features,
        depth=params.depth,
        use_resblock=params.use_resblock,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=params.factor, patience=params.patience
    )

    params.out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, params.n_epochs + 1):

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)

        scheduler.step(val_loss)

        logging.info(
            f"Epoch {epoch:03d} | "
            f"Train {train_loss:.5f} | "
            f"Val {val_loss:.5f} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                model.state_dict(),
                params.out_dir / "best_model.pt",
            )
            logging.info("Saved new best model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= params.early_stop_patience:
                logging.info("Early stopping triggered.")
                break

        # Periodic checkpoint
        if epoch % params.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                params.out_dir / f"checkpoint_epoch{epoch}.pt",
            )

    logging.info("Training complete.")


if __name__ == "__main__":
    main()
