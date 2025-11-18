#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..")))

from src.data import MuonDataset 
from src.model import ProbUNet3D, nll_loss_masked 

data_path = '/home/ucl/cp3/zdaher/MuProbNet/dataset_boneInConcrete/'
out_dir = "train_outputs" # to save trained model state 
n_epochs = 20
save_interval = 2 # interval of n_epochs for checkpointing model state 

def train_main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    # ================ preparing datasets and loaders =================

    # train dataset
    train_dataset = MuonDataset(
    poca_dir   = f"{data_path}/train/poca_voxels",
    target_dir = f"{data_path}/train/density_maps",
    )
    # val dataset
    val_dataset = MuonDataset(
        poca_dir   = f"{data_path}/val/poca_voxels",
        target_dir = f"{data_path}/val/density_maps",
    )
    
    print("Loading dataset...")
    # train loader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                          num_workers=4,pin_memory=True)
    # val loader
    val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False,
                              num_workers=2,pin_memory=True)

    # ============== building model and optimization strategy ===========
    
    # build model
    model = ProbUNet3D(
        in_channels=3,       # S, N_log, S_sigma_log
        out_channels=2,      # μ, logσ
        base_features=16,
        depth=4
    ).to(device)

    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # define scheduling strategy for learning rates
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # ================ training loop =======================================

    os.makedirs(out_dir, exist_ok=True)

    train_losses = []
    val_losses   = []


    print("\n===============================")
    print("Starting training...")
    print("===============================\n")

    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):

        # ======================================================
        # TRAINING
        # ======================================================
        model.train()
        train_loss = 0
        n_batches = 0

        for batch_idx, (x, y, mask, base) in enumerate(train_loader):

            x    = x.to(device, non_blocking=True)
            y    = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # First batch diagnostics
            if epoch == 1 and batch_idx == 0:
                print("\n=== First batch diagnostics ===")
                print(f"Input:   {x.shape}")
                print(f"Target:  {y.shape}")
                print(f"Mask:    {mask.shape}")
                print(f"Input range:  {x.min().item():.4f} → {x.max().item():.4f}")
                print(f"Target range: {y.min().item():.4f} → {y.max().item():.4f}")
                print("="*40 + "\n")

            optimizer.zero_grad()
            pred = model(x)

            loss = nll_loss_masked(pred, y, mask)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(1, n_batches)


        # ======================================================
        # VALIDATION
        # ======================================================
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for x, y, mask, base in val_loader:

                x    = x.to(device, non_blocking=True)
                y    = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                pred = model(x)
                loss = nll_loss_masked(pred, y, mask)

                val_loss += loss.item()
                val_batches += 1

                log_sigma = pred[:,1:2]  # (B,1,D,D,D)
       

        val_loss /= max(1, val_batches)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:2d}/{n_epochs} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.3e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state_dict": model.state_dict()},
                os.path.join(out_dir, "best_model.pt")
            )

        # Periodic checkpoint
        if epoch % save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                os.path.join(out_dir, f"checkpoint_epoch{epoch}.pt")
            )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    print("\n Training complete.")
    print(f"Best val loss: {best_val_loss:.5f}")
    print(f"Saved best model to {out_dir}/best_model.pt")
    torch.save(
    {
        "train_losses": train_losses,
        "val_losses": val_losses
    },
    os.path.join(out_dir, "loss_history.pt")
)

    print(f"\nSaved loss history to {out_dir}/loss_history.pt")


# ======================================================
# main
# ======================================================
if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)
    train_main()

