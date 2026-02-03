#!/usr/bin/env python3
import os
import sys
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

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

data_path = '/home/ucl/cp3/zdaher/MST_NN/train/datasets_leadInConcrete/'
out_dir = "train_outputs_lead"
n_epochs = 400
save_interval = 50

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_main():
    seed_everything(0) # Fixed seed for reproducibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #torch.backends.cudnn.benchmark = True

    if use_wandb:
        wandb.init(
            project="muon-tomography-3d-unet",
            config={
                "lr": 1e-3,
                "optimizer": "AdamW",
                "scheduler": "ReduceLROnPlateau",
                "epochs": n_epochs,
                "batch_size": 4,
            }
        )

    train_dataset = MuonDataset(
        poca_dir=f"{data_path}/train/poca_voxels",
        target_dir=f"{data_path}/train/density_maps",
    )

    val_dataset = MuonDataset(
        poca_dir=f"{data_path}/val/poca_voxels",
        target_dir=f"{data_path}/val/density_maps",
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = ProbUNet3D(
        in_channels=3,
        out_channels=2,
        base_features=16,
        depth=3,
        use_resblock = False
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    if use_wandb:
        wandb.watch(model, log="all", log_freq=200)

    os.makedirs(out_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    early_stop_patience = 10
    epochs_since_improvement = 0
    min_delta = 0.0

    for epoch in range(1, n_epochs + 1):

        model.train()
        train_loss = 0.0

        for batch_idx, (x, y, mask, base) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(x)
             # --- Warmup: MSE on Density Only ---
            # For the first 10 epochs, force model to learn the MEAN efficiently.
            # NLL can be unstable if mean is far off.
            if epoch <= 10:
                # pred[:,0] is log(density). y.log() is log(density).
                # Simple MSE on the log-values.
                mse_loss = F.mse_loss(pred[:, 0], y.log()[:, 0], reduction='none')
                loss = (mse_loss * mask[:, 0]).sum() / (mask.sum() + 1e-8)
            else:
                loss = nll_loss_masked(pred, y.log(), mask)
            # loss = nll_loss_masked(pred, y.log(), mask)
            # pred, kl = model(x, y)
            # recon = F.mse_loss(pred, y)
            # loss = recon + kl.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y, mask, base in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                pred = model(x)
                loss = nll_loss_masked(pred, y.log(), mask)
                #pred, kl = model(x, y)
                #recon = F.mse_loss(pred, y)
                #loss = recon

                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch
            })

            if epoch % 1 == 0:
                mu     = pred[:, 0:1]
                logsig = pred[:, 1:2]
                sigma  = logsig.exp()
            
                fig = plot_slices(mu, sigma, y, batch_idx=0, axis="z", n_slices=4)
            
                wandb.log({"slices": wandb.Image(fig), "epoch": epoch})
                print('here')
             


        print(
            f"Epoch {epoch}/{n_epochs}  "
            f"Train: {train_loss:.5f}  "
            f"Val: {val_loss:.5f}  "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_since_improvement = 0

            torch.save(
                {"model_state_dict": model.state_dict()},
                os.path.join(out_dir, "best_model.pt")
            )
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= early_stop_patience:
                print("Early stopping.")
                break

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

    torch.save(
        {"train_losses": train_losses, "val_losses": val_losses},
        os.path.join(out_dir, "loss_history.pt")
    )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train_main()

