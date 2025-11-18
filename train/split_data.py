#!/usr/bin/env python3
import os
import glob
import shutil
import numpy as np
import sys


# ============================================================
# Edit these
# ============================================================
POCA_DIR   = "/home/ucl/cp3/zdaher/simulation-cultural-heritage/muography-geant4/outputs/boneInConcrete/poca_voxels"
TARGET_DIR = "/home/ucl/cp3/zdaher/simulation-cultural-heritage/muography-geant4/outputs/boneInConcrete/density_maps"
OUT_DIR    = "datasets_boneInConcrete"

TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1
SEED       = 2025
# ============================================================


def usage_error(msg):
    """Print error + instructions and exit."""
    print(f"\n❌Error: {msg}\n")
    print("Please edit the paths at the top of split_dataset.py:\n")
    print("    POCA_DIR   = '/path/to/poca_voxels'")
    print("    TARGET_DIR = '/path/to/density_maps'")
    print("    OUT_DIR    = 'dataset_boneInConcrete'\n")
    sys.exit(1)


def main():

    # ------------------------------------
    # Validate settings
    # ------------------------------------
    if not os.path.isdir(POCA_DIR):
        usage_error(f"POCA_DIR does not exist: {POCA_DIR}")

    if not os.path.isdir(TARGET_DIR):
        usage_error(f"TARGET_DIR does not exist: {TARGET_DIR}")

    if not (0 < TRAIN_FRAC < 1):
        usage_error("TRAIN_FRAC must be between 0 and 1")

    if not (0 < VAL_FRAC < 1):
        usage_error("VAL_FRAC must be between 0 and 1")

    if TRAIN_FRAC + VAL_FRAC >= 1:
        usage_error("TRAIN_FRAC + VAL_FRAC must be < 1")

    # ------------------------------------
    # Create output folders
    # ------------------------------------
    for split in ["train", "val", "test"]:
        os.makedirs(f"{OUT_DIR}/{split}/poca_voxels", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/{split}/density_maps", exist_ok=True)

    # ------------------------------------
    # Gather POCA files
    # ------------------------------------
    files = sorted(glob.glob(os.path.join(POCA_DIR, "*_poca_voxels.npz")))
    N = len(files)

    if N == 0:
        usage_error(f"No *_poca_voxels.npz files found in {POCA_DIR}")

    print(f"Found {N} samples in POCA_DIR.")

    # ------------------------------------
    # Deterministic shuffle
    # ------------------------------------
    rng = np.random.default_rng(SEED)
    rng.shuffle(files)

    # ------------------------------------
    # Compute splits
    # ------------------------------------
    n_train = int(N * TRAIN_FRAC)
    n_val   = int(N * VAL_FRAC)
    n_test  = N - n_train - n_val

    print(f"Splits: train={n_train}, val={n_val}, test={n_test}")

    splits = {
        "train": files[:n_train],
        "val":   files[n_train:n_train+n_val],
        "test":  files[n_train+n_val:]
    }

    # ------------------------------------
    # Copy files
    # ------------------------------------
    for split, split_files in splits.items():
        print(f"\nCopying {split}: {len(split_files)} files")
        for p in split_files:
            base = os.path.basename(p).replace("_poca_voxels.npz", "")
            out_poca = f"{OUT_DIR}/{split}/poca_voxels/{base}_poca_voxels.npz"
            shutil.copy(p, out_poca)

            dens_src = f"{TARGET_DIR}/{base}_density.npy"
            if not os.path.exists(dens_src):
                usage_error(f"Missing target file: {dens_src}")

            dens_dst = f"{OUT_DIR}/{split}/density_maps/{base}_density.npy"
            shutil.copy(dens_src, dens_dst)

    print("\n✓ Dataset split complete.\n")


if __name__ == "__main__":
    main()

