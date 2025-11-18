#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
import torch
from typing import Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..")))

from src.poca_reconstruction import POCA

csv_dir    = "/home/ucl/cp3/zdaher/simulation-cultural-heritage/muography-geant4/outputs/boneInConcrete/processed_root"
target_dir = "/home/ucl/cp3/zdaher/simulation-cultural-heritage/muography-geant4/outputs/boneInConcrete/density_maps"
out_dir    = "/home/ucl/cp3/zdaher/simulation-cultural-heritage/muography-geant4/outputs/boneInConcrete/poca_voxels"

sigma_x=0.1
sigma_y=0.1
xyz_min=(-600., -600., -2700.)
xyz_max=( 600.,  600., -1300.)

device= "cuda" if torch.cuda.is_available() else 'cpu'

def precompute_all_poca(
    csv_dir: str,
    target_dir: str,
    out_dir: str,
    sigma_x: float = 0.1,
    sigma_y: float = 0.1,
    xyz_min: Tuple = (-600., -600., -2700.),
    xyz_max: Tuple = ( 600.,  600., -1300.),
    device: str = "cuda" if torch.cuda.is_available() else 'cpu'
):
    os.makedirs(out_dir, exist_ok=True)
    
    # match csv/target maps
    all_csv = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    pairs = []

    for c in all_csv:
        base = os.path.basename(c).replace("_processed.csv", "")
        tgt = os.path.join(target_dir, base + "_density.npy")
        if os.path.exists(tgt):
            pairs.append((c, tgt))

    if len(pairs) == 0:
        raise RuntimeError("No CSV-density pairs found.")

    print(f"Found {len(pairs)} samples.")

    # Use first density file to read voxel grid size D
    sample = np.load(pairs[0][1])
    D = sample.shape[0]
    print(f"Detected voxel grid size: D={D}")
    print(f"VOI: min={xyz_min}, max={xyz_max}")

    # Loop over all samples
    for i, (csv_path, tgt_path) in enumerate(pairs):

        base = os.path.basename(csv_path).replace("_processed.csv", "")
        out_path = os.path.join(out_dir, base + "_poca_voxels.npz")

        if os.path.exists(out_path):
            print(f"[{i+1}/{len(pairs)}] {base}: already exists → skip")
            continue

        print(f"[{i+1}/{len(pairs)}] {base}: computing...")

        # Create POCA solver for this CSV
        poca_solver = POCA(
            csv_path,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            xyz_min=xyz_min,
            xyz_max=xyz_max,
            device=device
        )

        # Use class method to compute + voxelize + save
        poca_solver.precompute_poca_voxels(
            target_path=tgt_path,
            out_path=out_path,
        )

        print(f"Saved: {out_path}")

    print("✓ All POCA precomputations completed.")

# main
if __name__ == "__main__":
    precompute_all_poca(
        csv_dir=csv_dir,
        target_dir=target_dir,
        out_dir=out_dir,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        device=device,
    )

