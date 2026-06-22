#!/usr/bin/env python3

"""
Data preprocessing and splitting script for the muography of triplet material (scenario of two small cubes in a larger host cube)
This script:
1. Scans density maps and CSV files based on materials in file names
2. Splits data into train/val/test based on material triplets
3. Performs PoCA reconstruction with an augmentation factor per trial, using a hit spatial smearing (0.1mm default)
    (Check the xyz limits and modify the defaults if needed)
4. Organizes output into structured directories for training
"""

import os
import sys
import glob
import re
import argparse
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR, "..")))

try:
    from src.poca_reconstruction import POCA
except Exception as e:
    print(f"Failed to import POCA: {e}", flush=True)
    raise

def parse_material_triplet(filename):
    pattern = r'H_G4_(.+?)_B1_G4_(.+?)_B2_G4_([^_]+)'
    match = re.search(pattern, filename)
    if match:
        return tuple(match.groups())
    return None

def assign_split(material_triplet):
    val_configs = {
        ('CONCRETE', 'Al', 'Al'),
        ('Be', 'Al', 'Al'),
        ('CONCRETE', 'Pb', 'Pb'),
    }
    test_configs = {
        ('CONCRETE', 'U', 'U'),
        ('Be', 'U', 'U'),
        ('Be', 'Pb', 'Pb'),
    }
    if material_triplet in val_configs:
        return 'val'
    elif material_triplet in test_configs:
        return 'test'
    else:
        return 'train'

def find_csv_files(density_path, csv_dir):
    base_name = os.path.basename(density_path).replace('_X0.npz', '')
    pattern = os.path.join(csv_dir, f"{base_name}_T*_processed.csv")
    return sorted(glob.glob(pattern))

def process_csv(csv_path, density_path, augment_factor, xyz_min, xyz_max, sigma_x, sigma_y, output_dir, split):
    """Process all augmentations for a single CSV file using one POCA engine"""
    try:
        print(f"\nCreating POCA engine for CSV: {os.path.basename(csv_path)}", flush=True)
        poca_engine = POCA(
            csv_path=csv_path,
            xyz_min=tuple(xyz_min),
            xyz_max=tuple(xyz_max),
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            device='cpu'
        )
        print(f"POCA engine ready: {poca_engine.B} muons", flush=True)
    except Exception as e:
        print(f"Failed to create POCA engine for {csv_path}: {e}", flush=True)
        return

    for aug_idx in range(augment_factor):
        trial_name = os.path.basename(csv_path).replace('_processed.csv', '')
        output_name = f"{trial_name}_aug{aug_idx:02d}"
        poca_output_path = os.path.join(output_dir, split, 'poca_voxels', f"{output_name}")
        if os.path.exists(os.path.join(output_dir, split, 'poca_voxels', f"{output_name}.npz")):
            print(f"  Skipping existing {output_name}", flush=True)
            continue
        print(f"  Starting augmentation {aug_idx}", flush=True)
        start = time.time()
        try:
            poca_engine.precompute_poca_voxels(target_path=density_path, out_path=poca_output_path)
            elapsed = time.time() - start
            print(f"[Complete] Aug {aug_idx} done in {elapsed:.2f}s", flush=True)
        except Exception as e:
            print(f"[Failed] Aug {aug_idx} failed: {e}", flush=True)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--density_dir', type=str, required=True)
    parser.add_argument('--csv_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--augment_factor', type=int, default=10)
    parser.add_argument('--xyz_min', nargs=3, type=float, default=[-600, -600, -1600])
    parser.add_argument('--xyz_max', nargs=3, type=float, default=[600, 600, -300])
    parser.add_argument('--sigma_x', type=float, default=0.1)
    parser.add_argument('--sigma_y', type=float, default=0.1)
    parser.add_argument('--limit', type=int, default=None, help='Limit number of csv files (for testing purposes)')

    args = parser.parse_args()

    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.output_dir, split, 'poca_voxels'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, split, 'density_maps'), exist_ok=True)

    # Find all density maps
    density_files = glob.glob(os.path.join(args.density_dir, '*_X0.npz'))
    print(f"Found {len(density_files)} density maps", flush=True)

    density_symlinks = {}
    csv_tasks = []

    for density_path in density_files:
        material_triplet = parse_material_triplet(os.path.basename(density_path))
        if material_triplet is None:
            continue

        split = assign_split(material_triplet)
        if split not in density_symlinks:
            density_symlinks[split] = set()
        density_symlinks[split].add(density_path)

        csv_files = find_csv_files(density_path, args.csv_dir)
        for csv_path in csv_files:
            csv_tasks.append((csv_path, density_path, split))

    if args.limit:
        csv_tasks = csv_tasks[:args.limit]

    print(f"Total CSV tasks: {len(csv_tasks)}", flush=True)

    start_time = time.time()
    for i, (csv_path, density_path, split) in enumerate(csv_tasks, 1):
        print(f"\n[{i}/{len(csv_tasks)}] Processing CSV: {os.path.basename(csv_path)}", flush=True)
        process_csv(
            csv_path,
            density_path,
            args.augment_factor,
            args.xyz_min,
            args.xyz_max,
            args.sigma_x,
            args.sigma_y,
            args.output_dir,
            split
        )

    print(f"\nTotal processing time: {(time.time() - start_time)/60:.1f} min", flush=True)

    # Create symlinks for density maps
    for split, density_paths in density_symlinks.items():
        for density_path in density_paths:
            link_path = os.path.join(args.output_dir, split, 'density_maps', os.path.basename(density_path))
            if not os.path.exists(link_path):
                os.symlink(os.path.abspath(density_path), link_path)
    print("Density symlinks created", flush=True)


if __name__ == "__main__":
    main()
