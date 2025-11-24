# Muon Scattering Tomography – Probabilistic 3D Reconstruction Network
*A full pipeline from Geant4 → POCA processing → voxelization → 3D probabilistic U-Net.*

---

## Overview

This repository implements a complete end-to-end pipeline for **Muon Scattering Tomography (MST)**:

I. **Takes in raw GEANT4 detector hits**  
II. **First level PoCA reconstruction**  
III. **3D voxelization of PoCA input features**  
IV. **Probabilistic 3D U-Net** for inverse reconstruction of material properties.

The network predicts per-voxel **Gaussian likelihood parameters**:

$$
p(y \mid x) = \mathcal{N}(\mu(x), \sigma(x)^2)
$$

This enables uncertainty-aware reconstruction and improves material discrimination in MST.

---

## I. Input Data: Geant4 Simulation

Raw muon hits are generated using a dedicated **GEANT4 muon tomography framework**:
 
<https://cp3-git.irmp.ucl.ac.be/muographycp3/simulation-cultural-heritage>

Each simulation outputs CSV files with ture, noise-free hit coordinates from the 6 detector planes and the associated ground truth voxelized map of the VOI.

---

## 2. POCA Reconstruction

For every muon, we compute:

- **Incoming track** from upper detector hits 
- **Outgoing track** from lower detector hits
- **PoCA point** (point of closest approach) from the two tracks
- **Scattering angle** θ  

Noise in the detector hits is simulated by injecting a Gaussian smearing with a user-defined transverse resolution in x and y.
Uncertainties on PoCA variables are estimated using gradient-based error propagation with PyTorch automatic differentiation.

---

## 3. Voxelization Pipeline

PoCA features are mapped into **3D voxel grids** of the input channels to the model. 
The PoCA positional uncertainties define a 3D Gaussian kernel to populate weighted contributions of each PoCA point in the voxels covered by its pdf.
The following input channels are defined:

1. `S`: Scattering density (θ²).

2. `N`: Muon count / occupancy

3. `Sσ`: Scattering uncertainty


---

## 4. Neural Network: Probabilistic 3D U-Net

The model takes 3 channels:

- `S`  
- `log(N + 1)`  
- `log(Sσ + 1)`  

and predicts 2 output channels:

- **μ(x)**: predicted material property for voxel x
- **logσ(x)**: predicted uncertainty for voxel x

### Architecture Highlights
- 3D U-Net with skip connections  
- Downsampling via `MaxPool3d(2)`  
- Upsampling via trilinear interpolation (avoids checkerboard artifacts)  
- Final `Conv3d` outputs 2 channels per voxel

The model implements a Gaussian NLL loss function for probabbilistic regression.

---



## About the Custom PyTorch Wheel

This project is intended to run on a **computing cluster node** where the system libraries
(GLIBC, CUDA runtime, NVIDIA drivers) are older than what official PyTorch builds expect.
Installing PyTorch from Conda or pip on these nodes leads to errors such as:

- `ImportError: GLIBC_2.27 not found`
- missing or incompatible CUDA libraries
- failure to load `torch._C`

To avoid these issues, this repository provides a **custom PyTorch 2.6.0 wheel**
that is pre-built to:

- work with the cluster’s older GLIBC version  
- match the CUDA runtime available on the cluster  
- avoid dependency conflicts with Conda’s CUDA packages

This wheel is stored via **Git LFS** and is downloaded automatically when cloning the repo
(after running `git lfs install`).

---

## Installation Guide

This project requires **Git LFS**, **Conda**, and the included **custom PyTorch wheel**.
Follow the steps below depending on your operating system.

### 1. Install Git LFS (required to download the custom PyTorch wheel)
1. Download Git LFS: https://git-lfs.github.com/
2. Run the installer.
3. In terminal, run:
```bash
git lfs install
```
### 2. Clone the repository
```bash
git clone https://github.com/zahraadaher/MST_NN.git
cd MST_NN
```
### 3. Create the Conda environement
```bash
conda env create -f environment.yml
conda activate mst
```
## Running Jupyter Notebook

When connecting to the cluster, forward port 8888:
```bash
ssh -L 8888:localhost:8888 user@cluster
```
Then start Jupyter:
```bash
jupyter lab --no-browser --port=8888 > jupyter.log 2>&1 &
```
`--no-browser` → prevents browser launch

`--port=8888` → matches your SSH port forwarding

`> jupyter.log 2>&1` → sends stdout + stderr to jupyter.log

`&` → run in background

When you start Jupyter, it prints a URL that contains an **access token**. You must copy–paste that entire URL into your local browser. 

Open in your local browser:
```bash
http://localhost:8888
```




