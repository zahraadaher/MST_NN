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



