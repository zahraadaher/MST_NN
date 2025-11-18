import numpy as np
from typing import Tuple, Optional
import torch
from torch import Tensor
from torch.func import vmap, jacrev
import matplotlib.pyplot as plt 

class POCA:
    """
    Reconstructs Points of Closest Approach (PoCA) and scattering angles from
    detector hit positions stored in a CSV file. A Gaussian smearing is applied
    to the transverse hit coordinates before fitting straight-line tracks.

    The incoming and outgoing tracks are fitted analytically using the smeared
    hit coordinates. The PoCA point is obtained from the closest points between
    the two fitted lines. Uncertainties on the PoCA coordinates and on the
    scattering angle are computed by automatic differentiation and standard
    Gaussian error propagation.

    Args:
        csv_path (str): path to CSV containing hit information in the upper and lower detector panels
        xyz_min (Tuple): Lower xyz ranges (in mm)
        xyz_max (Tuple): Upper xyz ranges (in mm)
        sigma_x (float): Gaussian smearing in x
        sigma_y (float): Gaussian smearing in y
        device   (str): cpu or cuda
    """

    def __init__(self, 
                 csv_path: str,
                 xyz_min: Tuple,
                 xyz_max: Tuple,
                 sigma_x: float = 0.1,  # mm
                 sigma_y: float = 0.1,  # mm
                 device: str = "cuda" if torch.cuda.is_available() else 'cpu',
                ):
        self.device = device
        self.sigma_x = float(sigma_x)
        self.sigma_y = float(sigma_y)
        self.xyz_min = np.array(xyz_min, dtype=np.float32)
        self.xyz_max = np.array(xyz_max, dtype=np.float32)
        self.true_hits = self._load_hits_from_csv(csv_path).to(device)  # (muons,6,3)
        self.B = self.true_hits.shape[0]

        # independent Gaussian uncertainties
        self.sigma_vec = torch.tensor(
            [self.sigma_x, self.sigma_y] * 6,
            device=self.device,
            dtype=torch.float32,
        ) # (12,)

        self._poca: Optional[np.ndarray] = None
        self._sigma_poca: Optional[np.ndarray] = None
        self._theta: Optional[np.ndarray] = None
        self._sigma_theta: Optional[np.ndarray] = None
        self._S: Optional[np.ndarray] = None 
        self._N: Optional[np.ndarray] = None
        self._S_sigma: Optional[np.ndarray] = None


    def _load_hits_from_csv(self, path: str) -> Tensor:
        """
        Loads hit xyz from csv file. 
        Structure in csv: X0, Y0, Z0 ... X5, Y5, Z5, for 3 hits above and 3 hits below the VOI

        Args:
            path (str):  path to csv file 
            
        Returns:
            hits (Tensor): hit tensor (muons,6,3)
        """
        data = np.genfromtxt(path, delimiter=",", names=True)
        B = len(data)
        if B == 0:
            return torch.zeros((0, 6, 3), dtype=torch.float32)

        hits = np.zeros((B, 6, 3), dtype=np.float32)
        for i in range(6):
            hits[:, i, 0] = data[f"X{i}"]
            hits[:, i, 1] = data[f"Y{i}"]
            hits[:, i, 2] = data[f"Z{i}"]

        return torch.tensor(hits, dtype=torch.float32)

    def _smear_hits(self) -> Tensor:
        """
        Gaussian smearing in x,y; z unchanged.
        
        Returns:
            hits (Tensor): smeared hits (muons,6,3)
        """
        hits = self.true_hits.clone()
        B = hits.shape[0]
        device = hits.device

        noise_x = torch.randn(B, 6, device=device) * self.sigma_x
        noise_y = torch.randn(B, 6, device=device) * self.sigma_y

        hits[:, :, 0] += noise_x
        hits[:, :, 1] += noise_y
        return hits

    def _get_muon_tracks(self, hits: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Fits a linear trajectory to a group of hits.
        The fit is performed via an analytical likelihood-maximisation.
    
        Args:
            hits: (muons,hits,xyz) tensor of hit positions
    
        Returns:
            vec: (muons,xyz) fitted-vector directions
            start: (muons,xyz) initial point of fitted-vector
        """
        stars, angles = [], []
        for i in range(2):  # separate x and y
            mean_xz = torch.mean(hits[:, :, [i, 2]], dim=1)
            mean_xz_z = torch.mean(hits[:, :, [i, 2]] * hits[:, :, 2:3], dim=1)
            mean_x = mean_xz[:, :1]
            mean_z = mean_xz[:, 1:]
            mean_x_z = mean_xz_z[:, :1]
            mean_z2 = mean_xz_z[:, 1:]
    
            stars.append((mean_x - ((mean_z * mean_x_z) / mean_z2)) /
                         (1 - (mean_z.square() / mean_z2)))
            angles.append((mean_x_z - (stars[-1] * mean_z)) / mean_z2)
    
        xy_star = torch.cat(stars, dim=-1)
        angle = torch.cat(angles, dim=-1)
    
        def _calc_xyz(z: Tensor) -> Tensor:
            return torch.cat([xy_star + (angle * z), z], dim=-1)
    
        start = _calc_xyz(hits[:, 0, 2:3])  # only z from hits used
        end = _calc_xyz(hits[:, 2, 2:3])
        vec = end - start
    
        return vec, start

    def _compute_poca(self, hits: Tensor) -> Tensor:
        """
        Args:
            hits: (muons, 6, 3)  smeared hits
        Returns:
          out: (muons, 4) = [poca_x, poca_y, poca_z, theta]
        """
        
        hits_in = hits[:, 0:3, :]
        hits_out = hits[:, 3:6, :]
    
        v_in, p_in = self._get_muon_tracks(hits_in)
        v_out, p_out = self._get_muon_tracks(hits_out)
    
        # normalize directions
        v_in = v_in / (v_in.norm(dim=1, keepdim=True) + 1e-9)
        v_out = v_out / (v_out.norm(dim=1, keepdim=True) + 1e-9)
    
        # scattering angle
        dots = (v_in * v_out).sum(dim=1).clamp(-1.0, 1.0)
        theta = torch.acos(dots)  # (N_mu,)
    
        # POCA between two lines
        n = torch.cross(v_in, v_out, dim=1)           # (N_mu,3)
        
        L = torch.stack([v_in, -v_out, n], dim=2)     # (N_mu,3,3)
        rhs = (p_out - p_in).unsqueeze(-1)            # (N_mu,3,1)
    
        coefs = torch.linalg.solve(L, rhs).squeeze(-1)  # (N_mu,3)
        t1 = coefs[:, 0]
        t2 = coefs[:, 1]
    
        P = p_in + t1.unsqueeze(-1) * v_in
        Q = p_out + t2.unsqueeze(-1) * v_out
        poca = 0.5 * (P + Q)  # (N_mu,3)
    
        return torch.cat([poca, theta.unsqueeze(-1)], dim=1)  # (N_mu,4)


    def _compute_with_uncertainties(self):
        """
        Returns:
            poca:        (muons,3)
            sigma_poca:  (muons,3)
            theta:       (muons,)
            sigma_theta: (muons,)
        """
        device = self.device
        B = self.B
    
        smeared_hits = self._smear_hits()  # (B,6,3)
    
        # Flatten xy to (B,12); these are the differentiable inputs
        x_flat = smeared_hits[:, :, :2].reshape(B, 12).detach().to(device)
        x_flat.requires_grad_(True)
    
        # Fixed z for each muon hit (no uncertainty)
        z_all = self.true_hits[:, :, 2].to(device)  # (B,6)
    
        # Single-muon function: R^12 x R^6 -> R^4
        def f_single(x_row: Tensor, z_row: Tensor) -> Tensor:
            # x_row: (12,) = (x0,y0,...,x5,y5)
            # z_row: (6,)
            hits_xy = x_row.view(6, 2)       # (6,2)
            z = z_row.unsqueeze(-1)          # (6,1)
            hits = torch.cat([hits_xy, z], dim=1)  # (6,3)
            return self._compute_poca(hits.unsqueeze(0))[0]  # (4,)
    
        # Vectorized over muons for central values
        f_batched = vmap(f_single, in_dims=(0, 0))    # (B,12),(B,6)->(B,4)
        y0 = f_batched(x_flat, z_all)                 # (B,4)
    
        poca = y0[:, :3]   # (B,3)
        theta = y0[:, 3]   # (B,)
    
        # Jacobian J = dy/dx via jacrev
        jacobian_fn = jacrev(f_single, argnums=0)  # derivative wrt x_row
        # jac: (B,4,12)
        jac = vmap(jacobian_fn, in_dims=(0, 0))(x_flat, z_all)

        sigma2 = self.sigma_vec ** 2  # (12,)
    
        # Var[y_k] = sum_j( J[k,j]**2 * sigma[j]**2 )
        var_y = (jac ** 2) * sigma2.view(1, 1, 12)
        var_y = var_y.sum(dim=2)  # (B,4)
    
        sigma_poca = torch.sqrt(torch.clamp(var_y[:, :3], min=0.0))  # (B,3)
        sigma_theta = torch.sqrt(torch.clamp(var_y[:, 3], min=0.0))  # (B,)
    
        return poca, sigma_poca, theta, sigma_theta
        

    def _poca_to_voxel(self, poca_phys: np.ndarray, D: int)-> np.ndarray:
        """
        Computes PoCA voxel coordinates
        Args:
            poca_phys (np.ndarray): coordinates of PoCA in physical units
            D (int): Number of voxels in each dimension.

        Returns:
            (np.ndarray): PoCA coordinates in voxel units
        """
        scale = (D - 1) / (self.xyz_max - self.xyz_min)
        return (poca_phys - self.xyz_min[None, :]) * scale[None, :]

    def _compute_numpy(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes PoCA spatial and scattering information and returns them as numpy arrays.
        """
        poca, sigma_poca, theta, sigma_theta = self._compute_with_uncertainties()
        return (
            poca.detach().cpu().numpy(),
            sigma_poca.detach().cpu().numpy(),
            theta.detach().cpu().numpy(),
            sigma_theta.detach().cpu().numpy(),
        )

    def _deposit_gaussian(self, grid: np.ndarray, centers_vox: np.ndarray, weights: np.ndarray, sigma_xyz_phys: np.ndarray, D)-> None:
        """
        Deposit anisotropic Gaussians on a voxel grid.

        Args:
            grid (np.ndarray): voxelized grid of a channel (D, D, D)
            centers_vox (np.ndarray): voxel coordinates (muons,3)
            weights (np.ndarray): quantites weighting the kernel (muons,)
            sigma_xyz_phys (np.ndarray): PoCA uncertainties in physical units (muons,3):  
        """

        voxel_size_phys = (self.xyz_max - self.xyz_min) / (D - 1)
        min_sigma_phys = 0.5 * voxel_size_phys
        max_sigma_phys = 3 * voxel_size_phys

        scale = (D - 1) / (self.xyz_max - self.xyz_min)

        N = centers_vox.shape[0]

        for i in range(N):
            cx, cy, cz = centers_vox[i]

            s_phys = sigma_xyz_phys[i]
            s_phys = np.clip(s_phys, min_sigma_phys, max_sigma_phys)
            s_vox  = s_phys * scale

            sx, sy, sz = s_vox.astype(np.float32)

            rx = int(np.ceil(3 * sx))
            ry = int(np.ceil(3 * sy))
            rz = int(np.ceil(3 * sz))

            if rx == ry == rz == 0:
                ix = int(round(cx))
                iy = int(round(cy))
                iz = int(round(cz))
                if 0 <= ix < D and 0 <= iy < D and 0 <= iz < D:
                    grid[ix, iy, iz] += weights[i]
                continue

            gx, gy, gz = np.meshgrid(
                np.arange(-rx, rx+1),
                np.arange(-ry, ry+1),
                np.arange(-rz, rz+1),
                indexing="ij"
            )

            kern = np.exp(
                -(gx*gx/(2*sx*sx+1e-9)
                 + gy*gy/(2*sy*sy+1e-9)
                 + gz*gz/(2*sz*sz+1e-9))
            )

            ix = int(round(cx))
            iy = int(round(cy))
            iz = int(round(cz))

            x0, y0, z0 = ix - rx, iy - ry, iz - rz
            x1, y1, z1 = ix + rx + 1, iy + ry + 1, iz + rz + 1

            gx0 = max(0, -x0); gy0 = max(0, -y0); gz0 = max(0, -z0)
            gx1 = kern.shape[0] - max(0, x1 - D)
            gy1 = kern.shape[1] - max(0, y1 - D)
            gz1 = kern.shape[2] - max(0, z1 - D)

            rx0 = max(0, x0); ry0 = max(0, y0); rz0 = max(0, z0)
            rx1 = min(D, x1); ry1 = min(D, y1); rz1 = min(D, z1)

            if rx0 < rx1 and ry0 < ry1 and rz0 < rz1:
                grid[rx0:rx1, ry0:ry1, rz0:rz1] += \
                    weights[i] * kern[gx0:gx1, gy0:gy1, gz0:gz1]

      
    def voxelize(
        self,
        poca: np.ndarray,
        theta: np.ndarray,
        sigma_poca: np.ndarray,
        sigma_theta: np.ndarray,
        D: int,
    )-> (np.ndarray, np.ndarray, np.ndarray):
        """
        Returns voxelized PoCA input channels for the nn model.
        """

        if poca.shape[0] == 0:
            Z = np.zeros((D, D, D), np.float32)
            return Z.copy(), Z.copy(), Z.copy()

        valid = np.isfinite(sigma_poca).all(axis=1) & np.isfinite(sigma_theta)
        poca = poca[valid]
        theta = theta[valid]
        sigma_poca = sigma_poca[valid]
        sigma_theta = sigma_theta[valid]

        if poca.shape[0] == 0:
            Z = np.zeros((D, D, D), np.float32)
            return Z.copy(), Z.copy(), Z.copy()

        poca_vox = self._poca_to_voxel(poca, D)  #(muons, 3)

        inside = np.all(
            (poca >= self.xyz_min[None, :]) &
            (poca <= self.xyz_max[None, :]),
            axis=1
        )

        poca_vox = poca_vox[inside]
        theta = theta[inside]
        sigma_poca = sigma_poca[inside]
        sigma_theta = sigma_theta[inside]

        S       = np.zeros((D, D, D), np.float32)
        N       = np.zeros_like(S)
        S_sigma = np.zeros_like(S)

        if poca_vox.shape[0] > 0:
            self._deposit_gaussian(S,       poca_vox, theta**2,   sigma_poca, D)
            self._deposit_gaussian(N,       poca_vox, np.ones_like(theta), sigma_poca, D)
            self._deposit_gaussian(S_sigma, poca_vox, sigma_theta, sigma_poca, D)

            N = np.rint(N).astype(np.int32)

        return S, N, S_sigma

    def plot_poca_projections(self, poca: np.ndarray, bins=50):
        poca_np = poca
        x, y, z = poca_np[:, 0], poca_np[:, 1], poca_np[:, 2]
    
        # user-defined ranges
        x_min, x_max = self.xyz_min[0],  self.xyz_max[0]
        y_min, y_max = self.xyz_min[1],  self.xyz_max[1]
        z_min, z_max = self.xyz_min[2],  self.xyz_max[2]
    
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
        # XZ
        axs[0].hist2d(
            x, z, bins=bins,
            range=[[x_min, x_max], [z_min, z_max]],
            cmap="viridis"
        )
        axs[0].set_xlabel("X [mm]")
        axs[0].set_ylabel("Z [mm]")
        axs[0].set_title("XZ Projection")
    
        # YZ
        axs[1].hist2d(
            y, z, bins=bins,
            range=[[y_min, y_max], [z_min, z_max]],
            cmap="viridis"
        )
        axs[1].set_xlabel("Y [mm]")
        axs[1].set_ylabel("Z [mm]")
        axs[1].set_title("YZ Projection")
    
        # XY
        axs[2].hist2d(
            x, y, bins=bins,
            range=[[x_min, x_max], [y_min, y_max]],
            cmap="viridis"
        )
        axs[2].set_xlabel("X [mm]")
        axs[2].set_ylabel("Y [mm]")
        axs[2].set_title("XY Projection")
    
        plt.tight_layout()
        plt.show()

    
    def precompute_poca_voxels(
        self,
        target_path: str,
        out_path: str,
    ):
        """
        Precomputes PoCA voxelized maps and stores in an output directory to be used for training.
        """
        target = np.load(target_path)
        D = target.shape[0]

        poca, sigma_poca, theta, sigma_theta = self._compute_numpy()

        S, N, S_sigma = self.voxelize(
            poca, theta, sigma_poca, sigma_theta,
            D=D
        )

        np.savez_compressed(
            out_path,
            S=S.astype(np.float32),
            N=N.astype(np.int16),
            S_sigma=S_sigma.astype(np.float32),
            voi_min=self.xyz_min,
            voi_max=self.xyz_max,
            D=np.array(D, dtype=np.int32)
        )
        
    @property
    def poca(self)-> Optional[np.ndarray]:
        """
        Returns:
            (muons, 3) array of poca smeared positions 
        """
        if self._poca is None:
            self._poca,_ , _, _ = self._compute_numpy()
            return self._poca
        return self._poca

    @property
    def sigma_poca(self)-> Optional[np.ndarray]:
        """
        Returns:
            (muons, 3) array of uncertainties on PoCA smeared positions 
        """
        if self._sigma_poca is None:
            _, self._sigma_poca, _, _ = self._compute_numpy()
            return self._sigma_poca
        return self._sigma_poca

    @property
    def theta(self)-> Optional[np.ndarray]:
        """
        Returns:
            (muons, ) array of PocA scattering angles 
        """
        if self._theta is None:
            _, _, self._theta, _ = self._compute_numpy()
            return self._theta
        return self._theta

    @property
    def theta(self)-> Optional[np.ndarray]:
        """
        Returns:
            (muons, ) array of uncertainties on PocA scattering angles 
        """
        if self._sigma_theta is None:
            _, _, _, self._sigma_theta = self._compute_numpy()
            return self._sigma_theta
        return self._sigma_theta
        
        
        
    
    




