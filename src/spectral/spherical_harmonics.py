import torch
import numpy as np

try:
    import torch_harmonics as th
    HAS_TORCH_HARMONICS = True
except ImportError:
    HAS_TORCH_HARMONICS = False
    from scipy.special import sph_harm

class SphericalHarmonics:
    """
    Wrapper for Spherical Harmonics transform.
    Uses torch-harmonics if available, otherwise falls back to scipy (slow, CPU only).
    """
    def __init__(self, lmax: int, nlat: int, nlon: int):
        self.lmax = lmax
        self.nlat = nlat
        self.nlon = nlon
        
        if HAS_TORCH_HARMONICS:
            self.sht = th.RealSHT(nlat, nlon, lmax=lmax, grid="legendre-gauss")
            self.isht = th.InverseRealSHT(nlat, nlon, lmax=lmax, grid="legendre-gauss")
        else:
            print("Warning: torch-harmonics not found. Using scipy fallback (slow).")
            
        # Always compute grid to ensure attributes are available
        self._compute_grid()

    def _compute_grid(self):
        # Gauss-Legendre grid
        theta, weights = np.polynomial.legendre.leggauss(self.nlat)
        # Shift theta from [-1, 1] to [0, pi]
        # leggauss returns nodes in [-1, 1] (cos(theta))
        # theta = arccos(x)
        self.cos_theta = torch.tensor(theta, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.theta = torch.acos(self.cos_theta)
        
        # Longitude [0, 2pi)
        self.phi = torch.linspace(0, 2*np.pi, self.nlon+1)[:-1]
        
        # Meshgrid
        self.Theta, self.Phi = torch.meshgrid(self.theta, self.phi, indexing='ij')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward SHT: Spatial -> Spectral
        x: [batch, nlat, nlon]
        Returns: coeffs [batch, lmax, lmax] (complex or real representation)
        """
        if HAS_TORCH_HARMONICS:
            return self.sht(x)
        else:
            # Naive integration using scipy sph_harm
            # This is extremely slow and just a placeholder for PoC if library fails
            raise NotImplementedError("Scipy fallback for full SHT not fully implemented for batch processing efficiently. Please install torch-harmonics.")

    def inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Inverse SHT: Spectral -> Spatial
        coeffs: [batch, lmax, lmax]
        Returns: x [batch, nlat, nlon]
        """
        if HAS_TORCH_HARMONICS:
            return self.isht(coeffs)
        else:
            raise NotImplementedError("Scipy fallback not implemented.")
