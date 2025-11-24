import torch

class FourierTransform:
    """
    Standard FFT wrapper for spatial slices in Minkowski space.
    """
    def __init__(self, n_points: int):
        self.n_points = n_points

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RFFT: Spatial -> Spectral
        x: [batch, n_points] (1D spatial)
        """
        return torch.fft.rfft(x, dim=-1)

    def inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        IRFFT: Spectral -> Spatial
        """
        return torch.fft.irfft(coeffs, n=self.n_points, dim=-1)
