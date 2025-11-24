import torch
from .manifold import Manifold

class Minkowski(Manifold):
    """Minkowski spacetime R^{d,1}."""
    
    def __init__(self, spatial_dim: int = 1):
        # Total dim = spatial_dim + 1 (time)
        super().__init__(spatial_dim + 1)
        self.spatial_dim = spatial_dim

    def metric(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Minkowski metric signature (-, +, +, ...).
        u, v: [batch, dim] where dim 0 is time.
        """
        # Assuming last dim is coordinates (t, x1, x2, ...)
        dt = u[..., 0] * v[..., 0]
        dx = torch.sum(u[..., 1:] * v[..., 1:], dim=-1)
        return -dt + dx

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Squared Lorentzian distance (Interval).
        s^2 = -dt^2 + dx^2
        Returns signed squared distance to distinguish timelike/spacelike.
        """
        if x.shape != y.shape:
            # Try broadcasting check
            try:
                torch.broadcast_shapes(x.shape, y.shape)
            except RuntimeError:
                raise RuntimeError(f"Minkowski.dist shape mismatch: x {x.shape}, y {y.shape}")
        diff = y - x
        return self.metric(diff, diff)

    def proper_time(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Proper time for timelike separation.
        tau = sqrt(-s^2) if s^2 < 0 else 0
        """
        s2 = self.dist(x, y)
        mask = s2 < 0
        res = torch.zeros_like(s2)
        res[mask] = torch.sqrt(-s2[mask])
        return res

    def proper_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Proper distance for spacelike separation.
        sigma = sqrt(s^2) if s^2 > 0 else 0
        """
        s2 = self.dist(x, y)
        mask = s2 > 0
        res = torch.zeros_like(s2)
        res[mask] = torch.sqrt(s2[mask])
        return res

    def exp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map in flat space is just addition."""
        return x + v

    def log(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map in flat space is just subtraction."""
        return y - x

    def is_timelike(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.dist(x, y) < 0

    def is_spacelike(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.dist(x, y) > 0

    def is_causal(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Check if y is in the future causal cone of x.
        (y_t > x_t) AND (s^2 <= 0)
        """
        if x.shape != y.shape:
             try:
                torch.broadcast_shapes(x.shape, y.shape)
             except RuntimeError:
                print(f"CRITICAL: Shape mismatch in is_causal: x {x.shape}, y {y.shape}")
                raise RuntimeError(f"Minkowski.is_causal shape mismatch: x {x.shape}, y {y.shape}")
        
        # print(f"DEBUG: is_causal x {x.shape}, y {y.shape}")
        # import sys
        # print(f"DEBUG: is_causal x {x.shape}, y {y.shape}")
        # sys.stdout.flush()
        
        with open("debug_log.txt", "a") as f:
            f.write(f"DEBUG: is_causal x {x.shape}, y {y.shape}\n")
        
        dt = y[..., 0] - x[..., 0]
        s2 = self.dist(x, y)
        return (dt > 0) & (s2 <= 0)

    def light_cone_coords(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform (t, x) to light cone coordinates (u, v).
        u = t - x
        v = t + x
        Only valid for 1+1D.
        """
        if self.spatial_dim != 1:
            raise ValueError("Light cone coords only implemented for 1+1D")
            
        t = x[..., 0]
        s = x[..., 1] # Spatial coordinate
        
        u = t - s
        v = t + s
        
        return torch.stack([u, v], dim=-1)
