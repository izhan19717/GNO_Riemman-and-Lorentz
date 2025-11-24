import torch
from .manifold import Manifold

class Torus(Manifold):
    """
    Flat Torus T^2 = S^1 x S^1.
    Represented as the unit square [0, 1]^2 with periodic boundary conditions.
    """
    def __init__(self):
        super().__init__(dim=2)
        
    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Geodesic distance on the Torus.
        d(x, y) = min(|x - y|, 1 - |x - y|) in each dimension.
        
        x, y: [..., 2] in [0, 1]^2
        """
        diff = torch.abs(x - y)
        # Distance in each dimension considering periodicity
        # d_i = min(diff_i, 1 - diff_i)
        d = torch.min(diff, 1 - diff)
        
        # Euclidean distance of the components
        return torch.norm(d, dim=-1)
        
    def exp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map: x + v modulo 1.
        """
        return (x + v) % 1.0
        
    def log(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Log map: vector from x to y.
        We need the shortest vector respecting periodicity.
        """
        diff = y - x
        # Adjust diff to be in [-0.5, 0.5]
        diff = diff - torch.round(diff)
        return diff
        
    def random_points(self, n: int) -> torch.Tensor:
        """
        Uniform random points on [0, 1]^2.
        """
        return torch.rand(n, 2)
