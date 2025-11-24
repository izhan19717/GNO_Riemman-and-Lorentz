from abc import ABC, abstractmethod
import torch

class Manifold(ABC):
    """Abstract base class for Riemannian and Lorentzian manifolds."""
    
    def __init__(self, dim: int):
        self.dim = dim

    @abstractmethod
    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute distance between points x and y."""
        pass

    @abstractmethod
    def exp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at x in direction v."""
        pass

    @abstractmethod
    def log(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from x to y."""
        pass
