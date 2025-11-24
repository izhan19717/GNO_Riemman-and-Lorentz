import torch
from .manifold import Manifold

class Sphere(Manifold):
    """Sphere manifold S^d embedded in R^{d+1}."""
    
    def __init__(self, dim: int = 2, radius: float = 1.0):
        super().__init__(dim)
        self.radius = radius

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Geodesic distance on the sphere.
        d(x, y) = R * arccos(<x, y> / R^2)
        """
        # Normalize to ensure numerical stability
        x_norm = x / torch.norm(x, dim=-1, keepdim=True) * self.radius
        y_norm = y / torch.norm(y, dim=-1, keepdim=True) * self.radius
        
        dot_prod = torch.sum(x_norm * y_norm, dim=-1)
        # Clamp for numerical stability
        # Gradient of acos is inf at +/- 1, so we clamp slightly inside
        cos_theta = dot_prod / self.radius**2
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
        
        return self.radius * torch.acos(cos_theta)

    def proj_tan(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project vector v onto tangent space at x.
        T_x S^d = {v in R^{d+1} : <x, v> = 0}
        """
        dot = torch.sum(x * v, dim=-1, keepdim=True)
        return v - dot * x / self.radius**2

    def exp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map on the sphere.
        x: point on sphere
        v: tangent vector at x
        """
        # Ensure v is tangent
        v = self.proj_tan(x, v)
        
        v_norm = torch.norm(v, dim=-1)
        # Avoid division by zero
        mask = v_norm > 1e-8
        
        res = x.clone()
        
        # For non-zero vectors
        if mask.any():
            x_masked = x[mask]
            v_masked = v[mask]
            vn_masked = v_norm[mask]
            
            res[mask] = torch.cos(vn_masked / self.radius).unsqueeze(-1) * x_masked + \
                        self.radius * torch.sin(vn_masked / self.radius).unsqueeze(-1) * (v_masked / vn_masked.unsqueeze(-1))
            
        return res

    def log(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map on the sphere.
        """
        d = self.dist(x, y)
        
        # Projection of y onto tangent space at x
        # P_x(y) = y - <x,y>x/|x|^2
        dot = torch.sum(x * y, dim=-1, keepdim=True)
        proj = y - dot * x / self.radius**2
        
        proj_norm = torch.norm(proj, dim=-1, keepdim=True)
        
        # Avoid division by zero
        mask = proj_norm.squeeze() > 1e-8
        
        res = torch.zeros_like(x)
        if mask.any():
            res[mask] = (d[mask].unsqueeze(-1) / proj_norm[mask]) * proj[mask]
            
        return res
