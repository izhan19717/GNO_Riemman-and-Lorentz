import torch
from .manifold import Manifold

class Hyperboloid(Manifold):
    """
    Hyperboloid model of Hyperbolic space H^n.
    Embedded in R^{n+1} with Minkowski metric.
    H^n = {x in R^{n+1} : <x, x>_L = -1, x[0] > 0}
    Metric signature: (-1, 1, 1, ...)
    """
    def __init__(self, dim: int = 2):
        super().__init__(dim)
        self.embedding_dim = dim + 1

    def minkowski_dot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute <x, y>_L = -x0*y0 + x1*y1 + ...
        x, y: [..., dim+1]
        """
        res = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
        return res

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Geodesic distance on H^n.
        d(x, y) = acosh(-<x, y>_L)
        """
        prod = self.minkowski_dot(x, y)
        prod = torch.clamp(prod, max=-1.0 - 1e-7)
        return torch.acosh(-prod)

    def proj_tan(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project vector v onto tangent space at x.
        T_x H^n = {v in R^{n+1} : <x, v>_L = 0}
        """
        prod = self.minkowski_dot(x, v)
        # x is unit timelike vector (<x,x> = -1)
        # Projection: v - <x, v>_L * x / <x, x>_L = v + <x, v>_L * x
        return v + prod.unsqueeze(-1) * x

    def exp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at x.
        y = cosh(|v|_L) x + sinh(|v|_L) v / |v|_L
        where |v|_L = sqrt(<v, v>_L)
        """
        # v must be in tangent space
        v = self.proj_tan(x, v)
        
        # Norm of v in Minkowski metric
        # Since v is spacelike (tangent to H^n), <v, v>_L >= 0
        sq_norm = self.minkowski_dot(v, v)
        sq_norm = torch.clamp(sq_norm, min=0.0)
        norm = torch.sqrt(sq_norm)
        
        # Avoid division by zero
        mask = norm > 1e-7
        
        res = x.clone()
        
        # Case norm > 0
        if mask.any():
            n_v = norm[mask].unsqueeze(-1)
            x_v = x[mask]
            v_v = v[mask]
            
            res[mask] = torch.cosh(n_v) * x_v + torch.sinh(n_v) * (v_v / n_v)
            
        # Case norm ~ 0: res = x (already set)
        
        return res

    def log(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at x.
        v = d(x, y) * (y + <x, y>_L x) / |y + <x, y>_L x|_L
        """
        prod = self.minkowski_dot(x, y)
        dist = self.dist(x, y)
        
        # u = y + <x, y>_L x
        # This vector is tangent to x
        u = y + prod.unsqueeze(-1) * x
        
        # Norm of u
        sq_norm_u = self.minkowski_dot(u, u)
        sq_norm_u = torch.clamp(sq_norm_u, min=0.0)
        norm_u = torch.sqrt(sq_norm_u)
        
        mask = dist > 1e-7
        res = torch.zeros_like(x)
        
        if mask.any():
            d_v = dist[mask].unsqueeze(-1)
            n_u_v = norm_u[mask].unsqueeze(-1)
            u_v = u[mask]
            
            res[mask] = d_v * u_v / n_u_v
            
        return res

    def random_point(self, num_samples=1):
        """
        Sample random points on H^n via projection from light cone or normal distribution.
        Method: Sample tangent vector at North Pole (1, 0, ...) and Exp map.
        """
        # North pole
        mu = torch.zeros(num_samples, self.embedding_dim)
        mu[:, 0] = 1.0
        
        # Random tangent vector
        # v = (0, v1, ..., vn)
        v = torch.randn(num_samples, self.embedding_dim)
        v[:, 0] = 0
        
        # Scale v to cover some area
        v = v * 1.5 
        
        return self.exp(mu, v)
    
    def parallel_transport(self, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parallel transport vector v from point x to point y along geodesic.
        
        For hyperboloid model, we use the formula:
        v_transported = v - <v, y>_L / (1 + <x, y>_L) * (x + y)
        
        Reference: Ungar (2001), "Hyperbolic Geometry"
        """
        v_dot_y = self.minkowski_dot(v, y)
        x_dot_y = self.minkowski_dot(x, y)
        factor = v_dot_y / (1.0 + x_dot_y)
        v_transported = v - factor.unsqueeze(-1) * (x + y)
        return v_transported
    
    def sectional_curvature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sectional curvature at point x.
        For hyperboloid model with constant negative curvature: K = -1 / R^2
        """
        batch_size = x.shape[0] if x.dim() > 1 else 1
        return torch.full((batch_size,), -1.0, device=x.device)
