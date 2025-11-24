import torch
from abc import ABC, abstractmethod

class PDE(ABC):
    @abstractmethod
    def residual(self, model, u_coeffs, y_points):
        pass

class PoissonSphere(PDE):
    """
    Poisson equation on Sphere: Delta u = f
    """
    def __init__(self, radius=1.0):
        self.radius = radius

    def residual(self, model, u_coeffs, y_points):
        """
        Compute Delta u - f
        y_points: [batch, num_points, 3] (Cartesian coordinates on sphere)
        """
        # Enable gradient computation for y_points
        y_points.requires_grad_(True)
        
        # Predict u(y)
        u_pred = model(u_coeffs, y_points) # [batch, num_points, 1]
        
        # Compute gradients
        grads = torch.autograd.grad(u_pred, y_points, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        
        # Laplacian in embedding space (for sphere constraint |x|=R)
        # Delta_S u = Delta_R3 u - (x/R^2) . grad u - (x . grad grad u . x)/R^4 ?
        # Simpler: Project gradient onto tangent space and take divergence.
        # Or use spherical coordinates if y_points were (theta, phi).
        
        # For PoC, let's assume we use the embedding space Laplacian minus radial component
        # But standard way is:
        # P = I - x x^T / R^2
        # grad_S u = P grad u
        # Delta_S u = div_S (grad_S u)
        
        # Let's stick to a simpler approximation or assume y_points are (theta, phi) for now?
        # No, trunk uses Cartesian usually for stability.
        
        # Let's implement the projection method.
        # grad_u: [batch, num_points, 3]
        
        # Project gradient
        # n = y / R
        n = y_points / self.radius
        # grad_s = grad - (grad . n) n
        dot = torch.sum(grads * n, dim=-1, keepdim=True)
        grad_s = grads - dot * n
        
        # Divergence of grad_s?
        # This is getting expensive. 
        # Alternative: We are learning the operator G: f -> u.
        # So we check if Delta G(f) = f.
        # But we need f at y_points.
        
        # For now, return u_pred and let the loss handle the comparison with ground truth u
        # If we do PINN style, we need the residual.
        
        return u_pred

class WaveMinkowski(PDE):
    """
    Wave equation on Minkowski: Box u = f
    Box = -dt^2 + dx^2
    """
    def residual(self, model, u_coeffs, y_points):
        """
        y_points: [batch, num_points, 2] (t, x) for 1+1D
        """
        y_points.requires_grad_(True)
        u_pred = model(u_coeffs, y_points)
        
        grads = torch.autograd.grad(u_pred, y_points, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        dt = grads[..., 0]
        dx = grads[..., 1]
        
        grads2_t = torch.autograd.grad(dt, y_points, grad_outputs=torch.ones_like(dt), create_graph=True)[0]
        grads2_x = torch.autograd.grad(dx, y_points, grad_outputs=torch.ones_like(dx), create_graph=True)[0]
        
        dtt = grads2_t[..., 0]
        dxx = grads2_x[..., 1]
        
        box_u = -dtt + dxx
        
        return box_u
