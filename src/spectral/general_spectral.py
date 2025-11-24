import torch
import numpy as np

class GeneralSpectralBasis:
    """
    Compute spectral basis (eigenfunctions of Laplace-Beltrami)
    using Diffusion Maps / Kernel method on arbitrary manifolds.
    
    Requires only a distance function or sampled points.
    """
    def __init__(self, num_eigenfunctions=64, epsilon=0.1):
        self.k = num_eigenfunctions
        self.epsilon = epsilon
        self.eigenvalues = None
        self.eigenvectors = None
        self.points = None

    def fit(self, points: torch.Tensor, manifold=None):
        """
        Compute basis functions from points.
        points: [N, dim]
        manifold: Object with .dist(x, y) method.
        """
        self.points = points
        N = points.shape[0]
        
        # Compute pairwise distance matrix
        # This can be expensive for large N. For PoC, N ~ 1000 is fine.
        # Use manifold.dist if available, else Euclidean (bad for curvature)
        
        if manifold is not None:
            # Batched distance computation?
            # dist_matrix[i, j] = dist(points[i], points[j])
            # Naive loop for safety, or broadcast if manifold supports it
            dists = torch.zeros(N, N)
            # TODO: Vectorize this. 
            # For Hyperboloid/Sphere, we can use dot product formulas vectorized.
            
            if hasattr(manifold, 'minkowski_dot') and hasattr(manifold, 'dist'):
                # Vectorized Hyperboloid
                # <x, y>_L
                # x: [N, 1, D], y: [1, N, D]
                x_exp = points.unsqueeze(1)
                y_exp = points.unsqueeze(0)
                # We need to call internal dot product logic manually or trust dist handles broadcasting
                # Let's try direct dist call if it supports broadcasting
                try:
                    dists = manifold.dist(x_exp, y_exp)
                except:
                    print("Manifold dist broadcasting failed, using loop (slow).")
                    for i in range(N):
                        dists[i] = manifold.dist(points[i:i+1], points)
            elif hasattr(manifold, 'dist'):
                 # Sphere or others
                 x_exp = points.unsqueeze(1)
                 y_exp = points.unsqueeze(0)
                 dists = manifold.dist(x_exp, y_exp)
        else:
            # Euclidean fallback
            dists = torch.cdist(points, points)
            
        # Gaussian Kernel
        # K_ij = exp(-d_ij^2 / epsilon)
        K = torch.exp(-dists**2 / self.epsilon)
        
        # Normalize Kernel (Graph Laplacian)
        # D_ii = sum_j K_ij
        D = K.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D))
        
        # Normalized Laplacian L = I - D^-1/2 K D^-1/2
        # We want eigenvectors of K_norm = D^-1/2 K D^-1/2
        K_norm = D_inv_sqrt @ K @ D_inv_sqrt
        
        # Eigendecomposition
        # torch.linalg.eigh for symmetric matrices
        eigvals, eigvecs = torch.linalg.eigh(K_norm)
        
        # Sort descending (largest eigenvalues of Kernel correspond to smallest eigenvalues of Laplacian)
        # indices = torch.argsort(eigvals, descending=True)
        # We want top k
        # eigvals are typically sorted ascending by eigh?
        # eigh returns ascending. So we take last k.
        
        self.eigenvalues = eigvals[-self.k:]
        self.eigenvectors = eigvecs[:, -self.k:] # [N, k]
        
        # Flip order to have dominant first
        self.eigenvalues = self.eigenvalues.flip(0)
        self.eigenvectors = self.eigenvectors.flip(1)
        
        return self

    def project(self, f_values: torch.Tensor) -> torch.Tensor:
        """
        Project function values f (defined on points) onto basis.
        f_values: [N, 1] or [N]
        Returns: coefficients [k]
        """
        # coeffs = <f, psi_i>
        # Since eigenvectors are orthonormal in discrete sense:
        # c = V^T f
        if f_values.dim() == 1:
            f_values = f_values.unsqueeze(1)
            
        coeffs = self.eigenvectors.T @ f_values
        return coeffs.squeeze()

    def reconstruct(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct function from coefficients.
        f = V c
        """
        if coeffs.dim() == 1:
            coeffs = coeffs.unsqueeze(1)
            
        res = self.eigenvectors @ coeffs
        return res.squeeze()
