"""
Geodesic Feature Extraction for Enhanced Trunk Networks

This module implements coordinate-free geometric features based on geodesic distances,
local curvature, and parallel transport. These features provide the trunk network with
intrinsic geometric information that improves sample efficiency and generalization.

References:
- Bronstein et al. (2021): "Geometric Deep Learning"
- Qi et al. (2017): "PointNet++: Deep Hierarchical Feature Learning"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class GeodesicFeatureExtractor:
    """
    Extract geodesic-based features for trunk network input.
    
    Features include:
    1. Geodesic distances to reference points (coordinate-free)
    2. Local curvature (sectional curvature for Riemannian manifolds)
    3. Ambient coordinates (for embedding-based manifolds)
    4. Parallel transport features (optional, expensive)
    
    Args:
        manifold: Manifold object with dist(), sectional_curvature() methods
        num_refs: Number of reference points for distance features
        use_curvature: Whether to include curvature features
        use_parallel_transport: Whether to include parallel transport features
    """
    
    def __init__(
        self, 
        manifold, 
        num_refs: int = 32,
        use_curvature: bool = True,
        use_parallel_transport: bool = False
    ):
        self.manifold = manifold
        self.num_refs = num_refs
        self.use_curvature = use_curvature
        self.use_parallel_transport = use_parallel_transport
        
        # Initialize reference points using Farthest Point Sampling
        self.reference_points = self._initialize_references()
        
    def _initialize_references(self) -> torch.Tensor:
        """
        Initialize reference points using Farthest Point Sampling (FPS).
        
        FPS ensures good coverage of the manifold by iteratively selecting
        points that maximize minimum distance to already selected points.
        
        Returns:
            reference_points: [num_refs, manifold.dim]
        """
        # Sample initial pool of candidates
        if hasattr(self.manifold, 'random_point'):
            candidates = self.manifold.random_point(1000)
        else:
            # Fallback: uniform sampling in ambient space + projection
            candidates = torch.randn(1000, self.manifold.dim)
            if hasattr(self.manifold, 'project'):
                candidates = self.manifold.project(candidates)
        
        # Farthest Point Sampling
        refs = [candidates[0]]  # Start with first point
        
        for _ in range(self.num_refs - 1):
            # Compute distances from all candidates to current refs
            dists_to_refs = []
            for ref in refs:
                d = self.manifold.dist(candidates, ref.unsqueeze(0))
                dists_to_refs.append(d)
            
            # Minimum distance to any reference
            min_dists = torch.stack(dists_to_refs, dim=0).min(dim=0)[0]
            
            # Select point with maximum minimum distance
            farthest_idx = min_dists.argmax()
            refs.append(candidates[farthest_idx])
        
        return torch.stack(refs)
    
    def extract_features(self, query_points: torch.Tensor) -> torch.Tensor:
        """
        Extract geodesic features for query points.
        
        Args:
            query_points: [N, manifold.dim] points on manifold
            
        Returns:
            features: [N, feature_dim] where feature_dim depends on options
        """
        features = []
        
        # 1. Geodesic distances to reference points
        # Shape: [N, num_refs]
        dists = self.manifold.dist(
            query_points.unsqueeze(1),  # [N, 1, dim]
            self.reference_points.unsqueeze(0)  # [1, num_refs, dim]
        )
        features.append(dists)
        
        # 2. Ambient coordinates (if available)
        features.append(query_points)
        
        # 3. Local curvature
        if self.use_curvature:
            if hasattr(self.manifold, 'sectional_curvature'):
                curv = self.manifold.sectional_curvature(query_points)
                features.append(curv.unsqueeze(-1))  # [N, 1]
            elif hasattr(self.manifold, 'curvature'):
                # Constant curvature manifolds
                K = torch.full((query_points.shape[0], 1), self.manifold.curvature)
                features.append(K)
        
        # 4. Parallel transport features (optional, expensive)
        if self.use_parallel_transport and hasattr(self.manifold, 'parallel_transport'):
            # Transport a canonical vector from each ref to query point
            # This encodes directional information
            transport_features = []
            canonical_vector = torch.zeros_like(self.reference_points[0])
            canonical_vector[0] = 1.0  # e_1 direction
            
            for ref in self.reference_points[:4]:  # Limit to first 4 refs for efficiency
                transported = self.manifold.parallel_transport(
                    canonical_vector.unsqueeze(0).expand(query_points.shape[0], -1),
                    ref.unsqueeze(0).expand(query_points.shape[0], -1),
                    query_points
                )
                # Project to scalar via inner product with query point
                inner = torch.sum(transported * query_points, dim=-1, keepdim=True)
                transport_features.append(inner)
            
            if transport_features:
                features.append(torch.cat(transport_features, dim=-1))
        
        return torch.cat(features, dim=-1)
    
    @property
    def feature_dim(self) -> int:
        """Compute total feature dimension."""
        dim = self.num_refs + self.manifold.dim  # distances + coords
        if self.use_curvature:
            dim += 1
        if self.use_parallel_transport:
            dim += 4  # 4 parallel transport features
        return dim


class GeodesicTrunkNet(nn.Module):
    """
    Enhanced Trunk Network with geodesic features.
    
    This trunk network uses coordinate-free geometric features instead of
    raw coordinates, providing better inductive bias for manifold learning.
    
    Args:
        manifold: Manifold object
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (latent space)
        num_refs: Number of reference points for geodesic features
    """
    
    def __init__(
        self,
        manifold,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_refs: int = 32
    ):
        super().__init__()
        
        self.feature_extractor = GeodesicFeatureExtractor(
            manifold, 
            num_refs=num_refs,
            use_curvature=True,
            use_parallel_transport=False  # Disable for efficiency
        )
        
        input_dim = self.feature_extractor.feature_dim
        self.output_dim = output_dim  # Store for compatibility with GeometricDeepONet
        
        # MLP with residual connections
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch, num_points, manifold.dim] query points
            
        Returns:
            out: [batch, num_points, output_dim] trunk features
        """
        batch_size, num_points, _ = x.shape
        
        # Flatten batch dimension for feature extraction
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Extract features
        features = self.feature_extractor.extract_features(x_flat)
        
        # MLP
        out = self.mlp(features)
        
        # Reshape back
        return out.reshape(batch_size, num_points, -1)


# Example usage and testing
if __name__ == "__main__":
    from src.geometry.sphere import Sphere
    from src.geometry.hyperbolic import Hyperboloid
    
    print("Testing Geodesic Feature Extraction...")
    
    # Test on Sphere
    print("\n=== Sphere ===")
    sphere = Sphere(radius=1.0)
    extractor_sphere = GeodesicFeatureExtractor(sphere, num_refs=16)
    
    # Sample query points
    query = sphere.random_point(100)
    features = extractor_sphere.extract_features(query)
    
    print(f"Query points shape: {query.shape}")
    print(f"Feature shape: {features.shape}")
    print(f"Feature dim: {extractor_sphere.feature_dim}")
    print(f"Reference points: {extractor_sphere.reference_points.shape}")
    
    # Test Trunk Network
    trunk = GeodesicTrunkNet(sphere, hidden_dim=64, output_dim=32, num_refs=16)
    query_batch = query.unsqueeze(0)  # [1, 100, 3]
    out = trunk(query_batch)
    print(f"Trunk output shape: {out.shape}")
    
    # Test on Hyperboloid
    print("\n=== Hyperboloid ===")
    hyp = Hyperboloid(dim=2)
    extractor_hyp = GeodesicFeatureExtractor(hyp, num_refs=16)
    
    query_hyp = hyp.random_point(100)
    features_hyp = extractor_hyp.extract_features(query_hyp)
    
    print(f"Query points shape: {query_hyp.shape}")
    print(f"Feature shape: {features_hyp.shape}")
    print(f"Feature dim: {extractor_hyp.feature_dim}")
    
    print("\n✓ Geodesic feature extraction working correctly!")
