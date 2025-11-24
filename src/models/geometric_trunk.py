import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from src.geometry.manifold import Manifold
from src.geometry.minkowski import Minkowski

class GeometricTrunk(nn.Module):
    """
    Geometric Trunk Network that explicitly computes features required by 
    Universal Approximation Theorems.
    
    Features:
    1. Geodesic distances to reference points {x_k}: d(x, x_k)
    2. (Lorentzian) Proper time/distance to reference events.
    3. (Lorentzian) Light cone coordinates.
    """
    def __init__(self, manifold: Manifold, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_references: int = 16):
        super().__init__()
        self.manifold = manifold
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_references = num_references
        
        # Learnable reference points
        # Initialize randomly in the domain
        # Use input_dim (embedding dimension)
        self.reference_points = nn.Parameter(torch.randn(num_references, input_dim))
        
        # Feature dimension depends on manifold type
        # Base features: input coords
        # Extra features: distances to refs
        self.feature_dim = input_dim + num_references
        
        if isinstance(manifold, Minkowski):
            # Add light cone coords (2) + causal indicators (num_refs)
            if manifold.spatial_dim == 1:
                self.feature_dim += 2 
            self.feature_dim += num_references # Causal indicators
            
        # MLP
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, points, input_dim]
        """
        batch_size = x.shape[0]
        num_points = x.shape[1]
        
        # Flatten batch for processing
        x_flat = x.view(-1, self.input_dim)
        
        features = [x_flat]
        
        # 1. Geodesic Distances to Reference Points
        # refs: [num_refs, dim]
        # x_flat: [N, dim]
        # Compute pairwise distances
        # We need to broadcast: x [N, 1, dim], refs [1, K, dim]
        
        # Note: Manifold.dist expects inputs of same shape or broadcastable
        # But our dist implementations might assume [..., dim]
        
        # Let's loop for safety or use expansion
        dists = []
        for k in range(self.num_references):
            ref = self.reference_points[k].unsqueeze(0).expand(x_flat.shape[0], -1)
            d = self.manifold.dist(x_flat, ref).unsqueeze(-1)
            dists.append(d)
        
        features.append(torch.cat(dists, dim=-1))
        
        # 2. Lorentzian Specific Features
        if isinstance(self.manifold, Minkowski):
            # Light Cone Coords
            if self.manifold.spatial_dim == 1:
                lc = self.manifold.light_cone_coords(x_flat)
                features.append(lc)
                
            # Causal Indicators
            # I(x \in J+(ref)) -> is ref in past of x?
            # is_causal(ref, x)
            indicators = []
            for k in range(self.num_references):
                # ref = self.reference_points[k].unsqueeze(0).expand(x_flat.shape[0], -1)
                ref = self.reference_points[k].unsqueeze(0).repeat(x_flat.shape[0], 1)
                
                # Check if x is in future of ref
                is_future = self.manifold.is_causal(ref, x_flat).float().unsqueeze(-1)
                indicators.append(is_future)
            features.append(torch.cat(indicators, dim=-1))
            
        # Concatenate all features
        feat_vec = torch.cat(features, dim=-1)
        
        # MLP
        out = self.net(feat_vec)
        
        # Reshape back
        return out.view(batch_size, num_points, -1)
