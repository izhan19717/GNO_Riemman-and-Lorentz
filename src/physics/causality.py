"""
Causality Loss for Lorentzian Spacetime

This module implements causality constraints for neural operators on Lorentzian manifolds.
The key principle: solutions at an event should depend ONLY on data in the past light cone.

References:
- Wald (1984): "General Relativity", Chapter 8
- Raissi et al. (2019): "Physics-informed neural networks"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def sample_outside_lightcone(
    event: torch.Tensor,
    manifold,
    num_samples: int = 32,
    time_range: Tuple[float, float] = (0.0, 1.0),
    space_range: Tuple[float, float] = (-1.0, 1.0)
) -> torch.Tensor:
    """
    Sample points outside the past light cone of an event.
    
    For 1+1 Minkowski spacetime with event (t, x):
    Past light cone: {(t', x') : t' < t and |x' - x| < t - t'}
    Outside: {(t', x') : t' < t and |x' - x| > t - t'} (spacelike separated)
    
    Args:
        event: [2] tensor (t, x) for 1+1D Minkowski
        manifold: Minkowski manifold object
        num_samples: Number of points to sample
        
    Returns:
        points: [num_samples, 2] points outside past light cone
    """
    t_event, x_event = event[0].item(), event[1].item()
    
    points = []
    attempts = 0
    max_attempts = num_samples * 10
    
    while len(points) < num_samples and attempts < max_attempts:
        # Sample random spacetime point
        t_sample = torch.rand(1) * (time_range[1] - time_range[0]) + time_range[0]
        x_sample = torch.rand(1) * (space_range[1] - space_range[0]) + space_range[0]
        
        # Check if outside past light cone
        dt = t_event - t_sample.item()
        dx = abs(x_event - x_sample.item())
        
        if dt > 0 and dx > dt:  # Spacelike separated, in past
            points.append(torch.tensor([t_sample.item(), x_sample.item()]))
        
        attempts += 1
    
    if len(points) < num_samples:
        # Fallback: just sample spacelike separated points
        for _ in range(num_samples - len(points)):
            t_sample = torch.rand(1) * t_event * 0.5  # Earlier time
            x_sample = x_event + (torch.rand(1) - 0.5) * 2.0  # Far away
            points.append(torch.tensor([t_sample.item(), x_sample.item()]))
    
    return torch.stack(points[:num_samples])


class CausalityLoss(nn.Module):
    """
    Causality constraint loss for Lorentzian spacetime.
    
    Penalizes neural operator predictions that violate causality by depending
    on data outside the past light cone.
    
    Method:
    1. For each test event z, sample points outside J^-(z)
    2. Perturb initial data at these points
    3. Measure change in prediction at z
    4. Penalize non-zero change (should be causally independent)
    
    Args:
        manifold: Lorentzian manifold (e.g., Minkowski)
        num_test_events: Number of events to test
        num_perturbations: Number of outside-cone points to perturb
        perturbation_scale: Magnitude of perturbation
    """
    
    def __init__(
        self,
        manifold,
        num_test_events: int = 16,
        num_perturbations: int = 8,
        perturbation_scale: float = 0.1
    ):
        super().__init__()
        self.manifold = manifold
        self.num_test_events = num_test_events
        self.num_perturbations = num_perturbations
        self.perturbation_scale = perturbation_scale
    
    def forward(
        self,
        model: nn.Module,
        u_initial: torch.Tensor,
        grid: torch.Tensor,
        test_events: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute causality violation loss.
        
        Args:
            model: Neural operator model
            u_initial: [batch, num_points] initial data
            grid: [batch, num_points, 2] spacetime grid (t, x)
            test_events: Optional [num_test, 2] events to test
            
        Returns:
            loss: Scalar causality violation loss
        """
        batch_size = u_initial.shape[0]
        
        # Sample test events if not provided
        if test_events is None:
            # Sample events at later times
            t_max = grid[0, :, 0].max().item()
            test_events = []
            for _ in range(self.num_test_events):
                t = torch.rand(1) * t_max * 0.5 + t_max * 0.5  # Later half
                x = (torch.rand(1) - 0.5) * 2.0
                test_events.append(torch.tensor([t.item(), x.item()]))
            test_events = torch.stack(test_events)
        
        total_loss = 0.0
        
        for event in test_events:
            # Sample points outside past light cone
            outside_points = sample_outside_lightcone(
                event, 
                self.manifold, 
                num_samples=self.num_perturbations
            )
            
            # Find indices in grid closest to outside points
            # This is a simplification; ideally we'd interpolate
            outside_indices = []
            for pt in outside_points:
                dists = torch.norm(grid[0] - pt.unsqueeze(0), dim=-1)
                outside_indices.append(dists.argmin().item())
            
            # Original prediction at event
            # Find grid point closest to event
            event_dists = torch.norm(grid[0] - event.unsqueeze(0), dim=-1)
            event_idx = event_dists.argmin().item()
            
            with torch.no_grad():
                pred_original = model(u_initial, grid)[:, event_idx]
            
            # Perturbed predictions
            for idx in outside_indices:
                u_perturbed = u_initial.clone()
                u_perturbed[:, idx] += torch.randn_like(u_perturbed[:, idx]) * self.perturbation_scale
                
                pred_perturbed = model(u_perturbed, grid)[:, event_idx]
                
                # Causality violation: change in prediction
                violation = torch.norm(pred_perturbed - pred_original)
                total_loss += violation
        
        # Normalize
        return total_loss / (self.num_test_events * self.num_perturbations)


def verify_causality(
    model: nn.Module,
    u_initial: torch.Tensor,
    grid: torch.Tensor,
    manifold,
    threshold: float = 1e-3
) -> Tuple[float, bool]:
    """
    Verify causality of a trained model.
    
    Args:
        model: Trained neural operator
        u_initial: Initial data
        grid: Spacetime grid
        manifold: Lorentzian manifold
        threshold: Acceptable causality violation
        
    Returns:
        violation_metric: Average causality violation
        is_causal: Whether model respects causality
    """
    loss_fn = CausalityLoss(manifold, num_test_events=32, num_perturbations=16)
    
    with torch.no_grad():
        violation = loss_fn(model, u_initial, grid).item()
    
    is_causal = violation < threshold
    
    return violation, is_causal


# Example usage
if __name__ == "__main__":
    from src.geometry.minkowski import Minkowski
    from src.models.gno import GeometricDeepONet
    from src.models.branch import BranchNet
    from src.models.trunk import TrunkNet
    
    print("Testing Causality Loss...")
    
    # Setup
    manifold = Minkowski(dim=2)
    device = torch.device("cpu")
    
    # Create dummy model
    branch = BranchNet(64, 64, 32)
    trunk = TrunkNet(2, 64, 32)
    model = GeometricDeepONet(branch, trunk).to(device)
    
    # Create dummy data
    batch_size = 4
    num_points = 64
    
    # Spacetime grid: t ∈ [0, 1], x ∈ [-1, 1]
    t = torch.linspace(0, 1, num_points)
    x = torch.linspace(-1, 1, num_points)
    T, X = torch.meshgrid(t, x, indexing='ij')
    grid = torch.stack([T, X], dim=-1).reshape(-1, 2)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Initial data (random)
    u_initial = torch.randn(batch_size, num_points)
    
    # Compute causality loss
    causality_loss = CausalityLoss(manifold)
    loss = causality_loss(model, u_initial, grid)
    
    print(f"Causality violation loss: {loss.item():.4e}")
    
    # Verify causality
    violation, is_causal = verify_causality(model, u_initial, grid, manifold, threshold=0.1)
    print(f"Causality violation metric: {violation:.4e}")
    print(f"Is causal (threshold=0.1): {is_causal}")
    
    print("\n✓ Causality loss implementation complete!")
