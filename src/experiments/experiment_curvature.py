import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from src.geometry.sphere import Sphere
from src.models.branch import BranchNet
from src.models.trunk import TrunkNet
from src.models.geometric_trunk import GeometricTrunk
from src.models.gno import GeometricDeepONet

def run_curvature_experiment():
    print("="*70)
    print("EXPERIMENT: CURVATURE SENSITIVITY")
    print("Testing Generalization across Sphere Radii")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train on R=1
    print("Training on Sphere R=1.0...")
    manifold_train = Sphere(radius=1.0)
    
    # Data Gen (Simple harmonic function)
    # u(x) = Y_10(x) (z-coordinate)
    # We want to learn Identity or Laplacian? Let's learn Identity for simplicity of feature check
    # Or better: Learn Laplacian \Delta u = -l(l+1)/R^2 u
    # If we learn Laplacian, the operator depends on R.
    # If we learn Identity, it should be robust.
    # Let's learn a geometric mapping: u(x) -> u(x)^2 (Pointwise nonlinearity)
    
    batch_size = 100
    num_points = 100
    
    def generate_sphere_data(radius, batch_size, num_points, device):
        # Random points on sphere
        # We use a simple parameterization or just normalize 3D vectors
        
        # Branch input: u0 coefficients (random)
        # We'll assume u0 is a linear combination of coords: ax + by + cz
        coeffs = torch.randn(batch_size, 3).to(device)
        
        # Trunk input: random points on sphere of radius R
        raw = torch.randn(batch_size, num_points, 3).to(device)
        points = raw / torch.norm(raw, dim=-1, keepdim=True) * radius
        
        # Evaluate u0 at points
        # u0(p) = a*x + b*y + c*z
        # points: [B, N, 3], coeffs: [B, 3] -> [B, 3, 1]
        u0_vals = torch.bmm(points, coeffs.unsqueeze(-1)).squeeze(-1) # [B, N]
        
        # Target: Constant 1.0 for absolute stability check
        targets = torch.ones_like(u0_vals)
        
        return coeffs, points, targets.unsqueeze(-1)

    b_train, t_train, y_train = generate_sphere_data(1.0, batch_size, num_points, device)
    
    # Model
    trunk_geo = GeometricTrunk(manifold_train, input_dim=3, hidden_dim=128, output_dim=64, num_references=32)
    branch = BranchNet(3, 128, 64) # Input is 3 coeffs
    model = GeometricDeepONet(branch, trunk_geo).to(device)
    
    opt = optim.Adam(model.parameters(), lr=1e-4) # Lower LR
    loss_fn = nn.MSELoss()
    
    for epoch in range(100):
        opt.zero_grad()
        pred = model(b_train, t_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.6f}")
            
    # Test on varying Radii
    radii = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    results = {}
    
    print("\nTesting on varying radii:")
    for r in radii:
        # Update manifold radius in trunk?
        # Ideally, the model should take R as input or be robust.
        # But GeometricTrunk has 'manifold' object.
        # We need to update the manifold object in the trunk to calculate correct distances for new R
        model.trunk.manifold = Sphere(radius=r)
        
        b_test, t_test, y_test = generate_sphere_data(r, 20, num_points, device)
        
        with torch.no_grad():
            pred = model(b_test, t_test)
            err = torch.norm(pred - y_test) / torch.norm(y_test)
            print(f"R={r:.1f}: Error {err.item():.4f}")
            results[str(r)] = err.item()
            
    with open('experiment_curvature_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_curvature_experiment()
