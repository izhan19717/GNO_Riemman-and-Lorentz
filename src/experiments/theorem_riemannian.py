"""
Theorem Validation: Riemannian Expressivity
Compare Geometric Trunk (Geodesic Features) vs Standard Trunk (Euclidean)
on PDEBench Darcy Flow (Real Data).
"""
import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
import json
from src.geometry.sphere import Sphere
from src.geometry.hyperbolic import Hyperboloid
from src.models.branch import BranchNet
from src.models.trunk import TrunkNet
from src.models.geometric_trunk import GeometricTrunk
from src.models.gno import GeometricDeepONet
from src.experiments.hyperbolic_real_data import load_darcy_data

def run_riemannian_validation():
    print("="*70)
    print("THEOREM VALIDATION: RIEMANNIAN EXPRESSIVITY")
    print("Comparing Geometric Trunk vs Standard Trunk on Darcy Flow")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data (Real World)
    data_path = "data/pdebench/2D_DarcyFlow_beta1.0_Train.hdf5"
    # Use fewer samples for speed, but enough to see difference
    nu, pressure = load_darcy_data(data_path, num_samples=200)
    
    batch_size, H, W = nu.shape
    
    # Flatten inputs
    branch_input = nu.reshape(batch_size, -1).to(device)
    targets = pressure.reshape(batch_size, -1).unsqueeze(-1).to(device)
    
    # 2. Geometry: Sphere (Positive Curvature)
    # Map [0,1]^2 to Sphere
    manifold = Sphere(dim=2)
    
    # Create Grid
    x = torch.linspace(0, 1, H)
    y = torch.linspace(0, 1, W)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Map to Sphere (theta, phi) -> (x, y, z)
    theta = X * torch.pi # [0, pi]
    phi = Y * 2 * torch.pi # [0, 2pi]
    
    sx = torch.sin(theta) * torch.cos(phi)
    sy = torch.sin(theta) * torch.sin(phi)
    sz = torch.cos(theta)
    
    points_sphere = torch.stack([sx, sy, sz], dim=-1).reshape(-1, 3).to(device)
    trunk_input = points_sphere.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Split Train/Test
    N_train = 150
    b_train, b_test = branch_input[:N_train], branch_input[N_train:]
    t_train, t_test = trunk_input[:N_train], trunk_input[N_train:]
    y_train, y_test = targets[:N_train], targets[N_train:]
    
    results = {}
    
    # --- Model A: Geometric Trunk ---
    print("\nTraining Model A: Geometric Trunk (Geodesic Features)...")
    # Input dim 3 (x,y,z)
    trunk_geo = GeometricTrunk(manifold, input_dim=3, hidden_dim=128, output_dim=64, num_references=16)
    branch_geo = BranchNet(H*W, 128, 64)
    model_geo = GeometricDeepONet(branch_geo, trunk_geo).to(device)
    
    opt_geo = optim.Adam(model_geo.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    for epoch in range(100):
        opt_geo.zero_grad()
        pred = model_geo(b_train, t_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt_geo.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Loss {loss.item():.6f}")
            
    with torch.no_grad():
        pred_test = model_geo(b_test, t_test)
        err_geo = torch.norm(pred_test - y_test) / torch.norm(y_test)
        results['geometric'] = err_geo.item()
        print(f"  Test Error: {err_geo.item():.4f}")
        
    # --- Model B: Standard Trunk ---
    print("\nTraining Model B: Standard Trunk (Euclidean Features)...")
    trunk_std = TrunkNet(input_dim=3, hidden_dim=128, output_dim=64)
    branch_std = BranchNet(H*W, 128, 64)
    model_std = GeometricDeepONet(branch_std, trunk_std).to(device)
    
    opt_std = optim.Adam(model_std.parameters(), lr=1e-3)
    
    for epoch in range(100):
        opt_std.zero_grad()
        pred = model_std(b_train, t_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt_std.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Loss {loss.item():.6f}")
            
    with torch.no_grad():
        pred_test = model_std(b_test, t_test)
        err_std = torch.norm(pred_test - y_test) / torch.norm(y_test)
        results['standard'] = err_std.item()
        print(f"  Test Error: {err_std.item():.4f}")
        
    # Save
    with open('theorem_riemannian_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_riemannian_validation()
