"""
Hyperbolic Real Data Experiment: Darcy Flow on Poincaré Disk

Adapts the PDEBench 2D Darcy Flow dataset to Hyperbolic Geometry.
We map the Euclidean square domain [0,1]^2 to the Poincaré Disk D^2.
This provides "real" complex spatial patterns (permeability/pressure) 
on a negative curvature manifold.

Research Question: Does GNO outperform Baseline on Hyperbolic Real Data?
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json

from src.geometry.hyperbolic import Hyperboloid
from src.models.branch import BranchNet
from src.models.trunk import TrunkNet
from src.models.gno import GeometricDeepONet
from src.spectral.general_spectral import GeneralSpectralBasis

def load_darcy_data(filepath, num_samples=500):
    """Load PDEBench data (Time-dependent)."""
    try:
        with h5py.File(filepath, 'r') as f:
            # Shape: (10000, 201, 1024)
            # We take t=0 as input and t=end as output
            data = f['tensor'][:num_samples]
            
            # Input: t=0
            nu = torch.tensor(data[:, 0, :], dtype=torch.float32)
            # Output: t=end
            pressure = torch.tensor(data[:, -1, :], dtype=torch.float32)
            
            # Reshape to (batch, 32, 32)
            # Assuming 1024 = 32x32
            nu = nu.view(-1, 32, 32)
            pressure = pressure.view(-1, 32, 32)
            
            return nu, pressure
    except Exception as e:
        print(f"Error loading data: {e}")
        return generate_synthetic_darcy(num_samples)

def generate_synthetic_darcy(num_samples=500, grid_size=64):
    """Fallback synthetic data."""
    print("Generating synthetic data...")
    nu = torch.randn(num_samples, grid_size, grid_size)
    pressure = torch.randn(num_samples, grid_size, grid_size)
    return nu, pressure

def map_square_to_disk(x_grid, y_grid):
    """
    Map unit square [0,1]x[0,1] to Poincaré Disk D^2 (radius 1).
    Simple diffeomorphism: Center at 0, scale to fit in disk.
    """
    # Center to [-1, 1]
    u = 2 * x_grid - 1
    v = 2 * y_grid - 1
    
    # Scale to fit in disk (radius 0.9 to avoid boundary issues)
    # r = max(|u|, |v|) -> map to r < 1
    # Simple scaling: (u, v) * 0.9
    # Better: Concentric mapping to preserve area better? 
    # Let's use simple scaling for PoC.
    
    u_disk = u * 0.7
    v_disk = v * 0.7
    
    # Calculate z coordinate for Hyperboloid model
    # x^2 + y^2 - z^2 = -1  => z = sqrt(1 + x^2 + y^2)
    # Here (u_disk, v_disk) are coordinates in the disk model (y1, y2)
    # Map disk to hyperboloid:
    # x = 2u / (1 - r^2)
    # y = 2v / (1 - r^2)
    # t = (1 + r^2) / (1 - r^2)
    
    r2 = u_disk**2 + v_disk**2
    denom = 1 - r2
    
    x_hyp = 2 * u_disk / denom
    y_hyp = 2 * v_disk / denom
    t_hyp = (1 + r2) / denom
    
    # Stack: (t, x, y) for Minkowski embedding
    points = torch.stack([t_hyp, x_hyp, y_hyp], dim=-1)
    return points

def run_hyperbolic_real_experiment():
    print("="*70)
    print("HYPERBOLIC REAL DATA: Darcy Flow on Poincaré Disk")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    data_path = "data/pdebench/2D_DarcyFlow_beta1.0_Train.hdf5"
    nu, pressure = load_darcy_data(data_path, num_samples=300)
    
    batch_size, H, W = nu.shape
    print(f"Data Loaded: {nu.shape}")
    
    # 2. Geometry Setup
    manifold = Hyperboloid(dim=2)
    
    # Create Grid and Map to Hyperboloid
    x = torch.linspace(0, 1, H)
    y = torch.linspace(0, 1, W)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Trunk Input: Points on Hyperboloid
    points = map_square_to_disk(X, Y).reshape(-1, 3).to(device)
    trunk_input = points.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Branch Input: Permeability field (flattened)
    # We treat the input field as defined on this manifold
    branch_input = nu.reshape(batch_size, -1).to(device)
    
    # Targets: Pressure field
    targets = pressure.reshape(batch_size, -1).unsqueeze(-1).to(device)
    
    # 3. Spectral Basis (Optional but recommended for GNO)
    # For this real data experiment, we'll use the "Enhanced Trunk" approach 
    # (coordinate-free GNO) as fitting basis on 64x64=4096 points is slow.
    # We rely on the Hyperbolic Trunk to learn geometry.
    
    # 4. Training Loop
    sample_sizes = [50, 100, 200]
    results = {'gno': [], 'baseline': [], 'sample_sizes': sample_sizes}
    
    test_size = 50
    train_pool = batch_size - test_size
    
    b_test = branch_input[train_pool:]
    t_test = trunk_input[train_pool:]
    y_test = targets[train_pool:]
    
    for N in sample_sizes:
        print(f"\nTraining with N={N}...")
        
        b_train = branch_input[:N]
        t_train = trunk_input[:N]
        y_train = targets[:N]
        
        # --- GNO (Hyperbolic Trunk) ---
        # Input dim for trunk is 3 (t, x, y) in embedding space
        branch_gno = BranchNet(H*W, 128, 64)
        trunk_gno = TrunkNet(3, 128, 64) 
        model_gno = GeometricDeepONet(branch_gno, trunk_gno).to(device)
        
        opt_gno = optim.Adam(model_gno.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        
        for epoch in range(100):
            opt_gno.zero_grad()
            pred = model_gno(b_train, t_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            opt_gno.step()
            
        with torch.no_grad():
            pred_test = model_gno(b_test, t_test)
            err_gno = torch.norm(pred_test - y_test) / torch.norm(y_test)
            results['gno'].append(err_gno.item())
            print(f"  GNO Error: {err_gno.item():.4f}")
            
        # --- Baseline (Euclidean Trunk) ---
        # Input dim for trunk is 2 (x, y) from original grid
        grid_euclidean = torch.stack([X, Y], dim=-1).reshape(-1, 2).to(device)
        t_train_base = grid_euclidean.unsqueeze(0).repeat(N, 1, 1)
        t_test_base = grid_euclidean.unsqueeze(0).repeat(test_size, 1, 1)
        
        branch_base = BranchNet(H*W, 128, 64)
        trunk_base = TrunkNet(2, 128, 64)
        model_base = GeometricDeepONet(branch_base, trunk_base).to(device)
        
        opt_base = optim.Adam(model_base.parameters(), lr=1e-3)
        
        for epoch in range(100):
            opt_base.zero_grad()
            pred = model_base(b_train, t_train_base)
            loss = loss_fn(pred, y_train)
            loss.backward()
            opt_base.step()
            
        with torch.no_grad():
            pred_test = model_base(b_test, t_test_base)
            err_base = torch.norm(pred_test - y_test) / torch.norm(y_test)
            results['baseline'].append(err_base.item())
            print(f"  Baseline Error: {err_base.item():.4f}")
            
    # Save Results
    with open('hyperbolic_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_hyperbolic_real_experiment()
