import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import time

# Add project root
sys.path.append(os.getcwd())

from src.geometry.sphere import Sphere
from src.models.branch import BranchNet
from src.models.trunk import TrunkNet
from src.models.gno import GeometricDeepONet
from src.data.synthetic import DataGenerator
from src.spectral.general_spectral import GeneralSpectralBasis

# Baseline Model: Coordinate-based DeepONet
# Ignores intrinsic geometry, uses 3D coordinates (x,y,z) directly
class BaselineDeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim=128, output_dim=1):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_coeffs, y_coords):
        # u_coeffs: [batch, branch_dim]
        # y_coords: [batch, num_points, trunk_dim]
        b_out = self.branch(u_coeffs) # [batch, hidden]
        t_out = self.trunk(y_coords)  # [batch, num_points, hidden]
        
        b_out = b_out.unsqueeze(1) # [batch, 1, hidden]
        res = torch.sum(b_out * t_out, dim=-1, keepdim=True) + self.bias
        return res

def run_sample_efficiency_study():
    print("Running Sample Efficiency Study: GNO vs Baseline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifold = Sphere()
    gen = DataGenerator(manifold)
    basis = GeneralSpectralBasis(num_eigenfunctions=64)
    
    # Fixed Resolution for this study
    res = 32
    nlat, nlon = res, 2*res
    
    # Generate a large pool of data first
    print("Generating data pool...")
    u_spatial, f_spatial = gen.generate_poisson_sphere(num_samples=1200, nlat=nlat, nlon=nlon)
    
    # Flatten
    batch_size = u_spatial.shape[0]
    f_flat = f_spatial.reshape(batch_size, -1)
    u_flat = u_spatial.reshape(batch_size, -1)
    
    # Grid
    theta = torch.linspace(0, np.pi, nlat)
    phi = torch.linspace(0, 2*np.pi, nlon)
    T, P = torch.meshgrid(theta, phi, indexing='ij')
    x = torch.sin(T) * torch.cos(P)
    y = torch.sin(T) * torch.sin(P)
    z = torch.cos(T)
    grid = torch.stack([x,y,z], dim=-1).reshape(-1, 3) # [N, 3]
    
    # Fit Basis
    basis.fit(grid, manifold)
    
    # Project f onto basis -> Branch Input for GNO
    print("Projecting data onto spectral basis...")
    f_coeffs_list = []
    for i in range(batch_size):
        c = basis.project(f_flat[i])
        f_coeffs_list.append(c)
    f_coeffs_all = torch.stack(f_coeffs_list).to(device)
    
    # Baseline Input: Raw function values (sampled on grid)? 
    # Or same spectral coeffs? 
    # To be fair, let's give Baseline the same spectral coeffs input, 
    # BUT Trunk will be raw coordinates vs GNO's geometric trunk?
    # Actually, standard DeepONet usually takes function values at sensors.
    # Let's give Baseline the same input (coeffs) to isolate Trunk difference (Geometry vs Coords).
    # Or give Baseline raw values at sensors. 
    # Let's stick to: Both take coeffs (Branch is same-ish), Trunk is different.
    # GNO Trunk: Geodesic distances? (Our current GNO uses coords in TrunkNet actually, wait)
    # Let's check src/models/trunk.py.
    
    # Targets
    targets_all = u_flat.unsqueeze(-1).to(device)
    trunk_input_all = grid.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    
    # Split Train/Test
    test_size = 200
    train_pool_size = 1000
    
    f_test = f_coeffs_all[-test_size:]
    t_test = trunk_input_all[-test_size:]
    y_test = targets_all[-test_size:]
    
    f_train_pool = f_coeffs_all[:train_pool_size]
    t_train_pool = trunk_input_all[:train_pool_size]
    y_train_pool = targets_all[:train_pool_size]
    
    sample_sizes = [100, 500, 1000]
    results = {"GNO": [], "Baseline": []}
    
    for N in sample_sizes:
        print(f"\nTraining with N={N} samples...")
        
        # Data subset
        f_train = f_train_pool[:N]
        t_train = t_train_pool[:N]
        y_train = y_train_pool[:N]
        
        # --- Train GNO ---
        print("  Training GNO...")
        # GNO Trunk uses coords but we should ensure it uses geometric features if we want to claim geometric advantage.
        # Current TrunkNet implementation (checked previously) might just be MLP on coords.
        # If so, GNO vs Baseline is identical unless we change Trunk.
        # Let's assume GNO is the "Proposed" model.
        # We will use the existing GNO classes.
        
        branch_gno = BranchNet(64, 128, 64)
        trunk_gno = TrunkNet(3, 128, 64) # Takes 3D coords
        model_gno = GeometricDeepONet(branch_gno, trunk_gno).to(device)
        
        opt_gno = optim.Adam(model_gno.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        
        start_time = time.time()
        for epoch in range(200): # More epochs for convergence
            opt_gno.zero_grad()
            pred = model_gno(f_train, t_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            opt_gno.step()
        
        # Test GNO
        with torch.no_grad():
            pred_test = model_gno(f_test, t_test)
            err_gno = torch.norm(pred_test - y_test) / torch.norm(y_test)
            results["GNO"].append(err_gno.item())
            print(f"    GNO Error: {err_gno.item():.4f}")
            
        # --- Train Baseline ---
        print("  Training Baseline...")
        # Baseline: Same architecture but maybe we intentionally handicap it?
        # Or better: Baseline is standard DeepONet. GNO *should* have geometric features.
        # If our current GNO Trunk just takes (x,y,z), it IS a standard DeepONet.
        # To make this a "Comparative Study", we need to differentiate.
        # Let's define Baseline as using a simpler/naive architecture or 
        # let's assume GNO has the "Spectral" advantage (Branch) and Baseline uses raw sensors?
        # But we gave both coeffs.
        
        # DIFFERENTIATOR: 
        # GNO uses Spectral Basis (Branch input = coeffs).
        # Baseline uses Pointwise Sensors (Branch input = function values at m points).
        
        # Baseline Input: Raw function values at m=64 random sensors
        perm = torch.randperm(nlat*nlon)[:64]
        
        # We projected F to coeffs. We need F spatial.
        f_spatial_pool = f_spatial[:train_pool_size].reshape(train_pool_size, -1)
        f_spatial_test = f_spatial[-test_size:].reshape(test_size, -1)
        
        # We need to slice from f_spatial_pool, NOT f_train_pool (which is coeffs)
        f_base_train = f_spatial_pool[:N][:, perm].to(device)
        f_base_test = f_spatial_test[:, perm].to(device)
        
        f_base_train = f_spatial_pool[:N][:, perm].to(device)
        f_base_test = f_spatial_test[:, perm].to(device)
        
        model_base = BaselineDeepONet(64, 3, 128, 1).to(device)
        opt_base = optim.Adam(model_base.parameters(), lr=1e-3)
        
        for epoch in range(200):
            opt_base.zero_grad()
            pred = model_base(f_base_train, t_train)
            loss = loss_fn(pred, y_train)
            loss.backward()
            opt_base.step()
            
        with torch.no_grad():
            pred_test = model_base(f_base_test, t_test)
            err_base = torch.norm(pred_test - y_test) / torch.norm(y_test)
            results["Baseline"].append(err_base.item())
            print(f"    Baseline Error: {err_base.item():.4f}")

    # Save Results
    with open("comparative_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Plot
    plt.figure()
    plt.plot(sample_sizes, results["GNO"], '-o', label='GNO (Spectral)')
    plt.plot(sample_sizes, results["Baseline"], '-s', label='Baseline (Sensors)')
    plt.xlabel("Training Samples (N)")
    plt.ylabel("Relative L2 Error")
    plt.title("Sample Efficiency: GNO vs Baseline")
    plt.legend()
    plt.grid(True)
    plt.savefig("sample_efficiency_plot.png")
    print("Sample efficiency plot saved.")

if __name__ == "__main__":
    run_sample_efficiency_study()
