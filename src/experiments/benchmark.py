import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json

# Add project root
sys.path.append(os.getcwd())

from src.geometry.sphere import Sphere
from src.models.branch import BranchNet
from src.models.trunk import TrunkNet
from src.models.gno import GeometricDeepONet
from src.data.synthetic import DataGenerator
from src.spectral.general_spectral import GeneralSpectralBasis

def compute_errors(pred, target):
    """Compute L2 and H1 (approx) errors."""
    # L2 Relative Error
    diff = pred - target
    l2_err = torch.norm(diff) / torch.norm(target)
    
    # H1 Error (Gradient based) - simplified for PoC
    # We just use L2 for now as H1 requires gradients on manifold
    return l2_err.item()

def run_convergence_study(manifold_name="sphere", resolutions=[16, 32, 64]):
    print(f"Running Convergence Study on {manifold_name}...")
    results = {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if manifold_name == "sphere":
        manifold = Sphere()
        gen = DataGenerator(manifold)
        # Use General Spectral Basis for rigorous "scientific" approach
        basis = GeneralSpectralBasis(num_eigenfunctions=64)
    else:
        raise NotImplementedError
        
    for res in resolutions:
        print(f"  Resolution: {res}")
        # Generate Data
        # For Sphere: res = nlat (nlon = 2*nlat)
        u_spatial, f_spatial = gen.generate_poisson_sphere(num_samples=100, nlat=res, nlon=2*res)
        
        # Flatten for General Basis
        batch_size = u_spatial.shape[0]
        num_points = u_spatial.shape[1] * u_spatial.shape[2]
        
        f_flat = f_spatial.reshape(batch_size, -1)
        u_flat = u_spatial.reshape(batch_size, -1)
        
        # Fit Basis on first sample grid (fixed grid)
        # Grid points
        # We need coordinates for the basis
        # Re-create grid points
        theta = torch.linspace(0, np.pi, res)
        phi = torch.linspace(0, 2*np.pi, 2*res)
        T, P = torch.meshgrid(theta, phi, indexing='ij')
        x = torch.sin(T) * torch.cos(P)
        y = torch.sin(T) * torch.sin(P)
        z = torch.cos(T)
        grid = torch.stack([x,y,z], dim=-1).reshape(-1, 3) # [N, 3]
        
        # Fit basis for current resolution grid
        basis.fit(grid, manifold)
             
        # Project f onto basis -> Branch Input
        # f_coeffs: [batch, k]
        f_coeffs_list = []
        for i in range(batch_size):
            c = basis.project(f_flat[i])
            f_coeffs_list.append(c)
        f_coeffs = torch.stack(f_coeffs_list).to(device)
        
        # Trunk Input: Grid points
        trunk_input = grid.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        
        # Targets
        targets = u_flat.unsqueeze(-1).to(device)
        
        # Model
        branch = BranchNet(64, 128, 64)
        trunk = TrunkNet(3, 128, 64)
        model = GeometricDeepONet(branch, trunk).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        
        # Train
        epochs = 100 # Short for benchmark script
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(f_coeffs, trunk_input)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()
            
        # Test Error
        with torch.no_grad():
            pred = model(f_coeffs, trunk_input)
            l2_err = compute_errors(pred, targets)
            print(f"    L2 Error: {l2_err:.4e}")
            results[res] = l2_err
            
    # Save Results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Plot
    res_vals = list(results.keys())
    err_vals = list(results.values())
    
    plt.figure()
    plt.loglog(res_vals, err_vals, '-o')
    plt.xlabel("Resolution (N)")
    plt.ylabel("Relative L2 Error")
    plt.title(f"Convergence Study: {manifold_name}")
    plt.grid(True, which="both", ls="-")
    plt.savefig("convergence_plot.png")
    print("Convergence plot saved to convergence_plot.png")

if __name__ == "__main__":
    run_convergence_study()
