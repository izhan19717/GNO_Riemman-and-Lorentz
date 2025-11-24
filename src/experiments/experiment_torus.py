import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt
from src.geometry.torus import Torus
from src.models.branch import BranchNet
from src.models.trunk import TrunkNet
from src.models.geometric_trunk import GeometricTrunk
from src.models.gno import GeometricDeepONet

def run_torus_experiment():
    print("="*70)
    print("EXPERIMENT: TORUS (T^2) HEAT EQUATION")
    print("Validating Topological Generalization")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Data Generation: Heat Equation on Torus ---
    # u_t = \Delta u
    # Solution: u(x, y, t) = sum c_k exp(-4 pi^2 |k|^2 t) exp(2 pi i k \cdot x)
    # We'll use a simple analytical solution for validation.
    # u(x, y) = sin(2 pi x) * cos(2 pi y) (Stationary for simplicity or simple decay)
    # Let's do a mapping problem: u0(x, y) -> u(x, y, t=0.1)
    
    batch_size_train = 100
    batch_size_test = 20
    num_points = 100
    
    def generate_torus_data(batch_size, num_points, device):
        # Branch Input: u0 coefficients (random Fourier modes)
        # We'll represent u0 by its values on a coarse grid or just random coeffs
        # Let's use random coeffs for low freq modes
        
        # Grid for evaluating u0 (Branch Input)
        # 16x16 grid flattened = 256
        n_grid = 16
        x = torch.linspace(0, 1, n_grid)
        y = torch.linspace(0, 1, n_grid)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        grid_points = torch.stack([X.flatten(), Y.flatten()], dim=-1).to(device) # [256, 2]
        
        u0_batch = []
        y_target_batch = []
        
        # Trunk inputs: random points in domain
        trunk_inputs = torch.rand(batch_size, num_points, 2).to(device)
        
        for b in range(batch_size):
            # Random modes: kx, ky in {-2, -1, 0, 1, 2}
            # u = sum a_k sin(...) + b_k cos(...)
            u0_vals = torch.zeros(n_grid*n_grid).to(device)
            y_vals = torch.zeros(num_points).to(device)
            
            for _ in range(3): # Superimpose 3 modes
                kx = torch.randint(-2, 3, (1,)).item()
                ky = torch.randint(-2, 3, (1,)).item()
                if kx == 0 and ky == 0: continue
                
                amp = torch.randn(1).item()
                phase = torch.rand(1).item() * 2 * 3.14159
                
                # u0
                arg = 2 * 3.14159 * (kx * grid_points[:, 0] + ky * grid_points[:, 1]) + phase
                u0_vals += amp * torch.sin(arg)
                
                # Target at t=0.1
                # Decay factor = exp(-4 pi^2 (kx^2 + ky^2) * t)
                decay = torch.exp(torch.tensor(-4 * (3.14159**2) * (kx**2 + ky**2) * 0.01))
                
                # Trunk eval
                arg_trunk = 2 * 3.14159 * (kx * trunk_inputs[b, :, 0] + ky * trunk_inputs[b, :, 1]) + phase
                y_vals += amp * decay.to(device) * torch.sin(arg_trunk)
                
            u0_batch.append(u0_vals)
            y_target_batch.append(y_vals.unsqueeze(-1))
            
        return torch.stack(u0_batch), trunk_inputs, torch.stack(y_target_batch)

    print("Generating data...")
    b_train, t_train, y_train = generate_torus_data(batch_size_train, num_points, device)
    b_test, t_test, y_test = generate_torus_data(batch_size_test, num_points, device)
    
    # --- Model Setup ---
    manifold = Torus()
    
    # GNO
    print("Initializing GNO (Torus)...")
    trunk_geo = GeometricTrunk(manifold, input_dim=2, hidden_dim=128, output_dim=64, num_references=32)
    branch = BranchNet(256, 128, 64) # 16x16 input
    model_gno = GeometricDeepONet(branch, trunk_geo).to(device)
    
    # Baseline
    print("Initializing Baseline...")
    trunk_std = TrunkNet(input_dim=2, hidden_dim=128, output_dim=64)
    branch_std = BranchNet(256, 128, 64)
    model_std = GeometricDeepONet(branch_std, trunk_std).to(device)
    
    # Training
    opt_gno = optim.Adam(model_gno.parameters(), lr=1e-3)
    opt_std = optim.Adam(model_std.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    results = {'gno': [], 'std': []}
    
    print("Training...")
    for epoch in range(200):
        # GNO
        opt_gno.zero_grad()
        pred_gno = model_gno(b_train, t_train)
        loss_gno = loss_fn(pred_gno, y_train)
        loss_gno.backward()
        opt_gno.step()
        
        # Std
        opt_std.zero_grad()
        pred_std = model_std(b_train, t_train)
        loss_std = loss_fn(pred_std, y_train)
        loss_std.backward()
        opt_std.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: GNO {loss_gno.item():.6f} | Std {loss_std.item():.6f}")
            results['gno'].append(loss_gno.item())
            results['std'].append(loss_std.item())
            
    # Evaluation
    with torch.no_grad():
        pred_gno_test = model_gno(b_test, t_test)
        err_gno = torch.norm(pred_gno_test - y_test) / torch.norm(y_test)
        
        pred_std_test = model_std(b_test, t_test)
        err_std = torch.norm(pred_std_test - y_test) / torch.norm(y_test)
        
        print(f"\nTest Error (Rel L2):")
        print(f"GNO (Torus): {err_gno.item():.4f}")
        print(f"Baseline:    {err_std.item():.4f}")
        
        results['test_gno'] = err_gno.item()
        results['test_std'] = err_std.item()
        
    with open('experiment_torus_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_torus_experiment()
