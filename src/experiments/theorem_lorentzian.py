"""
Theorem Validation: Lorentzian Causality
Compare Geometric Trunk (Causal Features) + Causality Loss vs Standard Trunk
on 1+1D Wave Equation.
"""
import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
import json
from src.geometry.minkowski import Minkowski
from src.models.branch import BranchNet
from src.models.trunk import TrunkNet
from src.models.geometric_trunk import GeometricTrunk
from src.models.gno import GeometricDeepONet
from src.physics.loss import PhysicsInformedLoss, CausalityLoss
# from src.data.synthetic import generate_wave_data

def run_lorentzian_validation():
    print("="*70)
    print("THEOREM VALIDATION: LORENTZIAN CAUSALITY")
    print("Comparing Causal Trunk vs Standard Trunk on Wave Equation")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Data Generation ---
    # We generate separate training and test sets to avoid slicing issues
    
    def generate_data(batch_size, num_points, device):
        # Branch Input: u0(x) on grid
        x_grid = torch.linspace(-1, 1, 64)
        u0 = torch.randn(batch_size, 64).to(device)
        
        # Trunk Input: Random (t, x) points
        t = torch.rand(batch_size, num_points).to(device)
        x = torch.rand(batch_size, num_points).to(device) * 2 - 1 # [-1, 1]
        trunk_input = torch.stack([t, x], dim=-1).to(device)
        
        # Target: u(t, x) = u0(x-t)
        # u0(x) = sin(k x)
        k = torch.randint(1, 5, (batch_size, 1)).float().to(device)
        u0_vals = torch.sin(k * x_grid.unsqueeze(0))
        
        # True solution u(t, x) = sin(k(x-t))
        targets = torch.sin(k.unsqueeze(1) * (x - t)).unsqueeze(-1)
        
        return u0_vals, trunk_input, targets, x_grid

    # Generate Train Set
    batch_size_train = 64
    num_points = 64
    b_train, t_train, y_train, x_grid = generate_data(batch_size_train, num_points, device)
    
    # Generate Test Set
    batch_size_test = 16
    b_test, t_test, y_test, _ = generate_data(batch_size_test, num_points, device)
    
    N_train = batch_size_train # For compatibility with existing code if needed
    
    manifold = Minkowski(spatial_dim=1)
    results = {}
    
    # --- Model A: Causal Trunk + Causality Loss ---
    print("\nTraining Model A: Causal Trunk + Causality Loss...")
    # Input dim 2 (t, x)
    # Change num_references to 64
    trunk_causal = GeometricTrunk(manifold, input_dim=2, hidden_dim=128, output_dim=64, num_references=64)
    branch_causal = BranchNet(64, 128, 64)
    model_causal = GeometricDeepONet(branch_causal, trunk_causal).to(device)
    
    opt_causal = optim.Adam(model_causal.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    causal_loss_fn = CausalityLoss(manifold)
    
    # Sample future points for causality check
    # We need input points location. u0 is at t=0, x in [-1, 1]
    input_locs = torch.stack([torch.zeros_like(x_grid), x_grid], dim=-1).to(device)
    input_locs = input_locs.unsqueeze(0).expand(N_train, -1, -1)
    
    try:
        for epoch in range(100):
            opt_causal.zero_grad()
            pred = model_causal(b_train, t_train)
            
            # MSE
            loss_data = mse_loss(pred, y_train)
            
            # Causality
            # Causality
            # Clone to ensure fresh leaf nodes and correct shapes
            b_train_c = b_train.clone().detach().requires_grad_(True)
            t_train_c = t_train.clone().detach().requires_grad_(True)
            
            loss_causal = causal_loss_fn(model_causal, b_train_c, t_train_c, input_locs)
            
            loss = loss_data + 0.1 * loss_causal
            loss.backward()
            opt_causal.step()
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: Loss {loss.item():.6f} (Causal {loss_causal.item():.6f})")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
            
    with torch.no_grad():
        pred_test = model_causal(b_test, t_test)
        err_causal = torch.norm(pred_test - y_test) / torch.norm(y_test)
        results['causal'] = err_causal.item()
        print(f"  Test Error: {err_causal.item():.4f}")
        
    # --- Model B: Standard Trunk ---
    print("\nTraining Model B: Standard Trunk (MSE Only)...")
    trunk_std = TrunkNet(input_dim=2, hidden_dim=128, output_dim=64)
    branch_std = BranchNet(64, 128, 64)
    model_std = GeometricDeepONet(branch_std, trunk_std).to(device)
    
    opt_std = optim.Adam(model_std.parameters(), lr=1e-3)
    
    for epoch in range(100):
        opt_std.zero_grad()
        pred = model_std(b_train, t_train)
        loss = mse_loss(pred, y_train)
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
    with open('theorem_lorentzian_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_lorentzian_validation()
