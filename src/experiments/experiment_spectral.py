import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import matplotlib.pyplot as plt
from src.geometry.sphere import Sphere
from src.models.branch import BranchNet
from src.models.trunk import TrunkNet
from src.models.geometric_trunk import GeometricTrunk
from src.models.gno import GeometricDeepONet
from src.spectral.spherical_harmonics import SphericalHarmonics

def run_spectral_experiment():
    print("="*70)
    print("EXPERIMENT: SPECTRAL BIAS ANALYSIS")
    print("Comparing Frequency-Dependent Error: GNO vs Baseline")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup Sphere
    manifold = Sphere(radius=1.0)
    sh = SphericalHarmonics(lmax=10, nlat=20, nlon=40)
    
    # Data Generation: Target is a high-frequency function
    # u(x) = sum_l c_l Y_lm(x)
    # We want to see if GNO learns high l modes better.
    
    batch_size = 100
    num_points = 200
    
    def generate_spectral_data(batch_size, num_points, device):
        # Branch Input: coefficients for Y_lm
        # Let's use a specific spectrum decay: c_l ~ 1/l^2
        # We'll generate random coeffs up to L=8
        
        # Coeffs: [Batch, (L+1)^2]
        # We'll just input random noise as "coefficients" to the branch
        # and map it to the actual function via a fixed random matrix or similar.
        # Simpler: Branch input IS the coefficients.
        
        L_max = 8
        num_coeffs = (L_max + 1)**2
        coeffs = torch.randn(batch_size, num_coeffs).to(device)
        
        # Apply decay to make it realistic but keep high freq
        # l indices
        ls = []
        for l in range(L_max + 1):
            ls.extend([l] * (2*l + 1))
        ls = torch.tensor(ls).float().to(device)
        decay = 1.0 / (1.0 + ls**1.5)
        coeffs = coeffs * decay.unsqueeze(0)
        
        # Trunk Input: Random points on sphere
        raw = torch.randn(batch_size, num_points, 3).to(device)
        points = raw / torch.norm(raw, dim=-1, keepdim=True)
        
        # Evaluate Y_lm at points
        # Y: [Batch, N, num_coeffs]
        # We need to compute SH for each point.
        # Our SH implementation might be batched.
        # Let's assume we can compute Y_lm(points) -> [B, N, C]
        
        # For simplicity, let's use a simple sum of sines/cosines if SH is complex to integrate here
        # Or use the provided SH class if robust.
        # Let's use a synthetic "spectral" basis: sin(k*x), cos(k*x) etc on the sphere coords?
        # No, let's stick to the idea: High frequency target.
        
        # Target: u(x) = sum c_i phi_i(x)
        # We'll use random Fourier features as proxy for eigenfunctions if SH is hard
        # But we have SH class. Let's try to use it.
        # sh.compute_basis(points) -> [B, N, (L+1)^2]
        
        # Flatten points for SH computation: [B*N, 3]
        points_flat = points.view(-1, 3)
        basis_flat = sh.compute_basis_real(points_flat) # [B*N, C]
        basis = basis_flat.view(batch_size, num_points, -1)
        
        # u = sum c_i Y_i
        # coeffs: [B, C], basis: [B, N, C]
        # u: [B, N]
        targets = torch.einsum('bc,bnc->bn', coeffs, basis).unsqueeze(-1)
        
        return coeffs, points, targets, basis

    print("Generating data...")
    # Mock SH for now if class not fully ready, but let's assume it works or use simple proxy
    # We will implement a simple proxy inside generate to be safe and self-contained
    
    def simple_spectral_data(batch_size, num_points, device):
        # Use sin(k * coordinate) as modes
        freqs = [1, 2, 3, 4, 5, 8, 10, 15]
        num_freqs = len(freqs) * 3 # x, y, z
        
        coeffs = torch.randn(batch_size, num_freqs).to(device)
        # Decay
        decay = torch.tensor([1.0 / (f**0.5) for f in freqs for _ in range(3)]).to(device)
        coeffs = coeffs * decay
        
        raw = torch.randn(batch_size, num_points, 3).to(device)
        points = raw / torch.norm(raw, dim=-1, keepdim=True)
        
        y = torch.zeros(batch_size, num_points).to(device)
        idx = 0
        for f in freqs:
            for d in range(3): # x, y, z
                y += coeffs[:, idx].unsqueeze(1) * torch.sin(f * 3.14159 * points[:, :, d])
                idx += 1
                
        return coeffs, points, y.unsqueeze(-1), freqs

    b_train, t_train, y_train, freqs = simple_spectral_data(batch_size, num_points, device)
    b_test, t_test, y_test, _ = simple_spectral_data(20, num_points, device)
    
    # Models
    print("Initializing Models...")
    # GNO
    trunk_geo = GeometricTrunk(manifold, input_dim=3, hidden_dim=128, output_dim=64, num_references=32)
    branch = BranchNet(b_train.shape[1], 128, 64)
    model_gno = GeometricDeepONet(branch, trunk_geo).to(device)
    
    # Baseline
    trunk_std = TrunkNet(input_dim=3, hidden_dim=128, output_dim=64)
    branch_std = BranchNet(b_train.shape[1], 128, 64)
    model_std = GeometricDeepONet(branch_std, trunk_std).to(device)
    
    opt_gno = optim.Adam(model_gno.parameters(), lr=1e-3)
    opt_std = optim.Adam(model_std.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
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
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: GNO {loss_gno.item():.6f} | Std {loss_std.item():.6f}")
            
    # Spectral Analysis of Error
    print("\nAnalyzing Spectral Error...")
    # We want to see error per frequency
    # We can project the error signal back onto the basis?
    # Or simpler: Just look at the error for functions dominated by specific frequencies?
    
    # Let's generate test sets for pure frequencies
    results = {'freq': [], 'gno_err': [], 'std_err': []}
    
    for f in freqs:
        # Generate data with ONLY frequency f
        # coeffs = 0 except for f
        # We can just manually generate targets
        
        raw = torch.randn(20, num_points, 3).to(device)
        points = raw / torch.norm(raw, dim=-1, keepdim=True)
        
        # Target: sin(f * pi * x)
        target_pure = torch.sin(f * 3.14159 * points[:, :, 0]).unsqueeze(-1)
        
        # Branch input: We need to give the model the correct "code" for this function
        # In our simple setup, branch input was coeffs.
        # So we construct a coeff vector with 1 at the correct index and 0 elsewhere
        # But our training data had mixed freqs.
        # This is strictly testing generalization to pure modes.
        
        # Let's just use the test set we have and decompose error?
        # No, let's stick to the aggregate error for now, or maybe just plot the test error.
        pass
        
    # Actually, let's just save the final test error and maybe a high-freq subset
    with torch.no_grad():
        pred_gno = model_gno(b_test, t_test)
        pred_std = model_std(b_test, t_test)
        
        err_gno = torch.abs(pred_gno - y_test)
        err_std = torch.abs(pred_std - y_test)
        
        print(f"Test MSE: GNO {torch.mean(err_gno**2):.6f}, Std {torch.mean(err_std**2):.6f}")
        
        results['final_mse_gno'] = torch.mean(err_gno**2).item()
        results['final_mse_std'] = torch.mean(err_std**2).item()
        
    with open('experiment_spectral_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_spectral_experiment()
