"""
Experiment 1.5e: Proper Physics-Informed Loss Implementation
Implementing CORRECT Laplace-Beltrami operator for sphere
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json
import time

from geometric_deeponet_sphere import (
    SpectralBranchNetwork, GeometricTrunkNetwork, 
    GeometricPoissonDataset, custom_collate_geometric
)


class PhysicsInformedGeometricDeepONet(nn.Module):
    """
    Geometric DeepONet with PROPER physics-informed loss.
    Computes actual Laplace-Beltrami operator via automatic differentiation.
    """
    
    def __init__(self, L_max=5, n_refs=10, p=64, R=1.0):
        super().__init__()
        
        self.R = R
        self.branch = SpectralBranchNetwork(L_max=L_max, hidden_dims=[128, 128], p=p)
        self.trunk = GeometricTrunkNetwork(n_refs=n_refs, hidden_dims=[128, 128], p=p, R=R)
    
    def forward(self, coeffs, coords_cartesian):
        """Standard forward pass."""
        branch_out = self.branch(coeffs)
        trunk_out = self.trunk(coords_cartesian)
        u_pred = torch.sum(branch_out.unsqueeze(1) * trunk_out, dim=-1)
        return u_pred
    
    def compute_laplace_beltrami(self, coeffs, theta, phi):
        """
        Compute Laplace-Beltrami operator on sphere using automatic differentiation.
        
        Laplace-Beltrami on S²:
        Δu = (1/sin²θ) ∂²u/∂φ² + (1/sinθ) ∂/∂θ(sinθ ∂u/∂θ)
        
        Args:
            coeffs: (batch, n_coeffs) - SH coefficients
            theta: (n_points,) - polar angles
            phi: (n_points,) - azimuthal angles
        
        Returns:
            laplacian: (batch, n_points) - Laplace-Beltrami of u
        """
        batch_size = coeffs.shape[0]
        n_points = len(theta)
        
        # Create coordinate tensors that require gradients
        theta_tensor = theta.clone().detach().requires_grad_(True)
        phi_tensor = phi.clone().detach().requires_grad_(True)
        
        # Convert to Cartesian for model input
        X = self.R * torch.sin(theta_tensor) * torch.cos(phi_tensor)
        Y = self.R * torch.sin(theta_tensor) * torch.sin(phi_tensor)
        Z = self.R * torch.cos(theta_tensor)
        coords_cartesian = torch.stack([X, Y, Z], dim=-1)
        
        # Expand for batch
        coords_batch = coords_cartesian.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Forward pass
        u = self.forward(coeffs, coords_batch)  # (batch, n_points)
        
        # Compute gradients
        laplacian_list = []
        
        for b in range(batch_size):
            u_b = u[b]  # (n_points,)
            
            # First derivatives
            du_dtheta = torch.autograd.grad(
                u_b, theta_tensor,
                grad_outputs=torch.ones_like(u_b),
                create_graph=True, retain_graph=True
            )[0]
            
            du_dphi = torch.autograd.grad(
                u_b, phi_tensor,
                grad_outputs=torch.ones_like(u_b),
                create_graph=True, retain_graph=True
            )[0]
            
            # Second derivatives
            d2u_dtheta2 = torch.autograd.grad(
                du_dtheta, theta_tensor,
                grad_outputs=torch.ones_like(du_dtheta),
                create_graph=True, retain_graph=True
            )[0]
            
            d2u_dphi2 = torch.autograd.grad(
                du_dphi, phi_tensor,
                grad_outputs=torch.ones_like(du_dphi),
                create_graph=True, retain_graph=True
            )[0]
            
            # Laplace-Beltrami formula
            sin_theta = torch.sin(theta_tensor)
            cos_theta = torch.cos(theta_tensor)
            
            # Avoid division by zero at poles
            sin_theta_safe = torch.clamp(sin_theta, min=1e-6)
            
            # Δu = (1/sin²θ) ∂²u/∂φ² + (1/sinθ) ∂/∂θ(sinθ ∂u/∂θ)
            #    = (1/sin²θ) ∂²u/∂φ² + ∂²u/∂θ² + (cosθ/sinθ) ∂u/∂θ
            
            term1 = d2u_dphi2 / (sin_theta_safe ** 2)
            term2 = d2u_dtheta2
            term3 = (cos_theta / sin_theta_safe) * du_dtheta
            
            laplacian_b = term1 + term2 + term3
            laplacian_list.append(laplacian_b)
        
        laplacian = torch.stack(laplacian_list, dim=0)  # (batch, n_points)
        
        return laplacian
    
    def compute_pde_loss(self, coeffs, theta, phi, source):
        """
        Compute PDE residual: ||Δu - f||²
        
        Args:
            coeffs: (batch, n_coeffs)
            theta: (n_points,)
            phi: (n_points,)
            source: (batch, n_points) - source function f
        
        Returns:
            pde_loss: scalar - mean squared PDE residual
        """
        # Compute Laplace-Beltrami
        laplacian = self.compute_laplace_beltrami(coeffs, theta, phi)
        
        # PDE residual: Δu - f
        residual = laplacian - source
        
        # L2 loss
        pde_loss = torch.mean(residual ** 2)
        
        return pde_loss


def train_with_proper_pde_loss(lambda_pde=0.1, epochs=100, device='cpu'):
    """Train with PROPER physics-informed loss."""
    
    print(f"\nTraining with λ_PDE = {lambda_pde}...")
    
    # Load datasets
    train_dataset = GeometricPoissonDataset('train_poisson_sphere.npz', L_max=5)
    test_dataset = GeometricPoissonDataset('test_poisson_sphere.npz', L_max=5)
    
    # Apply normalization
    all_coeffs = np.array([train_dataset[i]['coeffs'].numpy() for i in range(len(train_dataset))])
    coeff_mean = np.mean(all_coeffs, axis=0)
    coeff_std = np.std(all_coeffs, axis=0) + 1e-8
    
    for i in range(len(train_dataset)):
        train_dataset.source_coeffs[i] = (train_dataset.source_coeffs[i] - coeff_mean) / coeff_std
    for i in range(len(test_dataset)):
        test_dataset.source_coeffs[i] = (test_dataset.source_coeffs[i] - coeff_mean) / coeff_std
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  # Smaller batch for AD
                              collate_fn=custom_collate_geometric)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             collate_fn=custom_collate_geometric)
    
    # Create model
    model = PhysicsInformedGeometricDeepONet(L_max=5, n_refs=10, p=64, R=1.0)
    model.trunk.initialize_references(train_dataset.theta, train_dataset.phi)
    model = model.to(device)
    
    # Create theta, phi tensors for PDE loss
    theta_grid, phi_grid = np.meshgrid(train_dataset.theta, train_dataset.phi, indexing='ij')
    theta_flat = torch.FloatTensor(theta_grid.flatten()).to(device)
    phi_flat = torch.FloatTensor(phi_grid.flatten()).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_data_loss': [],
        'train_pde_loss': [],
        'train_total_loss': [],
        'test_loss': []
    }
    
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_data_loss = 0.0
        epoch_pde_loss = 0.0
        epoch_total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            coeffs = batch['coeffs'].to(device)
            coords = batch['coords'].to(device)
            u_true = batch['u_true'].to(device)
            
            batch_size = coeffs.shape[0]
            coords_batch = coords.unsqueeze(0).expand(batch_size, -1, -1)
            
            optimizer.zero_grad()
            
            # Forward pass
            u_pred = model(coeffs, coords_batch)
            
            # Data loss
            loss_data = criterion(u_pred, u_true)
            
            # PDE loss (compute on subset of batches to save time)
            if lambda_pde > 0 and batch_idx % 2 == 0:  # Every other batch
                # Get source functions (need to reconstruct from dataset)
                source_batch = []
                for i in range(batch_size):
                    idx = batch_idx * batch_size + i
                    if idx < len(train_dataset):
                        source = torch.FloatTensor(train_dataset.sources[idx].flatten()).to(device)
                        source_batch.append(source)
                
                if len(source_batch) > 0:
                    source_batch = torch.stack(source_batch[:batch_size])
                    loss_pde = model.compute_pde_loss(coeffs, theta_flat, phi_flat, source_batch)
                else:
                    loss_pde = torch.tensor(0.0, device=device)
            else:
                loss_pde = torch.tensor(0.0, device=device)
            
            # Total loss
            total_loss = loss_data + lambda_pde * loss_pde
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_data_loss += loss_data.item()
            epoch_pde_loss += loss_pde.item() if isinstance(loss_pde, torch.Tensor) else loss_pde
            epoch_total_loss += total_loss.item()
        
        epoch_data_loss /= len(train_loader)
        epoch_pde_loss /= len(train_loader)
        epoch_total_loss /= len(train_loader)
        
        history['train_data_loss'].append(epoch_data_loss)
        history['train_pde_loss'].append(epoch_pde_loss)
        history['train_total_loss'].append(epoch_total_loss)
        
        # Testing
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                coeffs = batch['coeffs'].to(device)
                coords = batch['coords'].to(device)
                u_true = batch['u_true'].to(device)
                
                batch_size = coeffs.shape[0]
                coords_batch = coords.unsqueeze(0).expand(batch_size, -1, -1)
                
                u_pred = model(coeffs, coords_batch)
                loss = criterion(u_pred, u_true)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        history['test_loss'].append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), f'proper_pde_model_lambda{lambda_pde}.pth')
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Data: {epoch_data_loss:.6f} | "
                  f"PDE: {epoch_pde_loss:.6f} | Total: {epoch_total_loss:.6f} | "
                  f"Test: {test_loss:.6f}")
    
    return history, best_test_loss


if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENT 1.5e: PROPER PHYSICS-INFORMED LOSS")
    print("="*70)
    print("\nImplementing CORRECT Laplace-Beltrami operator...")
    print("Formula: Δu = (1/sin²θ)∂²u/∂φ² + ∂²u/∂θ² + (cosθ/sinθ)∂u/∂θ")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test different lambda values
    lambda_values = [0.0, 0.1, 1.0]
    results = {}
    
    for lam in lambda_values:
        print("\n" + "="*70)
        print(f"TRAINING WITH λ = {lam}")
        print("="*70)
        
        history, best_loss = train_with_proper_pde_loss(lambda_pde=lam, epochs=50, device=device)
        
        results[lam] = {
            'history': history,
            'best_test_loss': best_loss,
            'final_test_loss': history['test_loss'][-1]
        }
        
        print(f"\n  Best test loss: {best_loss:.6f}")
        print(f"  Final test loss: {history['test_loss'][-1]:.6f}")
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON: DATA-ONLY vs PHYSICS-INFORMED")
    print("="*70)
    
    no_pde = results[0.0]['best_test_loss']
    with_pde_01 = results[0.1]['best_test_loss']
    with_pde_10 = results[1.0]['best_test_loss']
    
    print(f"\nNo PDE loss (λ=0.0):   {no_pde:.6f}")
    print(f"With PDE (λ=0.1):      {with_pde_01:.6f}")
    print(f"With PDE (λ=1.0):      {with_pde_10:.6f}")
    
    if with_pde_01 < no_pde:
        improvement = (no_pde - with_pde_01) / no_pde * 100
        print(f"\n✓ PDE loss helps! {improvement:.1f}% improvement with λ=0.1")
    
    if with_pde_10 < no_pde:
        improvement = (no_pde - with_pde_10) / no_pde * 100
        print(f"✓ PDE loss helps! {improvement:.1f}% improvement with λ=1.0")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training curves
    ax1 = axes[0]
    for lam in lambda_values:
        ax1.semilogy(results[lam]['history']['test_loss'], 
                    label=f'λ={lam}', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Test Loss', fontsize=12)
    ax1.set_title('Proper PDE Loss Impact', fontsize=13, weight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final comparison
    ax2 = axes[1]
    lambdas = list(results.keys())
    best_losses = [results[lam]['best_test_loss'] for lam in lambdas]
    
    bars = ax2.bar([str(l) for l in lambdas], best_losses, 
                   color=['steelblue', 'coral', 'lightgreen'],
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_xlabel('PDE Weight (λ)', fontsize=12)
    ax2.set_ylabel('Best Test Loss', fontsize=12)
    ax2.set_title('Impact of Physics-Informed Loss', fontsize=13, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, best_losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}',
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig('proper_pde_loss_results.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: proper_pde_loss_results.png")
    plt.close()
    
    # Save results
    results_json = {
        str(lam): {
            'best_test_loss': float(results[lam]['best_test_loss']),
            'final_test_loss': float(results[lam]['final_test_loss'])
        } for lam in lambda_values
    }
    
    with open('proper_pde_loss_results.json', 'w') as f:
        json.dump(results_json, f, indent=4)
    
    print("\n" + "="*70)
    print("EXPERIMENT 1.5e COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  - proper_pde_model_lambda*.pth (trained models)")
    print("  - proper_pde_loss_results.png")
    print("  - proper_pde_loss_results.json")
