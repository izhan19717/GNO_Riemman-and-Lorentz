"""
Optimized Geometric DeepONet on Sphere (Experiment 9.0)
Improvements:
1. GELU activations (instead of ReLU)
2. Fourier features in trunk network
3. Deeper architecture (4 hidden layers)
4. Correct Laplace-Beltrami PDE loss
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import json
import time

# ==============================================================================
# 1. Utilities
# ==============================================================================

def compute_spherical_harmonic_coeffs(u_values, theta, phi, L_max=5):
    """Compute spherical harmonic coefficients from function values."""
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    coeffs = []
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, Phi, Theta)
            integrand = u_values * np.conj(Y_lm) * np.sin(Theta)
            c_lm = np.trapz(np.trapz(integrand, phi, axis=1), theta, axis=0)
            coeffs.append(np.abs(c_lm))
    return np.array(coeffs, dtype=np.float32)

def geodesic_distance_sphere(x, y, R=1.0):
    """Compute geodesic distance on sphere."""
    x_norm = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10) * R
    y_norm = y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-10) * R
    dot_prod = np.sum(x_norm * y_norm, axis=-1)
    cos_theta = np.clip(dot_prod / (R ** 2), -1.0 + 1e-7, 1.0 - 1e-7)
    d = R * np.arccos(cos_theta)
    return d

# ==============================================================================
# 2. Optimized Networks
# ==============================================================================

class SpectralBranchNetwork(nn.Module):
    """Branch network with GELU and deeper architecture."""
    def __init__(self, L_max=5, hidden_dims=[128, 128, 128, 128], p=64):
        super().__init__()
        n_coeffs = (L_max + 1) ** 2
        layers = []
        in_dim = n_coeffs
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU()) # Optimization 1: GELU
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, p))
        self.network = nn.Sequential(*layers)
        
    def forward(self, coeffs):
        return self.network(coeffs)

class GeometricTrunkNetwork(nn.Module):
    """Trunk network with Fourier features, GELU, and deeper architecture."""
    def __init__(self, n_refs=10, hidden_dims=[128, 128, 128, 128], p=64, R=1.0):
        super().__init__()
        self.n_refs = n_refs
        self.R = R
        self.register_buffer('ref_points', torch.zeros(n_refs, 3))
        
        # Input: geodesic (n_refs) + cartesian (3) + curvature (1)
        # Fourier features: sin/cos of cartesian (3*2 = 6)
        # Total input dim = n_refs + 3 + 1 + 6
        input_dim = n_refs + 3 + 1 + 6
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU()) # Optimization 1: GELU
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, p))
        self.network = nn.Sequential(*layers)
        
    def initialize_references(self, theta, phi):
        n = self.n_refs
        indices = np.arange(n) + 0.5
        phi_refs = np.arccos(1 - 2 * indices / n)
        theta_refs = np.pi * (1 + 5**0.5) * indices
        x = self.R * np.sin(phi_refs) * np.cos(theta_refs)
        y = self.R * np.sin(phi_refs) * np.sin(theta_refs)
        z = self.R * np.cos(phi_refs)
        self.ref_points = torch.FloatTensor(np.stack([x, y, z], axis=-1))
        
    def forward(self, coords_cartesian):
        batch_size, n_points, _ = coords_cartesian.shape
        coords_flat = coords_cartesian.reshape(-1, 3)
        
        # Geodesic distances
        distances = []
        for i in range(self.n_refs):
            ref = self.ref_points[i:i+1].expand(coords_flat.shape[0], -1)
            coords_norm = coords_flat / (torch.norm(coords_flat, dim=-1, keepdim=True) + 1e-10) * self.R
            ref_norm = ref / (torch.norm(ref, dim=-1, keepdim=True) + 1e-10) * self.R
            dot_prod = torch.sum(coords_norm * ref_norm, dim=-1)
            cos_theta = torch.clamp(dot_prod / (self.R ** 2), -1.0 + 1e-7, 1.0 - 1e-7)
            d = self.R * torch.acos(cos_theta)
            distances.append(d.unsqueeze(-1))
        distances = torch.cat(distances, dim=-1)
        
        # Curvature
        curvature = torch.ones(coords_flat.shape[0], 1, device=coords_flat.device) / (self.R ** 2)
        
        # Optimization 2: Fourier Features
        # Simple sin/cos embedding of coordinates
        fourier_features = torch.cat([torch.sin(coords_flat), torch.cos(coords_flat)], dim=-1)
        
        # Concatenate all features
        features = torch.cat([distances, coords_flat, curvature, fourier_features], dim=-1)
        
        output_flat = self.network(features)
        return output_flat.reshape(batch_size, n_points, -1)

class OptimizedGeometricDeepONet(nn.Module):
    def __init__(self, L_max=5, n_refs=10, p=64, R=1.0):
        super().__init__()
        self.R = R
        # Optimization 3: Deeper architecture
        self.branch = SpectralBranchNetwork(L_max=L_max, hidden_dims=[128, 128, 128, 128], p=p)
        self.trunk = GeometricTrunkNetwork(n_refs=n_refs, hidden_dims=[128, 128, 128, 128], p=p, R=R)
        
    def forward(self, coeffs, coords_cartesian):
        branch_out = self.branch(coeffs)
        trunk_out = self.trunk(coords_cartesian)
        u_pred = torch.sum(branch_out.unsqueeze(1) * trunk_out, dim=-1)
        return u_pred

    # Optimization 4: Correct PDE Loss (with efficient sampling)
    def compute_laplace_beltrami(self, coeffs, theta, phi, n_collocation=50):
        """Compute Laplace-Beltrami operator on sphere using AD.
        Uses only n_collocation points to reduce computational cost.
        """
        batch_size = coeffs.shape[0]
        
        # Sample a subset of points for PDE loss
        n_total = len(theta)
        if n_total > n_collocation:
            indices = torch.randperm(n_total)[:n_collocation]
            theta_subset = theta[indices]
            phi_subset = phi[indices]
        else:
            theta_subset = theta
            phi_subset = phi
            
        theta_tensor = theta_subset.clone().detach().requires_grad_(True)
        phi_tensor = phi_subset.clone().detach().requires_grad_(True)
        
        X = self.R * torch.sin(theta_tensor) * torch.cos(phi_tensor)
        Y = self.R * torch.sin(theta_tensor) * torch.sin(phi_tensor)
        Z = self.R * torch.cos(theta_tensor)
        coords_cartesian = torch.stack([X, Y, Z], dim=-1)
        coords_batch = coords_cartesian.unsqueeze(0).expand(batch_size, -1, -1)
        
        u = self.forward(coeffs, coords_batch)
        
        laplacian_list = []
        for b in range(batch_size):
            u_b = u[b]
            du_dtheta = torch.autograd.grad(u_b, theta_tensor, torch.ones_like(u_b), create_graph=True)[0]
            du_dphi = torch.autograd.grad(u_b, phi_tensor, torch.ones_like(u_b), create_graph=True)[0]
            d2u_dtheta2 = torch.autograd.grad(du_dtheta, theta_tensor, torch.ones_like(du_dtheta), create_graph=True)[0]
            d2u_dphi2 = torch.autograd.grad(du_dphi, phi_tensor, torch.ones_like(du_dphi), create_graph=True)[0]
            
            sin_theta = torch.sin(theta_tensor)
            cos_theta = torch.cos(theta_tensor)
            sin_theta_safe = torch.clamp(sin_theta, min=1e-6)
            
            term1 = d2u_dphi2 / (sin_theta_safe ** 2)
            term2 = d2u_dtheta2
            term3 = (cos_theta / sin_theta_safe) * du_dtheta
            laplacian_list.append(term1 + term2 + term3)
            
        return torch.stack(laplacian_list, dim=0), indices if n_total > n_collocation else torch.arange(n_total)

# ==============================================================================
# 3. Training Loop
# ==============================================================================

class GeometricPoissonDataset(Dataset):
    def __init__(self, npz_path, L_max=5, normalize=True, mean=None, std=None):
        data = np.load(npz_path)
        self.sources = data['sources']
        self.solutions = data['solutions']
        self.theta = data['theta']
        self.phi = data['phi']
        self.n_samples = len(self.sources)
        
        # Precompute coeffs
        print(f"Precomputing coefficients for {npz_path}...")
        self.source_coeffs = []
        for i in range(self.n_samples):
            coeffs = compute_spherical_harmonic_coeffs(self.sources[i], self.theta, self.phi, L_max=L_max)
            self.source_coeffs.append(coeffs)
        self.source_coeffs = np.array(self.source_coeffs)
        
        # Normalize coefficients
        if normalize:
            if mean is None:
                self.coeff_mean = np.mean(self.source_coeffs, axis=0)
                self.coeff_std = np.std(self.source_coeffs, axis=0) + 1e-8
            else:
                self.coeff_mean = mean
                self.coeff_std = std
            
            self.source_coeffs = (self.source_coeffs - self.coeff_mean) / self.coeff_std
        else:
            self.coeff_mean = None
            self.coeff_std = None
        
        # Cartesian grid
        Theta, Phi = np.meshgrid(self.theta, self.phi, indexing='ij')
        X = np.sin(Theta) * np.cos(Phi)
        Y = np.sin(Theta) * np.sin(Phi)
        Z = np.cos(Theta)
        self.coords_cartesian = torch.FloatTensor(np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1))
        self.theta_flat = torch.FloatTensor(Theta.flatten())
        self.phi_flat = torch.FloatTensor(Phi.flatten())
        
    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        return {
            'coeffs': torch.FloatTensor(self.source_coeffs[idx]),
            'coords': self.coords_cartesian,
            'u_true': torch.FloatTensor(self.solutions[idx].flatten()),
            'source': torch.FloatTensor(self.sources[idx].flatten())
        }

def custom_collate(batch):
    return {
        'coeffs': torch.stack([item['coeffs'] for item in batch]),
        'coords': batch[0]['coords'],
        'u_true': torch.stack([item['u_true'] for item in batch]),
        'source': torch.stack([item['source'] for item in batch])
    }

def train_optimized_model():
    print("="*60)
    print("EXPERIMENT 9.0: OPTIMIZED GEOMETRIC DEEPONET")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Data
    print("Loading Training Data...")
    train_dataset = GeometricPoissonDataset('train_poisson_sphere.npz', L_max=5, normalize=True)
    
    print("Loading Test Data...")
    # Use training statistics for test normalization
    test_dataset = GeometricPoissonDataset('test_poisson_sphere.npz', L_max=5, normalize=True, 
                                           mean=train_dataset.coeff_mean, std=train_dataset.coeff_std)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)
    
    # Model
    model = OptimizedGeometricDeepONet(L_max=5, n_refs=10, p=64, R=1.0).to(device)
    model.trunk.initialize_references(train_dataset.theta, train_dataset.phi)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    history = []
    
    print("\nStarting training (100 epochs) with PDE Loss (lambda=0.1)...")
    start_time = time.time()
    
    # Get grid for PDE loss
    theta_flat = train_dataset.theta_flat.to(device)
    phi_flat = train_dataset.phi_flat.to(device)
    
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        data_loss_avg = 0.0
        pde_loss_avg = 0.0
        
        for batch in train_loader:
            coeffs = batch['coeffs'].to(device)
            coords = batch['coords'].to(device)
            u_true = batch['u_true'].to(device)
            source = batch['source'].to(device)
            
            coords_batch = coords.unsqueeze(0).expand(coeffs.shape[0], -1, -1)
            
            optimizer.zero_grad()
            
            # Forward pass
            try:
                u_pred = model(coeffs, coords_batch)
                
                # Data loss
                loss_data = criterion(u_pred, u_true)
                
                # PDE loss (using 50 collocation points for efficiency)
                laplacian, indices = model.compute_laplace_beltrami(coeffs, theta_flat, phi_flat, n_collocation=50)
                source_subset = source[:, indices]
                loss_pde = torch.mean((laplacian - source_subset)**2)
                
                # Total loss
                loss = loss_data + 0.1 * loss_pde
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                data_loss_avg += loss_data.item()
                pde_loss_avg += loss_pde.item()
            except Exception as e:
                print(f"Error in training step: {e}")
                import traceback
                traceback.print_exc()
                return
            
        train_loss /= len(train_loader)
        data_loss_avg /= len(train_loader)
        pde_loss_avg /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                coeffs = batch['coeffs'].to(device)
                coords = batch['coords'].to(device)
                u_true = batch['u_true'].to(device)
                coords_batch = coords.unsqueeze(0).expand(coeffs.shape[0], -1, -1)
                u_pred = model(coeffs, coords_batch)
                val_loss += criterion(u_pred, u_true).item()
        val_loss /= len(test_loader)
        
        scheduler.step()
        history.append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'optimized_geometric_model.pth')
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} (Data: {data_loss_avg:.6f}, PDE: {pde_loss_avg:.6f}) | Val: {val_loss:.6f}")
            
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best Validation Loss: {best_loss:.6f}")
    
    # Save results
    results = {
        'best_loss': best_loss,
        'history': history,
        'time': total_time
    }
    with open('optimized_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    train_optimized_model()
