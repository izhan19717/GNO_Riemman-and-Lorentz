"""
Experiment 9.2: Noise Robustness Study - Training Script
Train Geometric DeepONet on noisy data and measure degradation.

This is the simplest experiment to start with since we can reuse
the existing optimized model architecture.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import time
from scipy.special import sph_harm

# Import the optimized model
import sys
sys.path.append('.')

# ==============================================================================
# Dataset for Noisy Data
# ==============================================================================

def compute_spherical_harmonic_coeffs(u_values, theta, phi, L_max=5):
    """Compute spherical harmonic coefficients."""
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    coeffs = []
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, Phi, Theta)
            integrand = u_values * np.conj(Y_lm) * np.sin(Theta)
            c_lm = np.trapz(np.trapz(integrand, phi, axis=1), theta, axis=0)
            coeffs.append(np.abs(c_lm))
    return np.array(coeffs, dtype=np.float32)

class NoisyPoissonDataset(Dataset):
    def __init__(self, npz_path, L_max=5, normalize=True, mean=None, std=None):
        data = np.load(npz_path)
        self.sources = data['sources']
        self.solutions = data['solutions']
        self.theta = data['theta']
        self.phi = data['phi']
        self.n_samples = len(self.sources)
        
        print(f"Precomputing coefficients for {npz_path}...")
        self.source_coeffs = []
        for i in range(self.n_samples):
            coeffs = compute_spherical_harmonic_coeffs(self.sources[i], self.theta, self.phi, L_max=L_max)
            self.source_coeffs.append(coeffs)
        self.source_coeffs = np.array(self.source_coeffs)
        
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
        
        Theta, Phi = np.meshgrid(self.theta, self.phi, indexing='ij')
        X = np.sin(Theta) * np.cos(Phi)
        Y = np.sin(Theta) * np.sin(Phi)
        Z = np.cos(Theta)
        self.coords_cartesian = torch.FloatTensor(np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1))
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'coeffs': torch.FloatTensor(self.source_coeffs[idx]),
            'coords': self.coords_cartesian,
            'u_true': torch.FloatTensor(self.solutions[idx].flatten())
        }

def custom_collate(batch):
    return {
        'coeffs': torch.stack([item['coeffs'] for item in batch]),
        'coords': batch[0]['coords'],
        'u_true': torch.stack([item['u_true'] for item in batch])
    }

# ==============================================================================
# Simple Optimized Model (without PDE loss for speed)
# ==============================================================================

class OptimizedGeometricDeepONet(nn.Module):
    def __init__(self, L_max=5, n_refs=10, p=64, R=1.0):
        super().__init__()
        self.R = R
        n_coeffs = (L_max + 1) ** 2
        
        # Branch with Fourier features
        self.branch = nn.Sequential(
            nn.Linear(n_coeffs, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, p)
        )
        
        # Trunk
        self.trunk_net = nn.Sequential(
            nn.Linear(n_refs + 3 + 1 + 6, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, p)
        )
        
        self.n_refs = n_refs
        self.register_buffer('ref_points', torch.zeros(n_refs, 3))
        
    def initialize_references(self, theta, phi):
        n = self.n_refs
        indices = np.arange(n) + 0.5
        phi_refs = np.arccos(1 - 2 * indices / n)
        theta_refs = np.pi * (1 + 5**0.5) * indices
        x = self.R * np.sin(phi_refs) * np.cos(theta_refs)
        y = self.R * np.sin(phi_refs) * np.sin(theta_refs)
        z = self.R * np.cos(phi_refs)
        self.ref_points = torch.FloatTensor(np.stack([x, y, z], axis=-1))
        
    def trunk(self, coords_cartesian):
        batch_size, n_points, _ = coords_cartesian.shape
        coords_flat = coords_cartesian.reshape(-1, 3)
        
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
        
        curvature = torch.ones(coords_flat.shape[0], 1, device=coords_flat.device) / (self.R ** 2)
        fourier_features = torch.cat([torch.sin(coords_flat), torch.cos(coords_flat)], dim=-1)
        features = torch.cat([distances, coords_flat, curvature, fourier_features], dim=-1)
        
        output_flat = self.trunk_net(features)
        return output_flat.reshape(batch_size, n_points, -1)
        
    def forward(self, coeffs, coords_cartesian):
        branch_out = self.branch(coeffs)
        trunk_out = self.trunk(coords_cartesian)
        u_pred = torch.sum(branch_out.unsqueeze(1) * trunk_out, dim=-1)
        return u_pred

# ==============================================================================
# Training Function
# ==============================================================================

def train_on_noisy_data(noise_level, train_file, test_file='test_poisson_sphere.npz'):
    """Train model on specific noise level."""
    print(f"\n{'='*60}")
    print(f"TRAINING ON {int(noise_level*100)}% NOISE")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print(f"\nLoading training data from {train_file}...")
    train_dataset = NoisyPoissonDataset(train_file, L_max=5, normalize=True)
    
    print(f"Loading test data from {test_file}...")
    test_dataset = NoisyPoissonDataset(
        test_file, 
        L_max=5, 
        normalize=True,
        mean=train_dataset.coeff_mean,
        std=train_dataset.coeff_std
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    
    # Model
    model = OptimizedGeometricDeepONet(L_max=5, n_refs=10, p=64, R=1.0).to(device)
    model.initialize_references(train_dataset.theta, train_dataset.phi)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    history = []
    
    print("\nStarting training (100 epochs)...")
    start_time = time.time()
    
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            coeffs = batch['coeffs'].to(device)
            coords = batch['coords'].to(device)
            u_true = batch['u_true'].to(device)
            
            coords_batch = coords.unsqueeze(0).expand(coeffs.shape[0], -1, -1)
            
            optimizer.zero_grad()
            u_pred = model(coeffs, coords_batch)
            loss = criterion(u_pred, u_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
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
        
        history.append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'deeponet_noise{int(noise_level*100):02d}.pth')
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best Validation Loss: {best_loss:.6f}")
    
    return {
        'noise_level': noise_level,
        'best_loss': best_loss,
        'history': history,
        'time': total_time
    }

# ==============================================================================
# Main Experiment
# ==============================================================================

if __name__ == "__main__":
    print("="*60)
    print("EXPERIMENT 9.2: NOISE ROBUSTNESS STUDY")
    print("="*60)
    
    # Baseline (clean data)
    print("\n\nBASELINE: Clean Data")
    baseline_result = train_on_noisy_data(
        noise_level=0.0,
        train_file='train_poisson_sphere.npz',
        test_file='test_poisson_sphere.npz'
    )
    
    # Noisy data
    noise_levels = [0.01, 0.05, 0.10]
    results = [baseline_result]
    
    for noise_level in noise_levels:
        result = train_on_noisy_data(
            noise_level=noise_level,
            train_file=f'train_poisson_sphere_noise{int(noise_level*100):02d}.npz',
            test_file='test_poisson_sphere.npz'
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT 9.2 RESULTS")
    print("="*60)
    print(f"{'Noise Level':<15} {'Test Loss':<15} {'Degradation':<15}")
    print("-"*60)
    
    baseline_loss = results[0]['best_loss']
    for r in results:
        noise_pct = int(r['noise_level'] * 100)
        degradation = ((r['best_loss'] - baseline_loss) / baseline_loss) * 100
        print(f"{noise_pct:2d}%{'':<12} {r['best_loss']:.6f}{'':<8} {degradation:+.1f}%")
    
    # Save results
    with open('experiment_9_2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to experiment_9_2_results.json")
