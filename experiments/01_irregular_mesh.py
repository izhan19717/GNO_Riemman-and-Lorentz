"""
Experiment 9.1: Irregular Mesh Benchmark
Train Geometric DeepONet on random point clouds (no grid structure).

This tests DeepONet's mesh-agnostic nature.
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

# ==============================================================================
# Dataset for Irregular Mesh
# ==============================================================================

class IrregularMeshDataset(Dataset):
    def __init__(self, npz_path, L_max=5, normalize=True, mean=None, std=None):
        data = np.load(npz_path)
        self.coords = data['coords']  # [n_samples, n_points, 3]
        self.sources = data['sources']  # [n_samples, n_points]
        self.solutions = data['solutions']  # [n_samples, n_points]
        self.n_samples = len(self.sources)
        
        print(f"Loading {npz_path}...")
        print(f"  Samples: {self.n_samples}")
        print(f"  Points per sample: {self.coords.shape[1]}")
        
        # Compute spherical harmonic coefficients from source values
        print("Computing spherical harmonic coefficients...")
        self.source_coeffs = []
        
        for i in range(self.n_samples):
            # Convert Cartesian to spherical
            coords_i = self.coords[i]  # [n_points, 3]
            x, y, z = coords_i[:, 0], coords_i[:, 1], coords_i[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(np.sqrt(x**2 + y**2), z)  # [0, pi]
            phi = np.arctan2(y, x)  # [-pi, pi]
            
            # Compute coefficients via numerical integration
            source_i = self.sources[i]
            coeffs = []
            
            for l in range(L_max + 1):
                for m in range(-l, l + 1):
                    Y_lm = sph_harm(m, l, phi, theta)
                    # Weighted average (simple integration)
                    c_lm = np.mean(source_i * np.conj(Y_lm))
                    coeffs.append(np.abs(c_lm))
            
            self.source_coeffs.append(np.array(coeffs, dtype=np.float32))
        
        self.source_coeffs = np.array(self.source_coeffs)
        
        # Normalize
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
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'coeffs': torch.FloatTensor(self.source_coeffs[idx]),
            'coords': torch.FloatTensor(self.coords[idx]),
            'u_true': torch.FloatTensor(self.solutions[idx])
        }

def custom_collate(batch):
    return {
        'coeffs': torch.stack([item['coeffs'] for item in batch]),
        'coords': torch.stack([item['coords'] for item in batch]),
        'u_true': torch.stack([item['u_true'] for item in batch])
    }

# ==============================================================================
# Model (same as before)
# ==============================================================================

class OptimizedGeometricDeepONet(nn.Module):
    def __init__(self, L_max=5, n_refs=10, p=64, R=1.0):
        super().__init__()
        self.R = R
        n_coeffs = (L_max + 1) ** 2
        
        self.branch = nn.Sequential(
            nn.Linear(n_coeffs, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, p)
        )
        
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
        
    def initialize_references(self, sample_coords):
        """Initialize using Fibonacci sphere."""
        n = self.n_refs
        indices = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * indices / n)
        theta = np.pi * (1 + 5**0.5) * indices
        x = self.R * np.sin(phi) * np.cos(theta)
        y = self.R * np.sin(phi) * np.sin(theta)
        z = self.R * np.cos(phi)
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
# Training
# ==============================================================================

if __name__ == "__main__":
    print("="*60)
    print("EXPERIMENT 9.1: IRREGULAR MESH BENCHMARK")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading training data...")
    train_dataset = IrregularMeshDataset('train_irregular_poisson.npz', L_max=5, normalize=True)
    
    print("Loading test data...")
    test_dataset = IrregularMeshDataset(
        'test_irregular_poisson.npz',
        L_max=5,
        normalize=True,
        mean=train_dataset.coeff_mean,
        std=train_dataset.coeff_std
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate)
    
    # Model
    model = OptimizedGeometricDeepONet(L_max=5, n_refs=10, p=64, R=1.0).to(device)
    model.initialize_references(train_dataset.coords[0])
    
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
            
            optimizer.zero_grad()
            u_pred = model(coeffs, coords)
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
                u_pred = model(coeffs, coords)
                val_loss += criterion(u_pred, u_true).item()
        val_loss /= len(test_loader)
        
        history.append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'deeponet_irregular_mesh.pth')
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best Validation Loss: {best_loss:.6f}")
    
    # Compare to baseline
    baseline = 0.0596  # From regular grid
    degradation = ((best_loss - baseline) / baseline) * 100
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Baseline (regular grid): {baseline:.6f}")
    print(f"Irregular mesh:          {best_loss:.6f}")
    print(f"Degradation:             {degradation:+.1f}%")
    
    # Save results
    results = {
        'best_loss': best_loss,
        'baseline': baseline,
        'degradation_pct': degradation,
        'history': history,
        'time': total_time
    }
    
    with open('experiment_9_1_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to experiment_9_1_results.json")
