"""
Experiment 9.3: Deformed Geometry (Ellipsoid)
Train Geometric DeepONet on ellipsoid instead of sphere.

This tests whether DeepONet can adapt to deformed geometry
where SFNO's spherical harmonic basis breaks down.
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
# Dataset for Ellipsoid
# ==============================================================================

class EllipsoidPoissonDataset(Dataset):
    def __init__(self, npz_path, L_max=5, normalize=True, mean=None, std=None):
        data = np.load(npz_path)
        self.sources = data['sources']
        self.solutions = data['solutions']
        self.theta = data['theta']
        self.phi = data['phi']
        self.a = float(data['a'])
        self.b = float(data['b'])
        self.c = float(data['c'])
        self.n_samples = len(self.sources)
        
        print(f"Loading {npz_path}...")
        print(f"  Ellipsoid: a={self.a}, b={self.b}, c={self.c}")
        print(f"  Samples: {self.n_samples}")
        
        # Compute spherical harmonic coefficients
        print("Computing spherical harmonic coefficients...")
        self.source_coeffs = []
        for i in range(self.n_samples):
            coeffs = self.compute_sh_coeffs(self.sources[i], L_max)
            self.source_coeffs.append(coeffs)
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
        
        # Cartesian coordinates on ellipsoid
        Theta, Phi = np.meshgrid(self.theta, self.phi, indexing='ij')
        X = self.a * np.sin(Theta) * np.cos(Phi)
        Y = self.b * np.sin(Theta) * np.sin(Phi)
        Z = self.c * np.cos(Theta)
        self.coords_cartesian = torch.FloatTensor(np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1))
        
    def compute_sh_coeffs(self, u_values, L_max):
        """Compute spherical harmonic coefficients."""
        Theta, Phi = np.meshgrid(self.theta, self.phi, indexing='ij')
        coeffs = []
        for l in range(L_max + 1):
            for m in range(-l, l + 1):
                Y_lm = sph_harm(m, l, Phi, Theta)
                integrand = u_values * np.conj(Y_lm) * np.sin(Theta)
                c_lm = np.trapz(np.trapz(integrand, self.phi, axis=1), self.theta, axis=0)
                coeffs.append(np.abs(c_lm))
        return np.array(coeffs, dtype=np.float32)
        
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
# Model with Ellipsoid-Aware Geodesics
# ==============================================================================

class EllipsoidGeometricDeepONet(nn.Module):
    def __init__(self, L_max=5, n_refs=10, p=64, a=1.0, b=0.8, c=1.2):
        super().__init__()
        self.a, self.b, self.c = a, b, c
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
        
    def initialize_references(self, theta, phi):
        """Initialize reference points on ellipsoid."""
        n = self.n_refs
        indices = np.arange(n) + 0.5
        phi_refs = np.arccos(1 - 2 * indices / n)
        theta_refs = np.pi * (1 + 5**0.5) * indices
        x = self.a * np.sin(phi_refs) * np.cos(theta_refs)
        y = self.b * np.sin(phi_refs) * np.sin(theta_refs)
        z = self.c * np.cos(phi_refs)
        self.ref_points = torch.FloatTensor(np.stack([x, y, z], axis=-1))
        
    def trunk(self, coords_cartesian):
        batch_size, n_points, _ = coords_cartesian.shape
        coords_flat = coords_cartesian.reshape(-1, 3)
        
        # Euclidean distances (approximation of geodesic on ellipsoid)
        distances = []
        for i in range(self.n_refs):
            ref = self.ref_points[i:i+1].expand(coords_flat.shape[0], -1)
            d = torch.norm(coords_flat - ref, dim=-1)
            distances.append(d.unsqueeze(-1))
        distances = torch.cat(distances, dim=-1)
        
        # Gaussian curvature of ellipsoid (approximate)
        curvature = torch.ones(coords_flat.shape[0], 1, device=coords_flat.device) * 0.5
        
        # Fourier features
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
    print("EXPERIMENT 9.3: DEFORMED GEOMETRY (ELLIPSOID)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading training data...")
    train_dataset = EllipsoidPoissonDataset('train_ellipsoid_poisson.npz', L_max=5, normalize=True)
    
    print("Loading test data...")
    test_dataset = EllipsoidPoissonDataset(
        'test_ellipsoid_poisson.npz',
        L_max=5,
        normalize=True,
        mean=train_dataset.coeff_mean,
        std=train_dataset.coeff_std
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    
    # Model
    model = EllipsoidGeometricDeepONet(
        L_max=5, n_refs=10, p=64,
        a=train_dataset.a, b=train_dataset.b, c=train_dataset.c
    ).to(device)
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
            torch.save(model.state_dict(), 'deeponet_ellipsoid.pth')
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best Validation Loss: {best_loss:.6f}")
    
    # Compare to baseline
    baseline = 0.0596  # From sphere
    degradation = ((best_loss - baseline) / baseline) * 100
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Baseline (sphere):  {baseline:.6f}")
    print(f"Ellipsoid:          {best_loss:.6f}")
    print(f"Degradation:        {degradation:+.1f}%")
    
    # Save results
    results = {
        'best_loss': best_loss,
        'baseline': baseline,
        'degradation_pct': degradation,
        'ellipsoid_params': {'a': train_dataset.a, 'b': train_dataset.b, 'c': train_dataset.c},
        'history': history,
        'time': total_time
    }
    
    with open('experiment_9_3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to experiment_9_3_results.json")
