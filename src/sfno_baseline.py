"""
Experiment 1.6a: SFNO Implementation for Poisson Equation on Sphere
State-of-the-art comparison using Spherical Fourier Neural Operator
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import json
import time

print("="*70)
print("EXPERIMENT 1.6a: SFNO FOR POISSON EQUATION")
print("="*70)
print("\nImplementing Spherical Fourier Neural Operator for comparison")
print()

# Check torch-harmonics installation
try:
    import torch_harmonics as th
    from torch_harmonics import RealSHT, InverseRealSHT
    print("✓ torch-harmonics successfully imported")
    print(f"  Version: {th.__version__ if hasattr(th, '__version__') else 'unknown'}\n")
except ImportError as e:
    print(f"✗ Error importing torch-harmonics: {e}")
    print("  Please install: pip install --no-build-isolation torch-harmonics\n")
    exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")


class SphericalFourierLayer(nn.Module):
    """Single SFNO layer with spectral convolution."""
    
    def __init__(self, nlat, nlon, in_channels, out_channels, modes_lat, modes_lon):
        super().__init__()
        
        self.nlat = nlat
        self.nlon = nlon
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_lat = modes_lat
        self.modes_lon = modes_lon
        
        # Spherical harmonic transforms
        self.sht = RealSHT(nlat, nlon, grid='equiangular')
        self.isht = InverseRealSHT(nlat, nlon, grid='equiangular')
        
        # Learnable weights in spectral domain
        # SHT produces (batch, channels, nlat, nlon//2+1) complex coefficients
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes_lat, modes_lon, 2)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, nlat, nlon) - spatial domain
        Returns:
            (batch, out_channels, nlat, nlon) - spatial domain
        """
        batch_size = x.shape[0]
        
        # Transform to spectral domain
        x_spec = self.sht(x)  # (batch, in_channels, nlat, nlon//2+1) complex
        
        # Spectral convolution
        out_spec = torch.zeros(batch_size, self.out_channels, self.nlat, self.nlon//2+1,
                               dtype=torch.complex64, device=x.device)
        
        # Apply learnable weights in spectral domain
        for i in range(min(self.modes_lat, self.nlat)):
            for j in range(min(self.modes_lon, self.nlon//2+1)):
                # Convert real weights to complex
                weight_complex = torch.view_as_complex(self.weights[:, :, i, j, :].contiguous())
                
                # Spectral multiplication
                out_spec[:, :, i, j] = torch.einsum('bi,io->bo', 
                                                     x_spec[:, :, i, j], 
                                                     weight_complex)
        
        # Transform back to spatial domain
        out = self.isht(out_spec)  # (batch, out_channels, nlat, nlon)
        
        return out


class SFNO_Poisson(nn.Module):
    """
    Spherical Fourier Neural Operator for Poisson equation.
    Maps source function f to solution u on the sphere.
    """
    
    def __init__(self, nlat=50, nlon=100, hidden_channels=64, n_layers=4, modes=32):
        super().__init__()
        
        self.nlat = nlat
        self.nlon = nlon
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        
        # Input projection
        self.input_proj = nn.Conv2d(1, hidden_channels, kernel_size=1)
        
        # SFNO layers
        self.sfno_layers = nn.ModuleList([
            SphericalFourierLayer(nlat, nlon, hidden_channels, hidden_channels,
                                 modes, modes)
            for _ in range(n_layers)
        ])
        
        # Activation
        self.activation = nn.GELU()
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_channels, 1, kernel_size=1)
    
    def forward(self, source):
        """
        Args:
            source: (batch, 1, nlat, nlon) - source function f
        Returns:
            (batch, 1, nlat, nlon) - solution u
        """
        # Input projection
        x = self.input_proj(source)
        
        # SFNO layers with residual connections
        for layer in self.sfno_layers:
            x_spec = layer(x)
            x = self.activation(x_spec + x)  # Residual connection
        
        # Output projection
        u = self.output_proj(x)
        
        return u


class PoissonSphereDataset_SFNO(Dataset):
    """Dataset adapter for SFNO (equiangular grid format)."""
    
    def __init__(self, npz_path, nlat=50, nlon=100):
        data = np.load(npz_path)
        
        self.sources = data['sources']  # (N, H, W)
        self.solutions = data['solutions']  # (N, H, W)
        
        # Reshape to equiangular grid if needed
        N, H, W = self.sources.shape
        
        if H != nlat or W != nlon:
            print(f"  Reshaping data from ({H}, {W}) to ({nlat}, {nlon})...")
            from scipy.interpolate import RegularGridInterpolator
            
            # Original grid
            theta_old = np.linspace(0, np.pi, H)
            phi_old = np.linspace(0, 2*np.pi, W)
            
            # New equiangular grid
            theta_new = np.linspace(0, np.pi, nlat)
            phi_new = np.linspace(0, 2*np.pi, nlon)
            
            sources_new = []
            solutions_new = []
            
            for i in range(N):
                # Interpolate source
                interp_src = RegularGridInterpolator((theta_old, phi_old), self.sources[i])
                theta_grid, phi_grid = np.meshgrid(theta_new, phi_new, indexing='ij')
                src_new = interp_src(np.stack([theta_grid.flatten(), phi_grid.flatten()], axis=1))
                sources_new.append(src_new.reshape(nlat, nlon))
                
                # Interpolate solution
                interp_sol = RegularGridInterpolator((theta_old, phi_old), self.solutions[i])
                sol_new = interp_sol(np.stack([theta_grid.flatten(), phi_grid.flatten()], axis=1))
                solutions_new.append(sol_new.reshape(nlat, nlon))
            
            self.sources = np.array(sources_new)
            self.solutions = np.array(solutions_new)
        
        self.n_samples = len(self.sources)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'source': torch.FloatTensor(self.sources[idx]).unsqueeze(0),  # (1, H, W)
            'solution': torch.FloatTensor(self.solutions[idx]).unsqueeze(0)  # (1, H, W)
        }


def train_sfno(model, train_loader, test_loader, epochs=100, lr=1e-3, device='cpu'):
    """Train SFNO model."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    start_time = time.time()
    
    print("Starting SFNO training...")
    print("-" * 70)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            source = batch['source'].to(device)
            solution = batch['solution'].to(device)
            
            optimizer.zero_grad()
            u_pred = model(source)
            
            loss = criterion(u_pred, solution)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Testing
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                source = batch['source'].to(device)
                solution = batch['solution'].to(device)
                
                u_pred = model(source)
                loss = criterion(u_pred, solution)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'sfno_best_model.pth')
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | "
                  f"Test: {test_loss:.6f} | Best: {best_test_loss:.6f}")
    
    training_time = time.time() - start_time
    
    print("-" * 70)
    print(f"Training complete in {training_time:.1f}s ({training_time/60:.1f}min)")
    print(f"Best test loss: {best_test_loss:.6f}")
    
    return train_losses, test_losses, best_test_loss, training_time


if __name__ == "__main__":
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load datasets
    print("\nLoading Poisson sphere datasets...")
    train_dataset = PoissonSphereDataset_SFNO('train_poisson_sphere.npz', nlat=50, nlon=100)
    test_dataset = PoissonSphereDataset_SFNO('test_poisson_sphere.npz', nlat=50, nlon=100)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print("\n" + "="*70)
    print("INITIALIZING SFNO MODEL")
    print("="*70)
    
    # Create model
    model = SFNO_Poisson(nlat=50, nlon=100, hidden_channels=64, n_layers=4, modes=32)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"Architecture:")
    print(f"  Grid: 50×100 (equiangular)")
    print(f"  Hidden channels: 64")
    print(f"  SFNO layers: 4")
    print(f"  Spectral modes: 32")
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    # Train
    train_losses, test_losses, best_loss, train_time = train_sfno(
        model, train_loader, test_loader,
        epochs=100, lr=1e-3, device=device
    )
    
    print("\n" + "="*70)
    print("COMPARISON WITH GEOMETRIC DEEPONET")
    print("="*70)
    
    # Load comparison results
    baseline_loss = 0.1008
    geometric_original_loss = 1.2439
    geometric_fixed_loss = 0.0600
    
    print(f"\nTest Loss Comparison:")
    print(f"  Baseline DeepONet:           {baseline_loss:.4f}")
    print(f"  Geometric DeepONet (orig):   {geometric_original_loss:.4f}")
    print(f"  Geometric DeepONet (fixed):  {geometric_fixed_loss:.4f}")
    print(f"  SFNO:                        {best_loss:.4f}")
    
    if best_loss < geometric_fixed_loss:
        improvement = (geometric_fixed_loss - best_loss) / geometric_fixed_loss * 100
        print(f"\n  → SFNO is {improvement:.1f}% better than geometric DeepONet")
    elif best_loss > geometric_fixed_loss:
        degradation = (best_loss - geometric_fixed_loss) / geometric_fixed_loss * 100
        print(f"\n  → Geometric DeepONet is {degradation:.1f}% better than SFNO")
    else:
        print(f"\n  → Performance is comparable")
    
    # Save results
    results = {
        'sfno': {
            'best_test_loss': float(best_loss),
            'final_test_loss': float(test_losses[-1]),
            'training_time_seconds': float(train_time),
            'training_time_minutes': float(train_time / 60),
            'n_parameters': int(n_params),
            'n_samples': len(train_dataset)
        },
        'comparison': {
            'baseline': float(baseline_loss),
            'geometric_original': float(geometric_original_loss),
            'geometric_fixed': float(geometric_fixed_loss),
            'sfno': float(best_loss)
        }
    }
    
    with open('sfno_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Visualization
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Training curves
    ax1 = axes[0]
    ax1.semilogy(train_losses, label='Train', linewidth=2)
    ax1.semilogy(test_losses, label='Test', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss (log scale)', fontsize=11)
    ax1.set_title('SFNO Training Curves', fontsize=12, weight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Comparison
    ax2 = axes[1]
    methods = ['Baseline', 'Geometric\n(orig)', 'Geometric\n(fixed)', 'SFNO']
    losses = [baseline_loss, geometric_original_loss, geometric_fixed_loss, best_loss]
    colors = ['gray', 'coral', 'steelblue', 'green']
    
    bars = ax2.bar(methods, losses, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=2)
    ax2.set_ylabel('Test Loss', fontsize=11)
    ax2.set_title('Method Comparison', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig('sfno_comparison.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: sfno_comparison.png")
    plt.close()
    
    print("\n" + "="*70)
    print("EXPERIMENT 1.6a COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  - sfno_best_model.pth")
    print("  - sfno_results.json")
    print("  - sfno_comparison.png")
