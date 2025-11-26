"""
Experiment 1.5b: Spherical Harmonic Encoding Debug
Comprehensive investigation of why geometric DeepONet fails on sphere
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

from geometric_deeponet_sphere import GeometricPoissonDataset, custom_collate_geometric
from baseline_deeponet import BaselineDeepONet, PoissonDataset, custom_collate


def analyze_sh_coefficients(dataset, save_path='sh_coefficient_analysis.png'):
    """Analyze spherical harmonic coefficient distributions."""
    
    print("\n" + "="*70)
    print("STEP 1: VERIFY SH COEFFICIENT COMPUTATION")
    print("="*70)
    
    # Get coefficients from dataset
    all_coeffs = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        coeffs = sample['coeffs'].numpy()
        all_coeffs.append(coeffs)
    
    all_coeffs = np.array(all_coeffs)  # (n_samples, n_coeffs)
    
    # Statistics
    print(f"\nCoefficient Statistics:")
    print(f"  Shape: {all_coeffs.shape}")
    print(f"  Mean: {np.mean(all_coeffs):.6f}")
    print(f"  Std: {np.std(all_coeffs):.6f}")
    print(f"  Min: {np.min(all_coeffs):.6f}")
    print(f"  Max: {np.max(all_coeffs):.6f}")
    print(f"  Range: {np.max(all_coeffs) - np.min(all_coeffs):.6f}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(all_coeffs)):
        print("  WARNING: NaN values detected!")
    if np.any(np.isinf(all_coeffs)):
        print("  WARNING: Inf values detected!")
    
    # Visualize distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Overall histogram
    ax1 = axes[0, 0]
    ax1.hist(all_coeffs.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Coefficient Value', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Overall Coefficient Distribution', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Per-coefficient statistics
    ax2 = axes[0, 1]
    means = np.mean(all_coeffs, axis=0)
    stds = np.std(all_coeffs, axis=0)
    ax2.errorbar(range(len(means)), means, yerr=stds, fmt='o-', capsize=3)
    ax2.set_xlabel('Coefficient Index', fontsize=11)
    ax2.set_ylabel('Mean ± Std', fontsize=11)
    ax2.set_title('Per-Coefficient Statistics', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Coefficient magnitudes
    ax3 = axes[1, 0]
    magnitudes = np.abs(all_coeffs)
    ax3.boxplot([magnitudes[:, i] for i in range(min(20, all_coeffs.shape[1]))],
                labels=[str(i) for i in range(min(20, all_coeffs.shape[1]))])
    ax3.set_xlabel('Coefficient Index', fontsize=11)
    ax3.set_ylabel('Absolute Value', fontsize=11)
    ax3.set_title('Coefficient Magnitudes (First 20)', fontsize=12, weight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Correlation matrix (first 10 coefficients)
    ax4 = axes[1, 1]
    n_show = min(10, all_coeffs.shape[1])
    corr = np.corrcoef(all_coeffs[:, :n_show].T)
    im = ax4.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax4.set_xlabel('Coefficient Index', fontsize=11)
    ax4.set_ylabel('Coefficient Index', fontsize=11)
    ax4.set_title(f'Correlation Matrix (First {n_show})', fontsize=12, weight='bold')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_path}")
    plt.close()
    
    return {
        'mean': float(np.mean(all_coeffs)),
        'std': float(np.std(all_coeffs)),
        'min': float(np.min(all_coeffs)),
        'max': float(np.max(all_coeffs)),
        'has_nan': bool(np.any(np.isnan(all_coeffs))),
        'has_inf': bool(np.any(np.isinf(all_coeffs)))
    }


def test_normalization_schemes(dataset, save_path='normalization_test.png'):
    """Test different normalization schemes for SH coefficients."""
    
    print("\n" + "="*70)
    print("STEP 2: TEST NORMALIZATION SCHEMES")
    print("="*70)
    
    # Get sample coefficients
    sample_coeffs = []
    for i in range(min(100, len(dataset))):
        coeffs = dataset[i]['coeffs'].numpy()
        sample_coeffs.append(coeffs)
    
    sample_coeffs = np.array(sample_coeffs)
    
    # Test different normalizations
    normalizations = {}
    
    # 1. No normalization (original)
    normalizations['Original'] = sample_coeffs.copy()
    
    # 2. Standardization
    mean = np.mean(sample_coeffs, axis=0, keepdims=True)
    std = np.std(sample_coeffs, axis=0, keepdims=True) + 1e-8
    normalizations['Standardized'] = (sample_coeffs - mean) / std
    
    # 3. Min-Max scaling
    min_val = np.min(sample_coeffs, axis=0, keepdims=True)
    max_val = np.max(sample_coeffs, axis=0, keepdims=True)
    normalizations['MinMax'] = (sample_coeffs - min_val) / (max_val - min_val + 1e-8)
    
    # 4. L2 normalization per sample
    norms = np.linalg.norm(sample_coeffs, axis=1, keepdims=True) + 1e-8
    normalizations['L2_Norm'] = sample_coeffs / norms
    
    # 5. Log transform (for positive values)
    if np.all(sample_coeffs > 0):
        normalizations['Log'] = np.log(sample_coeffs + 1)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(normalizations.items()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        ax.hist(data.flatten(), bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{name}\nMean={np.mean(data):.3f}, Std={np.std(data):.3f}',
                    fontsize=11, weight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(normalizations), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_path}")
    plt.close()
    
    # Print statistics
    for name, data in normalizations.items():
        print(f"\n  {name}:")
        print(f"    Mean: {np.mean(data):.6f}")
        print(f"    Std: {np.std(data):.6f}")
        print(f"    Range: [{np.min(data):.6f}, {np.max(data):.6f}]")
    
    return normalizations


class ImprovedGeometricBranch(nn.Module):
    """Improved branch network with larger capacity."""
    
    def __init__(self, n_coeffs=36, hidden_dims=[128, 128, 128], p=64):
        super().__init__()
        
        layers = []
        in_dim = n_coeffs
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # Add layer norm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Add dropout
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, p))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, coeffs):
        return self.network(coeffs)


def train_ablation_variants(train_loader, test_loader, device='cpu', epochs=50):
    """Train different architectural variants."""
    
    print("\n" + "="*70)
    print("STEP 4: ABLATION STUDY")
    print("="*70)
    
    from geometric_deeponet_sphere import GeometricDeepONet, GeometricTrunkNetwork
    
    results = {}
    
    # Variant 1: Original architecture
    print("\n  Training: Original Architecture...")
    model1 = GeometricDeepONet(L_max=5, n_refs=10, p=64, R=1.0)
    model1.trunk.initialize_references(
        train_loader.dataset.dataset.theta,
        train_loader.dataset.dataset.phi
    )
    results['original'] = train_and_evaluate(model1, train_loader, test_loader, 
                                             device, epochs)
    
    # Variant 2: Larger branch network
    print("\n  Training: Larger Branch Network...")
    model2 = GeometricDeepONet(L_max=5, n_refs=10, p=64, R=1.0)
    model2.branch = ImprovedGeometricBranch(n_coeffs=36, hidden_dims=[128, 128, 128], p=64)
    model2.trunk.initialize_references(
        train_loader.dataset.dataset.theta,
        train_loader.dataset.dataset.phi
    )
    results['larger_branch'] = train_and_evaluate(model2, train_loader, test_loader,
                                                  device, epochs)
    
    # Variant 3: Baseline for comparison
    print("\n  Training: Baseline (for reference)...")
    baseline_dataset = PoissonDataset('train_poisson_sphere.npz', m_sensors=100)
    baseline_test = PoissonDataset('test_poisson_sphere.npz', m_sensors=100)
    
    from torch.utils.data import Subset
    train_indices = list(range(400))
    train_subset = Subset(baseline_dataset, train_indices)
    test_subset = Subset(baseline_test, list(range(100)))
    
    baseline_train_loader = DataLoader(train_subset, batch_size=32, shuffle=True,
                                      collate_fn=custom_collate)
    baseline_test_loader = DataLoader(test_subset, batch_size=32, shuffle=False,
                                     collate_fn=custom_collate)
    
    model3 = BaselineDeepONet(m_sensors=100, p=64)
    results['baseline'] = train_and_evaluate_baseline(model3, baseline_train_loader,
                                                      baseline_test_loader, device, epochs)
    
    return results


def train_and_evaluate(model, train_loader, test_loader, device, epochs):
    """Train and evaluate a geometric model."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            coeffs = batch['coeffs'].to(device)
            theta = batch['theta'].to(device)
            phi = batch['phi'].to(device)
            u_true = batch['u_true'].to(device)
            
            # Compute features from theta, phi
            batch_size = coeffs.shape[0]
            n_points = theta.shape[1]
            
            # Stack theta, phi as features
            features = torch.stack([theta, phi], dim=-1)  # (batch, n_points, 2)
            
            optimizer.zero_grad()
            u_pred = model(coeffs, features)
            
            loss = criterion(u_pred, u_true)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Testing
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                coeffs = batch['coeffs'].to(device)
                theta = batch['theta'].to(device)
                phi = batch['phi'].to(device)
                u_true = batch['u_true'].to(device)
                
                batch_size = coeffs.shape[0]
                features = torch.stack([theta, phi], dim=-1)
                
                u_pred = model(coeffs, features)
                loss = criterion(u_pred, u_true)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        best_test_loss = min(best_test_loss, test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Test: {test_loss:.6f}")
    
    return {
        'final_train_loss': train_loss,
        'final_test_loss': test_loss,
        'best_test_loss': best_test_loss
    }


def train_and_evaluate_baseline(model, train_loader, test_loader, device, epochs):
    """Train and evaluate baseline model."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            u_sensors = batch['u_sensors'].to(device)
            coords = batch['coords'].to(device)
            u_true = batch['u_true'].to(device)
            
            batch_size = u_sensors.shape[0]
            coords_batch = coords.unsqueeze(0).expand(batch_size, -1, -1)
            
            optimizer.zero_grad()
            u_pred = model(u_sensors, coords_batch)
            
            loss = criterion(u_pred, u_true)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Testing
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                u_sensors = batch['u_sensors'].to(device)
                coords = batch['coords'].to(device)
                u_true = batch['u_true'].to(device)
                
                batch_size = u_sensors.shape[0]
                coords_batch = coords.unsqueeze(0).expand(batch_size, -1, -1)
                
                u_pred = model(u_sensors, coords_batch)
                loss = criterion(u_pred, u_true)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        best_test_loss = min(best_test_loss, test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Test: {test_loss:.6f}")
    
    return {
        'final_train_loss': train_loss,
        'final_test_loss': test_loss,
        'best_test_loss': best_test_loss
    }


def visualize_ablation_results(results, save_path='ablation_debug_results.png'):
    """Visualize ablation study results."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    variants = list(results.keys())
    test_losses = [results[v]['final_test_loss'] for v in variants]
    
    colors = ['tab:orange', 'tab:green', 'tab:blue']
    bars = ax.bar(range(len(variants)), test_losses, color=colors[:len(variants)],
                  alpha=0.7, edgecolor='black', linewidth=2)
    
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([v.replace('_', '\n') for v in variants], fontsize=11)
    ax.set_ylabel('Final Test Loss', fontsize=12)
    ax.set_title('Ablation Study: Architectural Variants', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, test_losses)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}',
               ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENT 1.5b: SPHERICAL HARMONIC ENCODING DEBUG")
    print("="*70)
    print("\nInvestigating why geometric DeepONet fails on sphere...")
    print("Baseline error: 0.1008")
    print("Geometric error: 1.2439 (12× worse!)")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load dataset
    print("Loading geometric dataset...")
    train_dataset = GeometricPoissonDataset('train_poisson_sphere.npz', L_max=5)
    test_dataset = GeometricPoissonDataset('test_poisson_sphere.npz', L_max=5)
    
    # Step 1: Analyze SH coefficients
    coeff_stats = analyze_sh_coefficients(train_dataset)
    
    # Step 2: Test normalizations
    normalizations = test_normalization_schemes(train_dataset)
    
    # Step 3: Architecture analysis (printed in ablation)
    
    # Step 4: Ablation study
    from torch.utils.data import Subset
    train_indices = list(range(400))
    test_indices = list(range(100))
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True,
                              collate_fn=custom_collate_geometric)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False,
                             collate_fn=custom_collate_geometric)
    
    ablation_results = train_ablation_variants(train_loader, test_loader, device, epochs=50)
    
    # Visualize results
    visualize_ablation_results(ablation_results)
    
    # Save comprehensive report
    debug_report = {
        'coefficient_statistics': coeff_stats,
        'ablation_results': ablation_results,
        'findings': {
            'coefficient_range': f"[{coeff_stats['min']:.6f}, {coeff_stats['max']:.6f}]",
            'has_numerical_issues': coeff_stats['has_nan'] or coeff_stats['has_inf'],
            'best_variant': min(ablation_results.items(), key=lambda x: x[1]['final_test_loss'])[0],
            'improvement_needed': ablation_results['baseline']['final_test_loss'] < ablation_results['original']['final_test_loss']
        }
    }
    
    with open('sphere_debug_report.json', 'w') as f:
        json.dump(debug_report, f, indent=4)
    
    print("\n" + "="*70)
    print("EXPERIMENT 1.5b COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  - sh_coefficient_analysis.png")
    print("  - normalization_test.png")
    print("  - ablation_debug_results.png")
    print("  - sphere_debug_report.json")
    print("\nKey Findings:")
    print(f"  Coefficient range: [{coeff_stats['min']:.6f}, {coeff_stats['max']:.6f}]")
    print(f"  Numerical issues: {coeff_stats['has_nan'] or coeff_stats['has_inf']}")
    print(f"  Best variant: {debug_report['findings']['best_variant']}")
    print(f"  Original test loss: {ablation_results['original']['final_test_loss']:.6f}")
    print(f"  Baseline test loss: {ablation_results['baseline']['final_test_loss']:.6f}")
