"""
Experiment 1.5d: Physics-Informed Loss Analysis
Investigate why PDE residual term had minimal impact
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

from geometric_deeponet_sphere import GeometricDeepONet, GeometricPoissonDataset, custom_collate_geometric

print("="*70)
print("EXPERIMENT 1.5d: PHYSICS-INFORMED LOSS ANALYSIS")
print("="*70)
print("\nInvestigating why PDE loss didn't help...")
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load datasets
print("Loading datasets...")
train_dataset = GeometricPoissonDataset('train_poisson_sphere.npz', L_max=5)
test_dataset = GeometricPoissonDataset('test_poisson_sphere.npz', L_max=5)

# Apply normalization (from previous fix)
all_coeffs = np.array([train_dataset[i]['coeffs'].numpy() for i in range(len(train_dataset))])
coeff_mean = np.mean(all_coeffs, axis=0)
coeff_std = np.std(all_coeffs, axis=0) + 1e-8

for i in range(len(train_dataset)):
    train_dataset.source_coeffs[i] = (train_dataset.source_coeffs[i] - coeff_mean) / coeff_std
for i in range(len(test_dataset)):
    test_dataset.source_coeffs[i] = (test_dataset.source_coeffs[i] - coeff_mean) / coeff_std

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          collate_fn=custom_collate_geometric)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                         collate_fn=custom_collate_geometric)


def train_with_pde_loss(lambda_pde=0.1, epochs=100, verbose=True):
    """Train with physics-informed loss and track components."""
    
    model = GeometricDeepONet(L_max=5, n_refs=10, p=64, R=1.0)
    model.trunk.initialize_references(train_dataset.theta, train_dataset.phi)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Track loss components
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
        
        for batch in train_loader:
            coeffs = batch['coeffs'].to(device)
            coords = batch['coords'].to(device)
            u_true = batch['u_true'].to(device)
            
            batch_size = coeffs.shape[0]
            coords_batch = coords.unsqueeze(0).expand(batch_size, -1, -1)
            
            optimizer.zero_grad()
            u_pred = model(coeffs, coords_batch)
            
            # Data loss
            loss_data = criterion(u_pred, u_true)
            
            # PDE loss (if lambda > 0)
            if lambda_pde > 0:
                loss_pde = model.compute_physics_loss(coeffs, coords_batch, u_pred)
                total_loss = loss_data + lambda_pde * loss_pde
            else:
                loss_pde = torch.tensor(0.0)
                total_loss = loss_data
            
            total_loss.backward()
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
        
        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1:3d}/{epochs} | Data: {epoch_data_loss:.6f} | "
                  f"PDE: {epoch_pde_loss:.6f} | Total: {epoch_total_loss:.6f} | "
                  f"Test: {test_loss:.6f}")
    
    return model, history, best_test_loss


# Experiment 1: Track loss components with default λ=0.1
print("\n" + "="*70)
print("PART 1: LOSS COMPONENT ANALYSIS (λ=0.1)")
print("="*70)
print("\nTraining with PDE loss (λ=0.1)...")
print("\n  Epoch | L_data   | L_PDE    | L_total  | L_test")
print("  " + "-"*60)

model_baseline, history_baseline, best_baseline = train_with_pde_loss(lambda_pde=0.1, epochs=100)

# Print summary table
epochs_to_show = [1, 20, 40, 60, 80, 100]
for ep in epochs_to_show:
    idx = ep - 1
    if idx < len(history_baseline['train_data_loss']):
        print(f"  {ep:5d} | {history_baseline['train_data_loss'][idx]:.6f} | "
              f"{history_baseline['train_pde_loss'][idx]:.6f} | "
              f"{history_baseline['train_total_loss'][idx]:.6f} | "
              f"{history_baseline['test_loss'][idx]:.6f}")

# Experiment 2: Vary λ
print("\n" + "="*70)
print("PART 2: PDE WEIGHT SENSITIVITY ANALYSIS")
print("="*70)

lambda_values = [0.0, 0.01, 0.1, 1.0, 10.0]
results = {}

for lam in lambda_values:
    print(f"\nTraining with λ={lam}...")
    _, history, best_loss = train_with_pde_loss(lambda_pde=lam, epochs=50, verbose=False)
    results[lam] = {
        'final_test_loss': history['test_loss'][-1],
        'best_test_loss': best_loss,
        'history': history
    }
    print(f"  Final test loss: {history['test_loss'][-1]:.6f}")
    print(f"  Best test loss: {best_loss:.6f}")

# Experiment 3: Check PDE residual accuracy
print("\n" + "="*70)
print("PART 3: PDE RESIDUAL ACCURACY CHECK")
print("="*70)

print("\nComputing PDE residuals on test set...")

model_baseline.eval()
residuals = []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        if i >= 5:  # Check first 5 batches
            break
        
        coeffs = batch['coeffs'].to(device)
        coords = batch['coords'].to(device)
        
        batch_size = coeffs.shape[0]
        coords_batch = coords.unsqueeze(0).expand(batch_size, -1, -1)
        
        u_pred = model_baseline(coeffs, coords_batch)
        
        # Compute PDE residual
        pde_residual = model_baseline.compute_physics_loss(coeffs, coords_batch, u_pred)
        residuals.append(pde_residual.item())

print(f"\nPDE Residual Statistics:")
print(f"  Mean: {np.mean(residuals):.6f}")
print(f"  Std: {np.std(residuals):.6f}")
print(f"  Min: {np.min(residuals):.6f}")
print(f"  Max: {np.max(residuals):.6f}")

# Visualizations
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Plot 1: Loss components over time
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
epochs = range(1, len(history_baseline['train_data_loss']) + 1)
ax1.semilogy(epochs, history_baseline['train_data_loss'], label='Data Loss', linewidth=2)
ax1.semilogy(epochs, history_baseline['train_pde_loss'], label='PDE Loss', linewidth=2)
ax1.semilogy(epochs, history_baseline['train_total_loss'], label='Total Loss', linewidth=2)
ax1.semilogy(epochs, history_baseline['test_loss'], label='Test Loss', linewidth=2, linestyle='--')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Loss Components (λ=0.1)', fontsize=13, weight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Final error vs λ
ax2 = axes[1]
lambdas = list(results.keys())
final_losses = [results[lam]['final_test_loss'] for lam in lambdas]
best_losses = [results[lam]['best_test_loss'] for lam in lambdas]

ax2.semilogx(lambdas, final_losses, 'o-', linewidth=2, markersize=8, label='Final Test Loss')
ax2.semilogx(lambdas, best_losses, 's--', linewidth=2, markersize=8, label='Best Test Loss')
ax2.set_xlabel('PDE Weight (λ)', fontsize=12)
ax2.set_ylabel('Test Loss', fontsize=12)
ax2.set_title('PDE Weight Sensitivity', fontsize=13, weight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Add optimal point
optimal_lambda = min(results.keys(), key=lambda x: results[x]['best_test_loss'])
optimal_loss = results[optimal_lambda]['best_test_loss']
ax2.axvline(optimal_lambda, color='red', linestyle=':', alpha=0.7, label=f'Optimal λ={optimal_lambda}')
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('pde_loss_analysis.png', dpi=150, bbox_inches='tight')
print("\n  Saved: pde_loss_analysis.png")
plt.close()

# Analysis and conclusions
print("\n" + "="*70)
print("ANALYSIS & CONCLUSIONS")
print("="*70)

# Check if PDE loss helped
no_pde_loss = results[0.0]['best_test_loss']
with_pde_loss = results[optimal_lambda]['best_test_loss']

print(f"\nPerformance Comparison:")
print(f"  No PDE loss (λ=0):      {no_pde_loss:.6f}")
print(f"  Optimal PDE (λ={optimal_lambda}): {with_pde_loss:.6f}")

if with_pde_loss < no_pde_loss:
    improvement = (no_pde_loss - with_pde_loss) / no_pde_loss * 100
    print(f"  → PDE loss helps: {improvement:.1f}% improvement")
else:
    degradation = (with_pde_loss - no_pde_loss) / no_pde_loss * 100
    print(f"  → PDE loss hurts: {degradation:.1f}% degradation")

# Check PDE residual magnitude
print(f"\nPDE Residual Analysis:")
print(f"  Average residual: {np.mean(residuals):.6f}")
print(f"  Compared to data loss: {np.mean(residuals) / history_baseline['train_data_loss'][-1]:.2f}×")

if np.mean(residuals) > 10 * history_baseline['train_data_loss'][-1]:
    print("  ⚠ PDE residual is much larger than data loss!")
    print("    → May need stronger weighting or better balancing")

# Findings
findings = {
    'loss_components': {
        'final_data_loss': float(history_baseline['train_data_loss'][-1]),
        'final_pde_loss': float(history_baseline['train_pde_loss'][-1]),
        'final_total_loss': float(history_baseline['train_total_loss'][-1])
    },
    'lambda_sensitivity': {
        str(lam): {
            'final_loss': float(results[lam]['final_test_loss']),
            'best_loss': float(results[lam]['best_test_loss'])
        } for lam in lambda_values
    },
    'pde_residual_stats': {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals))
    },
    'conclusions': {
        'pde_helps': bool(with_pde_loss < no_pde_loss),
        'optimal_lambda': float(optimal_lambda),
        'improvement_percent': float((no_pde_loss - with_pde_loss) / no_pde_loss * 100) if with_pde_loss < no_pde_loss else float((with_pde_loss - no_pde_loss) / no_pde_loss * -100)
    }
}

with open('pde_loss_analysis_results.json', 'w') as f:
    json.dump(findings, f, indent=4)

print("\n" + "="*70)
print("EXPERIMENT 1.5d COMPLETE")
print("="*70)
print("\nOutputs:")
print("  - pde_loss_analysis.png")
print("  - pde_loss_analysis_results.json")
print("\nKey Findings:")
print(f"  1. Optimal λ: {optimal_lambda}")
print(f"  2. PDE loss {'helps' if findings['conclusions']['pde_helps'] else 'hurts'}")
print(f"  3. Change: {findings['conclusions']['improvement_percent']:.1f}%")
print(f"  4. PDE residual: {np.mean(residuals):.6f}")
