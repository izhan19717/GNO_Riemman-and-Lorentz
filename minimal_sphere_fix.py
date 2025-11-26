"""
Minimal Fix for Geometric DeepONet
Just add coefficient normalization - the key issue identified
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import json
import time

# Import existing classes
from geometric_deeponet_sphere import GeometricDeepONet, GeometricPoissonDataset, custom_collate_geometric

print("="*70)
print("MINIMAL FIX: ADD COEFFICIENT NORMALIZATION")
print("="*70)
print("\nKey finding from diagnosis:")
print("  Issue: Coefficients not zero-centered (mean=0.160)")
print("  Fix: Apply standardization normalization")
print()

# Load datasets
print("Loading datasets...")
train_dataset = GeometricPoissonDataset('train_poisson_sphere.npz', L_max=5)
test_dataset = GeometricPoissonDataset('test_poisson_sphere.npz', L_max=5)

# Compute normalization statistics from training data
print("\nComputing normalization statistics...")
all_coeffs = []
for i in range(len(train_dataset)):
    coeffs = train_dataset[i]['coeffs'].numpy()
    all_coeffs.append(coeffs)

all_coeffs = np.array(all_coeffs)
coeff_mean = np.mean(all_coeffs, axis=0)
coeff_std = np.std(all_coeffs, axis=0) + 1e-8

print(f"  Mean: {np.mean(coeff_mean):.6f}")
print(f"  Std: {np.mean(coeff_std):.6f}")

# Apply normalization to datasets
print("\nApplying normalization...")
for i in range(len(train_dataset)):
    train_dataset.source_coeffs[i] = (train_dataset.source_coeffs[i] - coeff_mean) / coeff_std

for i in range(len(test_dataset)):
    test_dataset.source_coeffs[i] = (test_dataset.source_coeffs[i] - coeff_mean) / coeff_std

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          collate_fn=custom_collate_geometric)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                         collate_fn=custom_collate_geometric)

# Create model
print("\nInitializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = GeometricDeepONet(L_max=5, n_refs=10, p=64, R=1.0)
model.trunk.initialize_references(train_dataset.theta, train_dataset.phi)
model = model.to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")

# Training
print(f"\nTraining for 100 epochs...")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_losses = []
test_losses = []
best_test_loss = float('inf')

start_time = time.time()

for epoch in range(100):
    # Train
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        coeffs = batch['coeffs'].to(device)
        coords = batch['coords'].to(device)
        u_true = batch['u_true'].to(device)
        
        batch_size = coeffs.shape[0]
        coords_batch = coords.unsqueeze(0).expand(batch_size, -1, -1)
        
        optimizer.zero_grad()
        u_pred = model(coeffs, coords_batch)
        
        loss = criterion(u_pred, u_true)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Test
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
    test_losses.append(test_loss)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'normalized_geometric_model.pth')
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/100 - Train: {train_loss:.6f}, Test: {test_loss:.6f}, Best: {best_test_loss:.6f}")

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.1f}s")
print(f"Best test loss: {best_test_loss:.6f}")

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(train_losses, label='Train', linewidth=2)
plt.semilogy(test_losses, label='Test', linewidth=2)
plt.axhline(best_test_loss, color='red', linestyle='--', label=f'Best ({best_test_loss:.6f})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Normalized Geometric DeepONet Training')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('normalized_training_curves.png', dpi=150)
print("\nSaved: normalized_training_curves.png")
plt.close()

# Compare with baseline
print("\n" + "="*70)
print("COMPARISON WITH BASELINE")
print("="*70)
baseline_error = 0.1008
original_geometric = 1.2439

print(f"\nBaseline:            {baseline_error:.6f}")
print(f"Original Geometric:  {original_geometric:.6f} (12.3× worse)")
print(f"Normalized Geometric: {best_test_loss:.6f}")

if best_test_loss < baseline_error:
    improvement = (baseline_error - best_test_loss) / baseline_error * 100
    print(f"\n✓ SUCCESS! {improvement:.1f}% better than baseline")
elif best_test_loss < original_geometric:
    improvement = (original_geometric - best_test_loss) / original_geometric * 100
    print(f"\n✓ IMPROVED! {improvement:.1f}% better than original geometric")
    ratio = best_test_loss / baseline_error
    print(f"  But still {ratio:.2f}× worse than baseline")
else:
    print(f"\n✗ No improvement")

# Save results
results = {
    'best_test_loss': float(best_test_loss),
    'training_time': float(training_time),
    'baseline_error': baseline_error,
    'original_geometric_error': original_geometric,
    'improvement_applied': 'Coefficient standardization normalization'
}

with open('normalized_geometric_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print("\nOutputs:")
print("  - normalized_geometric_model.pth")
print("  - normalized_training_curves.png")
print("  - normalized_geometric_results.json")
