"""
Simplified Sphere Debug Experiment
Focus on coefficient analysis and simple fixes
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import matplotlib.pyplot as plt
import json

print("="*70)
print("EXPERIMENT 1.5b: SPHERE FAILURE DIAGNOSIS")
print("="*70)
print("\nProblem: Geometric DeepONet 12× worse than baseline")
print("Baseline: 0.1008 | Geometric: 1.2439\n")

# Load dataset
print("Step 1: Loading dataset...")
data = np.load('train_poisson_sphere.npz')
sources = data['sources']
solutions = data['solutions']

print(f"  Sources shape: {sources.shape}")
print(f"  Solutions shape: {solutions.shape}")

# Compute SH coefficients manually
print("\nStep 2: Analyzing Spherical Harmonic Coefficients...")

from src.spectral.spherical_harmonics import SphericalHarmonics

L_max = 5
nlat, nlon = sources.shape[1], sources.shape[2]

sh = SphericalHarmonics(L_max=L_max)

# Compute coefficients for first 100 samples
n_samples = min(100, len(sources))
all_coeffs = []

for i in range(n_samples):
    coeffs = sh.forward(torch.FloatTensor(sources[i]))
    all_coeffs.append(coeffs.numpy())

all_coeffs = np.array(all_coeffs)

print(f"\n  Coefficient Statistics:")
print(f"    Shape: {all_coeffs.shape}")
print(f"    Mean: {np.mean(all_coeffs):.6f}")
print(f"    Std: {np.std(all_coeffs):.6f}")
print(f"    Min: {np.min(all_coeffs):.6f}")
print(f"    Max: {np.max(all_coeffs):.6f}")
print(f"    Range: {np.max(all_coeffs) - np.min(all_coeffs):.6f}")

# Check for issues
has_nan = np.any(np.isnan(all_coeffs))
has_inf = np.any(np.isinf(all_coeffs))
print(f"    Has NaN: {has_nan}")
print(f"    Has Inf: {has_inf}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Overall distribution
ax1 = axes[0, 0]
ax1.hist(all_coeffs.flatten(), bins=50, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Coefficient Value')
ax1.set_ylabel('Count')
ax1.set_title('SH Coefficient Distribution')
ax1.grid(True, alpha=0.3)

# Per-coefficient means
ax2 = axes[0, 1]
means = np.mean(all_coeffs, axis=0)
stds = np.std(all_coeffs, axis=0)
ax2.errorbar(range(len(means)), means, yerr=stds, fmt='o-', capsize=3)
ax2.set_xlabel('Coefficient Index')
ax2.set_ylabel('Mean ± Std')
ax2.set_title('Per-Coefficient Statistics')
ax2.grid(True, alpha=0.3)

# Coefficient magnitudes
ax3 = axes[1, 0]
magnitudes = np.abs(all_coeffs)
ax3.boxplot([magnitudes[:, i] for i in range(min(20, all_coeffs.shape[1]))],
            labels=[str(i) for i in range(min(20, all_coeffs.shape[1]))])
ax3.set_xlabel('Coefficient Index')
ax3.set_ylabel('Absolute Value')
ax3.set_title('Coefficient Magnitudes (First 20)')
ax3.grid(True, alpha=0.3, axis='y')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# Correlation
ax4 = axes[1, 1]
n_show = min(10, all_coeffs.shape[1])
corr = np.corrcoef(all_coeffs[:, :n_show].T)
im = ax4.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax4.set_title(f'Correlation Matrix (First {n_show})')
plt.colorbar(im, ax=ax4)

plt.tight_layout()
plt.savefig('sh_coefficient_analysis.png', dpi=150)
print("\n  Saved: sh_coefficient_analysis.png")
plt.close()

# Test normalizations
print("\nStep 3: Testing Normalization Schemes...")

normalizations = {}
normalizations['Original'] = all_coeffs.copy()

# Standardization
mean = np.mean(all_coeffs, axis=0, keepdims=True)
std = np.std(all_coeffs, axis=0, keepdims=True) + 1e-8
normalizations['Standardized'] = (all_coeffs - mean) / std

# Min-Max
min_val = np.min(all_coeffs, axis=0, keepdims=True)
max_val = np.max(all_coeffs, axis=0, keepdims=True)
normalizations['MinMax'] = (all_coeffs - min_val) / (max_val - min_val + 1e-8)

# L2 norm
norms = np.linalg.norm(all_coeffs, axis=1, keepdims=True) + 1e-8
normalizations['L2_Norm'] = all_coeffs / norms

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, data) in enumerate(normalizations.items()):
    ax = axes[idx]
    ax.hist(data.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title(f'{name}\nMean={np.mean(data):.3f}, Std={np.std(data):.3f}')
    ax.grid(True, alpha=0.3)
    
    print(f"\n  {name}:")
    print(f"    Mean: {np.mean(data):.6f}")
    print(f"    Std: {np.std(data):.6f}")
    print(f"    Range: [{np.min(data):.6f}, {np.max(data):.6f}]")

plt.tight_layout()
plt.savefig('normalization_test.png', dpi=150)
print("\n  Saved: normalization_test.png")
plt.close()

# Key findings
print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

findings = {
    'coefficient_stats': {
        'mean': float(np.mean(all_coeffs)),
        'std': float(np.std(all_coeffs)),
        'min': float(np.min(all_coeffs)),
        'max': float(np.max(all_coeffs)),
        'has_nan': bool(has_nan),
        'has_inf': bool(has_inf)
    },
    'issues_identified': [],
    'recommendations': []
}

# Check for issues
if np.std(all_coeffs) > 10 * np.abs(np.mean(all_coeffs)):
    findings['issues_identified'].append("High variance in coefficients")
    findings['recommendations'].append("Apply standardization normalization")

if np.max(np.abs(all_coeffs)) > 100:
    findings['issues_identified'].append("Large coefficient magnitudes")
    findings['recommendations'].append("Use gradient clipping or smaller learning rate")

if has_nan or has_inf:
    findings['issues_identified'].append("Numerical instability (NaN/Inf)")
    findings['recommendations'].append("Check SH implementation and add numerical safeguards")

# Check coefficient range
coeff_range = np.max(all_coeffs) - np.min(all_coeffs)
if coeff_range > 1000:
    findings['issues_identified'].append(f"Very large coefficient range ({coeff_range:.2f})")
    findings['recommendations'].append("Apply MinMax or log normalization")

print("\nIssues Identified:")
for issue in findings['issues_identified']:
    print(f"  ✗ {issue}")

print("\nRecommendations:")
for rec in findings['recommendations']:
    print(f"  → {rec}")

# Save report
with open('sphere_debug_report.json', 'w') as f:
    json.dump(findings, f, indent=4)

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
print("\nOutputs:")
print("  - sh_coefficient_analysis.png")
print("  - normalization_test.png")
print("  - sphere_debug_report.json")
print("\nNext Steps:")
print("  1. Implement recommended normalizations")
print("  2. Test with improved architecture")
print("  3. Re-train and compare with baseline")
