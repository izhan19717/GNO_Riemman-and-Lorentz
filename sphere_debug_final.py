"""
Ultra-Simplified Sphere Debug
Just analyze the pre-computed coefficients from the dataset
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

# Load geometric dataset directly
print("Step 1: Loading geometric dataset with pre-computed coefficients...")

from geometric_deeponet_sphere import GeometricPoissonDataset

dataset = GeometricPoissonDataset('train_poisson_sphere.npz', L_max=5)

print(f"  Dataset size: {len(dataset)}")

# Extract coefficients
print("\nStep 2: Extracting and analyzing SH coefficients...")

n_samples = min(100, len(dataset))
all_coeffs = []

for i in range(n_samples):
    sample = dataset[i]
    coeffs = sample['coeffs'].numpy()
    all_coeffs.append(coeffs)

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
ax1.hist(all_coeffs.flatten(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.set_xlabel('Coefficient Value', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('SH Coefficient Distribution', fontsize=12, weight='bold')
ax1.grid(True, alpha=0.3)
ax1.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero')
ax1.legend()

# Per-coefficient means
ax2 = axes[0, 1]
means = np.mean(all_coeffs, axis=0)
stds = np.std(all_coeffs, axis=0)
ax2.errorbar(range(len(means)), means, yerr=stds, fmt='o-', capsize=3, color='darkgreen')
ax2.set_xlabel('Coefficient Index', fontsize=11)
ax2.set_ylabel('Mean ± Std', fontsize=11)
ax2.set_title('Per-Coefficient Statistics', fontsize=12, weight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

# Coefficient magnitudes
ax3 = axes[1, 0]
magnitudes = np.abs(all_coeffs)
bp = ax3.boxplot([magnitudes[:, i] for i in range(min(20, all_coeffs.shape[1]))],
            labels=[str(i) for i in range(min(20, all_coeffs.shape[1]))],
            patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightcoral')
ax3.set_xlabel('Coefficient Index', fontsize=11)
ax3.set_ylabel('Absolute Value', fontsize=11)
ax3.set_title('Coefficient Magnitudes (First 20)', fontsize=12, weight='bold')
ax3.grid(True, alpha=0.3, axis='y')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# Correlation
ax4 = axes[1, 1]
n_show = min(10, all_coeffs.shape[1])
corr = np.corrcoef(all_coeffs[:, :n_show].T)
im = ax4.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax4.set_xlabel('Coefficient Index', fontsize=11)
ax4.set_ylabel('Coefficient Index', fontsize=11)
ax4.set_title(f'Correlation Matrix (First {n_show})', fontsize=12, weight='bold')
cbar = plt.colorbar(im, ax=ax4)
cbar.set_label('Correlation', fontsize=10)

plt.tight_layout()
plt.savefig('sh_coefficient_analysis.png', dpi=150, bbox_inches='tight')
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

# L2 norm per sample
norms = np.linalg.norm(all_coeffs, axis=1, keepdims=True) + 1e-8
normalizations['L2_Norm'] = all_coeffs / norms

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

colors = ['steelblue', 'darkgreen', 'coral', 'purple']

for idx, (name, data) in enumerate(normalizations.items()):
    ax = axes[idx]
    ax.hist(data.flatten(), bins=50, edgecolor='black', alpha=0.7, color=colors[idx])
    ax.set_xlabel('Value', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{name}\nMean={np.mean(data):.3f}, Std={np.std(data):.3f}',
                fontsize=11, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    print(f"\n  {name}:")
    print(f"    Mean: {np.mean(data):.6f}")
    print(f"    Std: {np.std(data):.6f}")
    print(f"    Range: [{np.min(data):.6f}, {np.max(data):.6f}]")

plt.tight_layout()
plt.savefig('normalization_test.png', dpi=150, bbox_inches='tight')
print("\n  Saved: normalization_test.png")
plt.close()

# Key findings
print("\n" + "="*70)
print("KEY FINDINGS & DIAGNOSIS")
print("="*70)

findings = {
    'coefficient_stats': {
        'mean': float(np.mean(all_coeffs)),
        'std': float(np.std(all_coeffs)),
        'min': float(np.min(all_coeffs)),
        'max': float(np.max(all_coeffs)),
        'range': float(np.max(all_coeffs) - np.min(all_coeffs)),
        'has_nan': bool(has_nan),
        'has_inf': bool(has_inf)
    },
    'issues_identified': [],
    'recommendations': []
}

# Diagnostic checks
print("\nDiagnostic Checks:")

# Check 1: Coefficient variance
if np.std(all_coeffs) > 10 * np.abs(np.mean(all_coeffs)):
    issue = f"High variance (std={np.std(all_coeffs):.3f} >> mean={np.mean(all_coeffs):.3f})"
    findings['issues_identified'].append(issue)
    findings['recommendations'].append("Apply standardization normalization before training")
    print(f"  ✗ {issue}")
else:
    print(f"  ✓ Coefficient variance is reasonable")

# Check 2: Large magnitudes
max_abs = np.max(np.abs(all_coeffs))
if max_abs > 100:
    issue = f"Large coefficient magnitudes (max={max_abs:.2f})"
    findings['issues_identified'].append(issue)
    findings['recommendations'].append("Use gradient clipping (max_norm=1.0)")
    print(f"  ✗ {issue}")
else:
    print(f"  ✓ Coefficient magnitudes are reasonable (max={max_abs:.2f})")

# Check 3: Numerical stability
if has_nan or has_inf:
    issue = "Numerical instability detected (NaN/Inf values)"
    findings['issues_identified'].append(issue)
    findings['recommendations'].append("Add numerical safeguards in SH computation")
    print(f"  ✗ {issue}")
else:
    print(f"  ✓ No numerical instability (no NaN/Inf)")

# Check 4: Coefficient range
coeff_range = np.max(all_coeffs) - np.min(all_coeffs)
if coeff_range > 1000:
    issue = f"Very large coefficient range ({coeff_range:.2f})"
    findings['issues_identified'].append(issue)
    findings['recommendations'].append("Apply MinMax scaling to [0, 1] range")
    print(f"  ✗ {issue}")
elif coeff_range > 100:
    issue = f"Large coefficient range ({coeff_range:.2f})"
    findings['issues_identified'].append(issue)
    findings['recommendations'].append("Consider MinMax or standardization")
    print(f"  ⚠ {issue}")
else:
    print(f"  ✓ Coefficient range is reasonable ({coeff_range:.2f})")

# Check 5: Zero-centered
if np.abs(np.mean(all_coeffs)) > 0.1 * np.std(all_coeffs):
    issue = f"Coefficients not zero-centered (mean={np.mean(all_coeffs):.3f})"
    findings['issues_identified'].append(issue)
    findings['recommendations'].append("Center coefficients by subtracting mean")
    print(f"  ⚠ {issue}")
else:
    print(f"  ✓ Coefficients are approximately zero-centered")

print("\n" + "-"*70)
print("SUMMARY")
print("-"*70)

if len(findings['issues_identified']) == 0:
    print("\n✓ No major issues found with SH coefficients!")
    print("  The problem may be in the network architecture or training procedure.")
else:
    print(f"\n✗ Found {len(findings['issues_identified'])} issue(s):")
    for i, issue in enumerate(findings['issues_identified'], 1):
        print(f"  {i}. {issue}")
    
    print(f"\n→ Recommended fixes:")
    for i, rec in enumerate(findings['recommendations'], 1):
        print(f"  {i}. {rec}")

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
print("  1. Implement recommended normalizations in geometric_deeponet_sphere.py")
print("  2. Add layer normalization and dropout to branch network")
print("  3. Increase branch network capacity (36 → 128 → 128 → 64)")
print("  4. Re-train and compare with baseline")
