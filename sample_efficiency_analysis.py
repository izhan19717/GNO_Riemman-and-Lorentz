"""
Sample Efficiency Analysis and Visualization
Complete analysis of sample efficiency study results
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd
import json


def power_law(x, a, alpha):
    """Power law: y = a * x^(-alpha)"""
    return a * x**(-alpha)


def analyze_and_plot_results(results_file='sample_efficiency_results.json'):
    """Complete analysis and visualization of sample efficiency results."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract data
    sphere_data = results['sphere']
    minkowski_data = results['minkowski']
    
    n_train = np.array(sphere_data['n_train'])
    
    # Compute means and stds
    sphere_geo_mean = np.array([np.mean(errors) for errors in sphere_data['geometric_errors']])
    sphere_geo_std = np.array([np.std(errors) for errors in sphere_data['geometric_errors']])
    sphere_base_mean = np.array([np.mean(errors) for errors in sphere_data['baseline_errors']])
    sphere_base_std = np.array([np.std(errors) for errors in sphere_data['baseline_errors']])
    
    mink_geo_mean = np.array([np.mean(errors) for errors in minkowski_data['geometric_errors']])
    mink_geo_std = np.array([np.std(errors) for errors in minkowski_data['geometric_errors']])
    mink_base_mean = np.array([np.mean(errors) for errors in minkowski_data['baseline_errors']])
    mink_base_std = np.array([np.std(errors) for errors in minkowski_data['baseline_errors']])
    
    # Fit power laws
    print("Fitting power laws...")
    
    # Sphere
    sphere_geo_params, _ = curve_fit(power_law, n_train, sphere_geo_mean, p0=[1, 0.5])
    sphere_base_params, _ = curve_fit(power_law, n_train, sphere_base_mean, p0=[1, 0.5])
    
    # Minkowski
    mink_geo_params, _ = curve_fit(power_law, n_train, mink_geo_mean, p0=[1, 0.5])
    mink_base_params, _ = curve_fit(power_law, n_train, mink_base_mean, p0=[1, 0.5])
    
    print(f"\nPower Law Exponents:")
    print(f"  Sphere - Geometric: α = {sphere_geo_params[1]:.4f}")
    print(f"  Sphere - Baseline: α = {sphere_base_params[1]:.4f}")
    print(f"  Minkowski - Geometric: α = {mink_geo_params[1]:.4f}")
    print(f"  Minkowski - Baseline: α = {mink_base_params[1]:.4f}")
    
    # Statistical tests
    print("\nComputing statistical tests...")
    
    sphere_pvalues = []
    mink_pvalues = []
    
    for i in range(len(n_train)):
        # Sphere
        geo_errors = sphere_data['geometric_errors'][i]
        base_errors = sphere_data['baseline_errors'][i]
        t_stat, p_val = stats.ttest_rel(geo_errors, base_errors, alternative='greater')
        sphere_pvalues.append(p_val)
        
        # Minkowski
        geo_errors = minkowski_data['geometric_errors'][i]
        base_errors = minkowski_data['baseline_errors'][i]
        t_stat, p_val = stats.ttest_rel(geo_errors, base_errors, alternative='greater')
        mink_pvalues.append(p_val)
    
    # Create plots
    print("\nGenerating plots...")
    
    # Sphere plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Plot data with error bars
    ax.errorbar(n_train, sphere_base_mean, yerr=sphere_base_std, 
                fmt='o-', linewidth=2, markersize=8, capsize=5,
                label='Baseline DeepONet', color='tab:blue')
    ax.errorbar(n_train, sphere_geo_mean, yerr=sphere_geo_std,
                fmt='s-', linewidth=2, markersize=8, capsize=5,
                label='Geometric DeepONet', color='tab:orange')
    
    # Plot power law fits
    n_fit = np.logspace(np.log10(n_train[0]), np.log10(n_train[-1]), 100)
    ax.plot(n_fit, power_law(n_fit, *sphere_base_params), '--',
            color='tab:blue', alpha=0.5, linewidth=1.5,
            label=f'Baseline fit: N^(-{sphere_base_params[1]:.3f})')
    ax.plot(n_fit, power_law(n_fit, *sphere_geo_params), '--',
            color='tab:orange', alpha=0.5, linewidth=1.5,
            label=f'Geometric fit: N^(-{sphere_geo_params[1]:.3f})')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Training Set Size (N)', fontsize=13)
    ax.set_ylabel('Test L² Error', fontsize=13)
    ax.set_title('Sample Efficiency: Sphere (Poisson Equation)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('sample_efficiency_sphere.png', dpi=150, bbox_inches='tight')
    print("  Saved sample_efficiency_sphere.png")
    plt.close()
    
    # Minkowski plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    ax.errorbar(n_train, mink_base_mean, yerr=mink_base_std,
                fmt='o-', linewidth=2, markersize=8, capsize=5,
                label='Baseline DeepONet', color='tab:blue')
    ax.errorbar(n_train, mink_geo_mean, yerr=mink_geo_std,
                fmt='s-', linewidth=2, markersize=8, capsize=5,
                label='Causal DeepONet', color='tab:orange')
    
    ax.plot(n_fit, power_law(n_fit, *mink_base_params), '--',
            color='tab:blue', alpha=0.5, linewidth=1.5,
            label=f'Baseline fit: N^(-{mink_base_params[1]:.3f})')
    ax.plot(n_fit, power_law(n_fit, *mink_geo_params), '--',
            color='tab:orange', alpha=0.5, linewidth=1.5,
            label=f'Causal fit: N^(-{mink_geo_params[1]:.3f})')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Training Set Size (N)', fontsize=13)
    ax.set_ylabel('Test L² Error', fontsize=13)
    ax.set_title('Sample Efficiency: Minkowski (Wave Equation)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('sample_efficiency_minkowski.png', dpi=150, bbox_inches='tight')
    print("  Saved sample_efficiency_minkowski.png")
    plt.close()
    
    # Create summary table
    print("\nCreating summary table...")
    
    table_data = []
    
    for i, n in enumerate(n_train):
        # Sphere
        improvement_sphere = (sphere_base_mean[i] - sphere_geo_mean[i]) / sphere_base_mean[i] * 100
        table_data.append({
            'Geometry': 'Sphere',
            'N_train': n,
            'Geometric_Error': f'{sphere_geo_mean[i]:.6f} ± {sphere_geo_std[i]:.6f}',
            'Baseline_Error': f'{sphere_base_mean[i]:.6f} ± {sphere_base_std[i]:.6f}',
            'Improvement_%': f'{improvement_sphere:.2f}',
            'p_value': f'{sphere_pvalues[i]:.4f}'
        })
        
        # Minkowski
        improvement_mink = (mink_base_mean[i] - mink_geo_mean[i]) / mink_base_mean[i] * 100
        table_data.append({
            'Geometry': 'Minkowski',
            'N_train': n,
            'Geometric_Error': f'{mink_geo_mean[i]:.6f} ± {mink_geo_std[i]:.6f}',
            'Baseline_Error': f'{mink_base_mean[i]:.6f} ± {mink_base_std[i]:.6f}',
            'Improvement_%': f'{improvement_mink:.2f}',
            'p_value': f'{mink_pvalues[i]:.4f}'
        })
    
    df = pd.DataFrame(table_data)
    df.to_csv('efficiency_summary_table.csv', index=False)
    print("  Saved efficiency_summary_table.csv")
    
    # Save statistical tests
    statistical_results = {
        'sphere': {
            'power_law_exponent_geometric': float(sphere_geo_params[1]),
            'power_law_exponent_baseline': float(sphere_base_params[1]),
            'p_values': [float(p) for p in sphere_pvalues],
            'mean_errors_geometric': sphere_geo_mean.tolist(),
            'mean_errors_baseline': sphere_base_mean.tolist()
        },
        'minkowski': {
            'power_law_exponent_geometric': float(mink_geo_params[1]),
            'power_law_exponent_baseline': float(mink_base_params[1]),
            'p_values': [float(p) for p in mink_pvalues],
            'mean_errors_geometric': mink_geo_mean.tolist(),
            'mean_errors_baseline': mink_base_mean.tolist()
        },
        'n_train_sizes': n_train.tolist()
    }
    
    with open('statistical_tests.json', 'w') as f:
        json.dump(statistical_results, f, indent=4)
    print("  Saved statistical_tests.json")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nKey Findings:")
    print(f"\n1. Sphere (Poisson Equation):")
    print(f"   - Baseline converges as N^(-{sphere_base_params[1]:.3f})")
    print(f"   - Geometric converges as N^(-{sphere_geo_params[1]:.3f})")
    print(f"   - Baseline performs BETTER (lower error)")
    
    print(f"\n2. Minkowski (Wave Equation):")
    print(f"   - Baseline converges as N^(-{mink_base_params[1]:.3f})")
    print(f"   - Causal converges as N^(-{mink_geo_params[1]:.3f})")
    print(f"   - Similar performance at large N")
    
    print("\n3. Statistical Significance:")
    print(f"   - Sphere: p-values range from {min(sphere_pvalues):.4f} to {max(sphere_pvalues):.4f}")
    print(f"   - Minkowski: p-values range from {min(mink_pvalues):.4f} to {max(mink_pvalues):.4f}")
    
    return statistical_results


if __name__ == "__main__":
    print("="*70)
    print("SAMPLE EFFICIENCY ANALYSIS")
    print("="*70)
    print()
    
    results = analyze_and_plot_results()
    
    print()
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - sample_efficiency_sphere.png")
    print("  - sample_efficiency_minkowski.png")
    print("  - efficiency_summary_table.csv")
    print("  - statistical_tests.json")
