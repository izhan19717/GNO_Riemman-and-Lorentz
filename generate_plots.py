import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_json(filename):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None
    with open(filename, 'r') as f:
        return json.load(f)

def plot_curvature_sensitivity():
    data = load_json('experiment_curvature_results.json')
    if not data: return

    radii = []
    errors = []
    for r_str, err in data.items():
        radii.append(float(r_str))
        errors.append(err)
    
    # Sort
    sorted_pairs = sorted(zip(radii, errors))
    radii, errors = zip(*sorted_pairs)

    plt.figure(figsize=(8, 5))
    plt.plot(radii, errors, 'o-', linewidth=2, color='#2E86C1')
    plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Training Radius (R=1.0)')
    plt.xlabel('Sphere Radius (R)', fontsize=12)
    plt.ylabel('Relative Test Error', fontsize=12)
    plt.title('Curvature Sensitivity: Generalization across Metric Scales', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot_curvature.png', dpi=300)
    print("Generated plot_curvature.png")

def plot_torus_comparison():
    data = load_json('experiment_torus_results.json')
    if not data: return
    
    # Training curves
    if 'gno' in data and 'std' in data:
        plt.figure(figsize=(8, 5))
        plt.plot(data['gno'], label='Geometric DeepONet (Torus)', linewidth=2, color='#27AE60')
        plt.plot(data['std'], label='Standard DeepONet', linewidth=2, color='#E74C3C', linestyle='--')
        plt.xlabel('Epochs (x20)', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Topological Generalization: Torus Heat Equation', fontsize=14)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plot_torus_training.png', dpi=300)
        print("Generated plot_torus_training.png")

    # Bar chart for final test error
    if 'test_gno' in data and 'test_std' in data:
        plt.figure(figsize=(6, 5))
        models = ['Geometric DeepONet', 'Standard DeepONet']
        errors = [data['test_gno'], data['test_std']]
        colors = ['#27AE60', '#E74C3C']
        
        plt.bar(models, errors, color=colors, alpha=0.8, width=0.5)
        plt.ylabel('Relative Test Error', fontsize=12)
        plt.title('Torus Generalization Performance', fontsize=14)
        for i, v in enumerate(errors):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=11)
        plt.tight_layout()
        plt.savefig('plot_torus_bar.png', dpi=300)
        print("Generated plot_torus_bar.png")

def plot_lorentzian_results():
    data = load_json('theorem_lorentzian_results.json')
    if not data: return
    
    # Bar chart
    plt.figure(figsize=(6, 5))
    models = ['Causal GNO', 'Standard DeepONet']
    # Assuming keys are 'causal' and 'standard'
    errors = [data.get('causal', 0), data.get('standard', 0)]
    colors = ['#8E44AD', '#95A5A6']
    
    plt.bar(models, errors, color=colors, alpha=0.8, width=0.5)
    plt.ylabel('Relative Test Error', fontsize=12)
    plt.title('Lorentzian Wave Equation: Causal Learning', fontsize=14)
    for i, v in enumerate(errors):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig('plot_lorentzian.png', dpi=300)
    print("Generated plot_lorentzian.png")

def plot_unified_validation():
    data = load_json('unified_validation_results.json')
    if not data: return
    
    # 1. Positive & Negative Curvature (Sample Efficiency)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Positive (Sphere)
    if 'positive' in data:
        pos = data['positive']
        ax1.plot(pos['sample_sizes'], pos['gno'], 'o-', label='GNO', color='#27AE60', linewidth=2)
        ax1.plot(pos['sample_sizes'], pos['baseline'], 's--', label='Baseline', color='#E74C3C', linewidth=2)
        ax1.set_title('Positive Curvature (Sphere)', fontsize=12)
        ax1.set_xlabel('Training Samples', fontsize=10)
        ax1.set_ylabel('Relative Test Error', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
    # Negative (Hyperboloid)
    if 'negative' in data:
        neg = data['negative']
        ax2.plot(neg['sample_sizes'], neg['gno'], 'o-', label='GNO', color='#27AE60', linewidth=2)
        ax2.plot(neg['sample_sizes'], neg['baseline'], 's--', label='Baseline', color='#E74C3C', linewidth=2)
        ax2.set_title('Negative Curvature (Hyperboloid)', fontsize=12)
        ax2.set_xlabel('Training Samples', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
    plt.suptitle('Sample Efficiency across Curvatures', fontsize=14)
    plt.tight_layout()
    plt.savefig('plot_unified_curvature.png', dpi=300)
    print("Generated plot_unified_curvature.png")
    
    # 2. Indefinite Curvature (Causality)
    if 'indefinite' in data:
        indef = data['indefinite']
        plt.figure(figsize=(6, 5))
        models = ['No Causality', 'With Causality']
        errors = [indef.get('no_causality', 0), indef.get('with_causality', 0)]
        colors = ['#95A5A6', '#8E44AD']
        
        plt.bar(models, errors, color=colors, alpha=0.8, width=0.5)
        plt.ylabel('Relative Test Error', fontsize=12)
        plt.title('Indefinite Curvature (Minkowski): Causality Impact', fontsize=14)
        for i, v in enumerate(errors):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=11)
        plt.tight_layout()
        plt.savefig('plot_unified_indefinite.png', dpi=300)
        print("Generated plot_unified_indefinite.png")

if __name__ == "__main__":
    plot_curvature_sensitivity()
    plot_torus_comparison()
    plot_lorentzian_results()
    plot_unified_validation()
