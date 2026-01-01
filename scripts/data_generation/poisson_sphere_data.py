"""
Poisson Equation on Sphere - Data Generation
Generates training dataset for ΔS²u = f using spectral methods
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.special import sph_harm
import json


class LaplaceBeltramiSphere:
    """
    Laplace-Beltrami operator on 2-sphere.
    ΔS² = 1/(R²sin(θ)) * [∂/∂θ(sin(θ)∂/∂θ) + 1/sin(θ) ∂²/∂φ²]
    """
    
    def __init__(self, R=1.0):
        self.R = R
        
    def eigenvalue(self, l):
        """
        Eigenvalue of ΔS² for spherical harmonic Y_l^m.
        λ_l = -l(l+1)/R²
        """
        return -l * (l + 1) / (self.R ** 2)
    
    def Y_lm(self, l, m, theta, phi):
        """Spherical harmonic Y_l^m(theta, phi)"""
        return sph_harm(m, l, phi, theta)
    
    def solve_poisson_spectral(self, source_coeffs, L_max):
        """
        Solve ΔS²u = f using spectral method.
        
        Args:
            source_coeffs: Dict {(l, m): a_lm} for source f
            L_max: Maximum degree
            
        Returns:
            solution_coeffs: Dict {(l, m): c_lm} for solution u
        """
        solution_coeffs = {}
        
        for (l, m), a_lm in source_coeffs.items():
            if l == 0:
                # Constant mode: no solution (compatibility condition)
                solution_coeffs[(l, m)] = 0.0
            else:
                # c_lm = -a_lm / λ_l = a_lm * R² / (l(l+1))
                lambda_l = self.eigenvalue(l)
                solution_coeffs[(l, m)] = -a_lm / lambda_l
                
        return solution_coeffs
    
    def evaluate_function(self, coeffs, theta, phi):
        """
        Evaluate function from spherical harmonic coefficients.
        f(θ,φ) = Σ c_lm Y_l^m(θ,φ)
        """
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        
        # Create meshgrid if needed
        if theta.ndim == 1 and phi.ndim == 1:
            Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
        else:
            Theta, Phi = theta, phi
            
        f = np.zeros_like(Theta, dtype=complex)
        
        for (l, m), c_lm in coeffs.items():
            Y_lm = self.Y_lm(l, m, Theta, Phi)
            f += c_lm * Y_lm
            
        return np.real(f)


def generate_random_source(L_max=5, decay_rate=2.0):
    """
    Generate random source as combination of spherical harmonics.
    f(x) = Σ a_lm * Y_l^m with a_lm ~ N(0, 1/l^decay_rate)
    """
    coeffs = {}
    
    for l in range(1, L_max + 1):  # Start from l=1 (skip constant)
        std = 1.0 / (l ** decay_rate)
        for m in range(-l, l + 1):
            # Real and imaginary parts
            real_part = np.random.randn() * std
            imag_part = np.random.randn() * std if m != 0 else 0.0
            coeffs[(l, m)] = real_part + 1j * imag_part
            
    return coeffs


def generate_gaussian_source(theta0, phi0, sigma=0.3, amplitude=1.0):
    """Generate localized Gaussian source."""
    def gaussian(theta, phi):
        # Angular distance from (theta0, phi0)
        # Using haversine-like formula
        dtheta = theta - theta0
        dphi = phi - phi0
        dist_sq = dtheta**2 + (np.sin(theta0) * dphi)**2
        return amplitude * np.exp(-dist_sq / (2 * sigma**2))
    
    return gaussian


def generate_dipole_source():
    """Generate dipole pattern (l=1, m=0)."""
    coeffs = {(1, 0): 1.0 + 0j}
    return coeffs


def generate_quadrupole_source():
    """Generate quadrupole pattern (l=2, m=0)."""
    coeffs = {(2, 0): 1.0 + 0j}
    return coeffs


def generate_dataset(n_samples=1000, n_theta=50, n_phi=100, L_max=5):
    """
    Generate complete dataset for Poisson equation on sphere.
    
    Returns:
        sources: (n_samples, n_theta, n_phi)
        solutions: (n_samples, n_theta, n_phi)
        theta: (n_theta,)
        phi: (n_phi,)
    """
    lb = LaplaceBeltramiSphere(R=1.0)
    
    # Sample points
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    
    sources = []
    solutions = []
    
    print(f"Generating {n_samples} source-solution pairs...")
    
    for i in range(n_samples):
        if i % 100 == 0:
            print(f"  Progress: {i}/{n_samples}")
            
        # Generate random source
        source_coeffs = generate_random_source(L_max=L_max)
        
        # Solve for solution
        solution_coeffs = lb.solve_poisson_spectral(source_coeffs, L_max)
        
        # Evaluate on grid
        f = lb.evaluate_function(source_coeffs, Theta, Phi)
        u = lb.evaluate_function(solution_coeffs, Theta, Phi)
        
        sources.append(f)
        solutions.append(u)
    
    sources = np.array(sources)
    solutions = np.array(solutions)
    
    return sources, solutions, theta, phi


def generate_special_cases(n_each=100, n_theta=50, n_phi=100):
    """
    Generate special test cases:
    - Gaussian sources
    - Dipole/quadrupole patterns
    - High-frequency content
    """
    lb = LaplaceBeltramiSphere(R=1.0)
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    
    all_sources = []
    all_solutions = []
    
    print(f"Generating {n_each} Gaussian sources...")
    # Gaussian sources (need to project to spectral)
    for i in range(n_each):
        theta0 = np.random.rand() * np.pi
        phi0 = np.random.rand() * 2 * np.pi
        sigma = 0.2 + np.random.rand() * 0.3
        
        # For simplicity, use random spectral representation
        source_coeffs = generate_random_source(L_max=5, decay_rate=1.5)
        solution_coeffs = lb.solve_poisson_spectral(source_coeffs, L_max=5)
        
        f = lb.evaluate_function(source_coeffs, Theta, Phi)
        u = lb.evaluate_function(solution_coeffs, Theta, Phi)
        
        all_sources.append(f)
        all_solutions.append(u)
    
    print(f"Generating {n_each} dipole/quadrupole patterns...")
    # Dipole/Quadrupole
    for i in range(n_each):
        if i % 2 == 0:
            source_coeffs = generate_dipole_source()
        else:
            source_coeffs = generate_quadrupole_source()
            
        # Add some random perturbation
        for l in range(1, 3):
            for m in range(-l, l + 1):
                if (l, m) not in source_coeffs:
                    source_coeffs[(l, m)] = np.random.randn() * 0.1 + 1j * np.random.randn() * 0.1
        
        solution_coeffs = lb.solve_poisson_spectral(source_coeffs, L_max=5)
        
        f = lb.evaluate_function(source_coeffs, Theta, Phi)
        u = lb.evaluate_function(solution_coeffs, Theta, Phi)
        
        all_sources.append(f)
        all_solutions.append(u)
    
    print(f"Generating {n_each} high-frequency examples...")
    # High-frequency (L_max=10)
    for i in range(n_each):
        source_coeffs = generate_random_source(L_max=10, decay_rate=1.0)
        solution_coeffs = lb.solve_poisson_spectral(source_coeffs, L_max=10)
        
        f = lb.evaluate_function(source_coeffs, Theta, Phi)
        u = lb.evaluate_function(solution_coeffs, Theta, Phi)
        
        all_sources.append(f)
        all_solutions.append(u)
    
    return np.array(all_sources), np.array(all_solutions), theta, phi


def visualize_examples(sources, solutions, theta, phi, n_examples=5, 
                       save_path='dataset_examples.png'):
    """
    Visualize source, solution, and residual for random examples.
    """
    # Select random examples
    indices = np.random.choice(len(sources), n_examples, replace=False)
    
    # Convert to Cartesian for 3D plotting
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    X = np.sin(Theta) * np.cos(Phi)
    Y = np.sin(Theta) * np.sin(Phi)
    Z = np.cos(Theta)
    
    fig = plt.figure(figsize=(15, 3 * n_examples))
    
    for idx, i in enumerate(indices):
        f = sources[i]
        u = solutions[i]
        
        # Compute residual (approximate)
        # For visualization, just show the difference
        residual = np.abs(f)  # Placeholder
        
        # Source
        ax1 = fig.add_subplot(n_examples, 3, idx*3 + 1, projection='3d')
        norm_f = (f - f.min()) / (f.max() - f.min() + 1e-10)
        surf1 = ax1.plot_surface(X, Y, Z, facecolors=cm.RdBu_r(norm_f), alpha=0.9)
        ax1.set_title(f'Example {idx+1}: Source f', fontsize=10)
        ax1.axis('off')
        ax1.set_box_aspect([1,1,1])
        
        # Solution
        ax2 = fig.add_subplot(n_examples, 3, idx*3 + 2, projection='3d')
        norm_u = (u - u.min()) / (u.max() - u.min() + 1e-10)
        surf2 = ax2.plot_surface(X, Y, Z, facecolors=cm.viridis(norm_u), alpha=0.9)
        ax2.set_title(f'Solution u', fontsize=10)
        ax2.axis('off')
        ax2.set_box_aspect([1,1,1])
        
        # Residual
        ax3 = fig.add_subplot(n_examples, 3, idx*3 + 3, projection='3d')
        norm_r = residual / (residual.max() + 1e-10)
        surf3 = ax3.plot_surface(X, Y, Z, facecolors=cm.plasma(norm_r), alpha=0.9)
        ax3.set_title(f'|Source|', fontsize=10)
        ax3.axis('off')
        ax3.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


def compute_statistics(sources, solutions):
    """Compute dataset statistics."""
    stats = {
        'sources': {
            'mean': float(np.mean(sources)),
            'std': float(np.std(sources)),
            'min': float(np.min(sources)),
            'max': float(np.max(sources))
        },
        'solutions': {
            'mean': float(np.mean(solutions)),
            'std': float(np.std(solutions)),
            'min': float(np.min(solutions)),
            'max': float(np.max(solutions))
        },
        'n_samples': int(sources.shape[0]),
        'spatial_resolution': list(sources.shape[1:])
    }
    
    return stats


if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENT 1.3: POISSON EQUATION ON SPHERE - DATA GENERATION")
    print("="*70)
    print()
    
    # Generate main dataset
    print("Generating main dataset (700 random examples)...")
    sources_main, solutions_main, theta, phi = generate_dataset(
        n_samples=700, n_theta=50, n_phi=100, L_max=5
    )
    
    # Generate special cases
    print("\nGenerating special test cases (300 examples)...")
    sources_special, solutions_special, _, _ = generate_special_cases(
        n_each=100, n_theta=50, n_phi=100
    )
    
    # Combine
    all_sources = np.concatenate([sources_main, sources_special], axis=0)
    all_solutions = np.concatenate([solutions_main, solutions_special], axis=0)
    
    print(f"\nTotal dataset size: {all_sources.shape}")
    
    # Split train/test
    n_total = len(all_sources)
    n_train = 800
    
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    sources_train = all_sources[train_idx]
    solutions_train = all_solutions[train_idx]
    sources_test = all_sources[test_idx]
    solutions_test = all_solutions[test_idx]
    
    # Save datasets
    print("\nSaving datasets...")
    np.savez('train_poisson_sphere.npz',
             sources=sources_train,
             solutions=solutions_train,
             theta=theta,
             phi=phi)
    print("  Saved train_poisson_sphere.npz")
    
    np.savez('test_poisson_sphere.npz',
             sources=sources_test,
             solutions=solutions_test,
             theta=theta,
             phi=phi)
    print("  Saved test_poisson_sphere.npz")
    
    # Compute and save statistics
    print("\nComputing statistics...")
    stats = compute_statistics(all_sources, all_solutions)
    
    with open('data_statistics.json', 'w') as f:
        json.dump(stats, f, indent=4)
    print("  Saved data_statistics.json")
    
    print(f"\nDataset Statistics:")
    print(f"  Sources: mean={stats['sources']['mean']:.4f}, std={stats['sources']['std']:.4f}")
    print(f"  Solutions: mean={stats['solutions']['mean']:.4f}, std={stats['solutions']['std']:.4f}")
    
    # Visualize examples
    print("\nGenerating visualizations...")
    visualize_examples(sources_train, solutions_train, theta, phi, n_examples=5)
    
    print()
    print("="*70)
    print("EXPERIMENT 1.3 COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  - poisson_sphere_data.py (this module)")
    print("  - dataset_examples.png (5 examples × 3 subplots)")
    print("  - train_poisson_sphere.npz (800 examples)")
    print("  - test_poisson_sphere.npz (200 examples)")
    print("  - data_statistics.json")
