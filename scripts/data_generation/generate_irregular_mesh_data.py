"""
Experiment 9.1: Irregular Mesh Benchmark
Generate Poisson equation data on random point clouds (no grid structure).

This tests DeepONet's mesh-agnostic nature vs SFNO's grid requirement.
"""

import numpy as np
from scipy.special import sph_harm
import json

def generate_random_spherical_points(n_points=1000, R=1.0):
    """Generate uniformly distributed random points on sphere."""
    # Use Fibonacci sphere for better uniformity
    indices = np.arange(n_points) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    
    # Add small random perturbation to break grid structure
    phi += 0.1 * np.random.randn(n_points)
    theta += 0.1 * np.random.randn(n_points)
    
    # Clamp to valid ranges
    phi = np.clip(phi, 0.01, np.pi - 0.01)
    theta = theta % (2 * np.pi)
    
    # Convert to Cartesian
    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * np.cos(phi)
    
    return np.stack([x, y, z], axis=-1), phi, theta

def generate_random_source(theta, phi, L_max=5, n_modes=5):
    """Generate random source as sum of spherical harmonics."""
    source = np.zeros_like(theta)
    
    for _ in range(n_modes):
        l = np.random.randint(0, L_max + 1)
        m = np.random.randint(-l, l + 1)
        amplitude = np.random.randn() * 10.0
        
        Y_lm = sph_harm(m, l, theta, phi)
        source += amplitude * np.real(Y_lm)
    
    return source

def solve_poisson_spectral_irregular(source, theta, phi, L_max=10):
    """
    Solve Poisson equation on irregular points using spectral method.
    
    Steps:
    1. Project source onto spherical harmonic basis
    2. Divide by eigenvalues: λ_l = -l(l+1)
    3. Reconstruct solution
    """
    # Compute spherical harmonic coefficients of source
    source_coeffs = []
    
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, theta, phi)
            # Numerical integration (trapezoidal on irregular points)
            c_lm = np.mean(source * np.conj(Y_lm))
            source_coeffs.append((l, m, c_lm))
    
    # Solve: Δu = f => u_lm = -f_lm / l(l+1)
    solution = np.zeros_like(source, dtype=complex)
    
    for l, m, f_lm in source_coeffs:
        if l == 0:
            continue  # Skip constant mode (not invertible)
        
        eigenvalue = -l * (l + 1)
        u_lm = f_lm / eigenvalue
        
        Y_lm = sph_harm(m, l, theta, phi)
        solution += u_lm * Y_lm
    
    return np.real(solution)

def generate_irregular_poisson_dataset(n_samples=1000, n_points=1000, output_file='irregular_poisson_data.npz'):
    """Generate complete irregular mesh dataset."""
    print(f"Generating {n_samples} samples with {n_points} irregular points each...")
    
    all_coords = []
    all_sources = []
    all_solutions = []
    
    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples...")
        
        # Random points
        coords, phi, theta = generate_random_spherical_points(n_points)
        
        # Random source
        source = generate_random_source(theta, phi)
        
        # Solve Poisson
        solution = solve_poisson_spectral_irregular(source, theta, phi)
        
        all_coords.append(coords)
        all_sources.append(source)
        all_solutions.append(solution)
    
    # Save
    np.savez(
        output_file,
        coords=np.array(all_coords),
        sources=np.array(all_sources),
        solutions=np.array(all_solutions)
    )
    
    print(f"\nSaved to {output_file}")
    print(f"  Coords shape: {np.array(all_coords).shape}")
    print(f"  Sources shape: {np.array(all_sources).shape}")
    print(f"  Solutions shape: {np.array(all_solutions).shape}")
    
    # Statistics
    stats = {
        'n_samples': n_samples,
        'n_points': n_points,
        'source_mean': float(np.mean(all_sources)),
        'source_std': float(np.std(all_sources)),
        'solution_mean': float(np.mean(all_solutions)),
        'solution_std': float(np.std(all_solutions))
    }
    
    with open('irregular_mesh_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

if __name__ == "__main__":
    print("="*60)
    print("EXPERIMENT 9.1: IRREGULAR MESH DATA GENERATION")
    print("="*60)
    
    # Generate training data
    print("\nGenerating training data...")
    train_stats = generate_irregular_poisson_dataset(
        n_samples=800,
        n_points=1000,
        output_file='train_irregular_poisson.npz'
    )
    
    # Generate test data
    print("\nGenerating test data...")
    test_stats = generate_irregular_poisson_dataset(
        n_samples=200,
        n_points=1000,
        output_file='test_irregular_poisson.npz'
    )
    
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE")
    print("="*60)
    print(f"Training samples: {train_stats['n_samples']}")
    print(f"Test samples: {test_stats['n_samples']}")
    print(f"Points per sample: {train_stats['n_points']}")
