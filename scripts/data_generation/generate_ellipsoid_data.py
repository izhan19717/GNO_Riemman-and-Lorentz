"""
Experiment 9.3: Deformed Geometry (Ellipsoid)
Generate Poisson equation data on an ellipsoid instead of sphere.

This tests whether SFNO (spherical harmonic basis) breaks down
when the geometry is deformed.
"""

import numpy as np
from scipy.special import sph_harm
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve
import json

def generate_ellipsoid_mesh(a=1.0, b=0.8, c=1.2, n_theta=50, n_phi=100):
    """
    Generate ellipsoid mesh: x²/a² + y²/b² + z²/c² = 1
    
    Args:
        a, b, c: Semi-axes lengths
        n_theta, n_phi: Grid resolution
    
    Returns:
        X, Y, Z: Cartesian coordinates
        Theta, Phi: Parametric coordinates
    """
    theta = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    
    # Parametric ellipsoid
    X = a * np.sin(Theta) * np.cos(Phi)
    Y = b * np.sin(Theta) * np.sin(Phi)
    Z = c * np.cos(Theta)
    
    return X, Y, Z, Theta, Phi, theta, phi

def compute_ellipsoid_metric_coefficients(a, b, c, theta, phi):
    """
    Compute first fundamental form coefficients for ellipsoid.
    
    Returns:
        E, F, G: Metric tensor components
    """
    # Partial derivatives
    # r_theta = (a*cos(θ)*cos(φ), b*cos(θ)*sin(φ), -c*sin(θ))
    # r_phi = (-a*sin(θ)*sin(φ), b*sin(θ)*cos(φ), 0)
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # E = <r_theta, r_theta>
    E = (a**2 * cos_theta**2 * cos_phi**2 + 
         b**2 * cos_theta**2 * sin_phi**2 + 
         c**2 * sin_theta**2)
    
    # F = <r_theta, r_phi>
    F = (a**2 - b**2) * sin_theta * cos_theta * cos_phi * sin_phi
    
    # G = <r_phi, r_phi>
    G = (a**2 * sin_theta**2 * sin_phi**2 + 
         b**2 * sin_theta**2 * cos_phi**2)
    
    return E, F, G

def build_laplace_beltrami_ellipsoid(a, b, c, theta, phi):
    """
    Build Laplace-Beltrami operator matrix for ellipsoid using finite differences.
    
    Δ_g u = (1/√g) * ∂_i(√g * g^{ij} * ∂_j u)
    
    where g is the metric determinant and g^{ij} is the inverse metric.
    """
    n_theta = len(theta)
    n_phi = len(phi)
    n = n_theta * n_phi
    
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    
    # Build sparse matrix
    L = lil_matrix((n, n))
    
    for i in range(n_theta):
        for j in range(n_phi):
            idx = i * n_phi + j
            
            # Compute metric at this point
            E, F, G = compute_ellipsoid_metric_coefficients(a, b, c, theta[i], phi[j])
            
            # Metric determinant
            g = E * G - F**2
            sqrt_g = np.sqrt(g)
            
            # Inverse metric
            g_inv_11 = G / g
            g_inv_12 = -F / g
            g_inv_22 = E / g
            
            # Central differences for Laplace-Beltrami
            # Simplified version (assumes small F)
            coeff_theta = sqrt_g * g_inv_11 / (dtheta**2)
            coeff_phi = sqrt_g * g_inv_22 / (dphi**2)
            
            # Diagonal
            L[idx, idx] = -2 * (coeff_theta + coeff_phi) / sqrt_g
            
            # Theta neighbors
            if i > 0:
                idx_prev = (i-1) * n_phi + j
                L[idx, idx_prev] = coeff_theta / sqrt_g
            if i < n_theta - 1:
                idx_next = (i+1) * n_phi + j
                L[idx, idx_next] = coeff_theta / sqrt_g
            
            # Phi neighbors (periodic)
            j_prev = (j - 1) % n_phi
            j_next = (j + 1) % n_phi
            idx_prev = i * n_phi + j_prev
            idx_next = i * n_phi + j_next
            L[idx, idx_prev] = coeff_phi / sqrt_g
            L[idx, idx_next] = coeff_phi / sqrt_g
    
    return L.tocsr()

def solve_poisson_on_ellipsoid(source, a, b, c, theta, phi):
    """Solve Δ_g u = f on ellipsoid."""
    n_theta = len(theta)
    n_phi = len(phi)
    
    # Build Laplace-Beltrami operator
    L = build_laplace_beltrami_ellipsoid(a, b, c, theta, phi)
    
    # Flatten source
    f = source.flatten()
    
    # Solve (add small regularization for stability)
    L_reg = L - 1e-6 * diags(np.ones(L.shape[0]))
    u = spsolve(L_reg, f)
    
    # Reshape
    return u.reshape(n_theta, n_phi)

def generate_ellipsoid_poisson_dataset(n_samples=1000, a=1.0, b=0.8, c=1.2, 
                                       n_theta=50, n_phi=100,
                                       output_file='ellipsoid_poisson_data.npz'):
    """Generate complete ellipsoid Poisson dataset."""
    print(f"Generating {n_samples} samples on ellipsoid (a={a}, b={b}, c={c})...")
    print(f"Grid: {n_theta} x {n_phi} = {n_theta*n_phi} points")
    
    # Generate mesh
    X, Y, Z, Theta, Phi, theta, phi = generate_ellipsoid_mesh(a, b, c, n_theta, n_phi)
    
    all_sources = []
    all_solutions = []
    
    for i in range(n_samples):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples...")
        
        # Random source (using spherical harmonics as basis, even though not eigenfunctions)
        source = np.zeros_like(Theta)
        for _ in range(5):
            l = np.random.randint(0, 6)
            m = np.random.randint(-l, l + 1)
            amplitude = np.random.randn() * 10.0
            Y_lm = sph_harm(m, l, Phi, Theta)
            source += amplitude * np.real(Y_lm)
        
        # Solve Poisson on ellipsoid
        solution = solve_poisson_on_ellipsoid(source, a, b, c, theta, phi)
        
        all_sources.append(source)
        all_solutions.append(solution)
    
    # Save
    np.savez(
        output_file,
        sources=np.array(all_sources),
        solutions=np.array(all_solutions),
        X=X, Y=Y, Z=Z,
        theta=theta,
        phi=phi,
        a=a, b=b, c=c
    )
    
    print(f"\nSaved to {output_file}")
    
    # Statistics
    stats = {
        'n_samples': n_samples,
        'ellipsoid_params': {'a': a, 'b': b, 'c': c},
        'grid_size': {'n_theta': n_theta, 'n_phi': n_phi},
        'source_mean': float(np.mean(all_sources)),
        'source_std': float(np.std(all_sources)),
        'solution_mean': float(np.mean(all_solutions)),
        'solution_std': float(np.std(all_solutions))
    }
    
    with open('ellipsoid_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

if __name__ == "__main__":
    print("="*60)
    print("EXPERIMENT 9.3: ELLIPSOID GEOMETRY DATA GENERATION")
    print("="*60)
    
    # Generate training data
    print("\nGenerating training data...")
    train_stats = generate_ellipsoid_poisson_dataset(
        n_samples=800,
        a=1.0, b=0.8, c=1.2,
        n_theta=50, n_phi=100,
        output_file='train_ellipsoid_poisson.npz'
    )
    
    # Generate test data
    print("\nGenerating test data...")
    test_stats = generate_ellipsoid_poisson_dataset(
        n_samples=200,
        a=1.0, b=0.8, c=1.2,
        n_theta=50, n_phi=100,
        output_file='test_ellipsoid_poisson.npz'
    )
    
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE")
    print("="*60)
    print(f"Ellipsoid: a={train_stats['ellipsoid_params']['a']}, "
          f"b={train_stats['ellipsoid_params']['b']}, "
          f"c={train_stats['ellipsoid_params']['c']}")
    print(f"Training samples: {train_stats['n_samples']}")
    print(f"Test samples: {test_stats['n_samples']}")
