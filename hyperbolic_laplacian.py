"""
Hyperbolic Laplacian Implementation
Graph-based approximation for Poincaré disk
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, cg
from scipy.spatial import KDTree
import time

# Import from previous experiment
from hyperbolic_geometry import (
    poincare_to_hyperboloid, 
    hyperbolic_distance,
    hyperboloid_to_poincare
)


def sample_poincare_disk_uniform_hyperbolic(n_points, R=1.0, max_radius=0.95):
    """
    Sample points uniformly in hyperbolic distance (exponential density in Poincaré).
    """
    # In Poincaré disk, uniform hyperbolic sampling requires density ~ 1/(1-r²)²
    # Use rejection sampling
    points = []
    
    while len(points) < n_points:
        # Sample uniformly in disk
        angle = np.random.rand() * 2 * np.pi
        r = np.random.rand() * max_radius
        
        u = r * np.cos(angle)
        v = r * np.sin(angle)
        
        # Accept with probability proportional to hyperbolic density
        density = 1.0 / ((1 - r**2)**2 + 1e-10)
        if np.random.rand() < min(density / 100, 1.0):  # Normalize
            points.append([u, v])
    
    return np.array(points[:n_points])


def build_hyperbolic_graph(points, k_neighbors=8, R=1.0):
    """
    Build graph with hyperbolic distance weights.
    
    Args:
        points: (n, 2) array of points in Poincaré disk
        k_neighbors: number of nearest neighbors
        R: hyperbolic radius
    
    Returns:
        adjacency matrix (sparse)
    """
    n = len(points)
    
    # Convert to hyperboloid for distance computation
    points_hyp = poincare_to_hyperboloid(points, R=R)
    
    # Build KD-tree for efficient neighbor search (in Euclidean space)
    tree = KDTree(points)
    
    # Build adjacency matrix
    rows = []
    cols = []
    data = []
    
    for i in range(n):
        # Find k nearest neighbors (Euclidean, as proxy)
        distances_eucl, indices = tree.query(points[i], k=k_neighbors+1)
        
        for j in indices[1:]:  # Skip self
            # Compute hyperbolic distance
            d_hyp = hyperbolic_distance(points_hyp[i], points_hyp[j], R=R)
            
            # Weight: Gaussian kernel
            weight = np.exp(-d_hyp**2 / (2 * 0.1**2))
            
            rows.append(i)
            cols.append(j)
            data.append(weight)
    
    # Create symmetric sparse matrix
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))
    adjacency = adjacency + adjacency.T
    
    return adjacency


def compute_graph_laplacian(adjacency):
    """
    Compute graph Laplacian: L = D - A
    where D is degree matrix, A is adjacency matrix.
    """
    # Degree matrix
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    D = csr_matrix((degrees, (range(len(degrees)), range(len(degrees)))), 
                   shape=adjacency.shape)
    
    # Laplacian
    L = D - adjacency
    
    return L


def solve_poisson_hyperbolic(L, source, regularization=1e-6):
    """
    Solve Δ_H u = f using graph Laplacian.
    
    Args:
        L: graph Laplacian (sparse)
        source: source term f
        regularization: small value to make system invertible
    
    Returns:
        solution u
    """
    # Add regularization to make system positive definite
    L_reg = L + regularization * csr_matrix(np.eye(L.shape[0]))
    
    # Solve using conjugate gradient
    solution, info = cg(L_reg, source, tol=1e-6, maxiter=1000)
    
    if info != 0:
        print(f"  Warning: CG did not converge (info={info})")
    
    return solution


def generate_smooth_source(points, n_centers=3, R=1.0):
    """Generate smooth source function as sum of Gaussians."""
    n = len(points)
    source = np.zeros(n)
    
    # Random centers
    center_indices = np.random.choice(n, n_centers, replace=False)
    
    points_hyp = poincare_to_hyperboloid(points, R=R)
    
    for center_idx in center_indices:
        center_hyp = points_hyp[center_idx]
        
        # Compute distances to center
        for i in range(n):
            d = hyperbolic_distance(center_hyp, points_hyp[i], R=R)
            source[i] += np.exp(-d**2 / (2 * 0.3**2))
    
    # Normalize
    source = source / (np.max(np.abs(source)) + 1e-10)
    
    return source


def generate_dataset(n_points=1000, n_samples=500, k_neighbors=8, R=1.0):
    """Generate dataset of Poisson equation solutions."""
    
    print(f"Generating hyperbolic Poisson dataset...")
    print(f"  Discretization: {n_points} points")
    print(f"  Samples: {n_samples}")
    
    # Sample points
    print("  Sampling points in Poincaré disk...")
    points = sample_poincare_disk_uniform_hyperbolic(n_points, R=R)
    
    # Build graph
    print("  Building hyperbolic graph...")
    adjacency = build_hyperbolic_graph(points, k_neighbors=k_neighbors, R=R)
    
    # Compute Laplacian
    print("  Computing graph Laplacian...")
    L = compute_graph_laplacian(adjacency)
    
    # Generate samples
    print("  Generating source-solution pairs...")
    sources = []
    solutions = []
    
    for i in range(n_samples):
        if i % 100 == 0:
            print(f"    Progress: {i}/{n_samples}")
        
        # Generate source
        source = generate_smooth_source(points, n_centers=3, R=R)
        
        # Solve for solution
        solution = solve_poisson_hyperbolic(L, source)
        
        sources.append(source)
        solutions.append(solution)
    
    return {
        'points': points,
        'sources': np.array(sources),
        'solutions': np.array(solutions),
        'adjacency': adjacency,
        'laplacian': L
    }


def visualize_laplacian_validation(data, save_path='laplacian_validation.png'):
    """Visualize Laplacian validation."""
    
    points = data['points']
    sources = data['sources']
    solutions = data['solutions']
    
    # Pick a random example
    idx = np.random.randint(len(sources))
    source = sources[idx]
    solution = solutions[idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot source
    ax1 = axes[0]
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], c=source, 
                          cmap='RdBu_r', s=20, alpha=0.7)
    circle1 = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(circle1)
    plt.colorbar(scatter1, ax=ax1, label='Source f')
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('u', fontsize=12)
    ax1.set_ylabel('v', fontsize=12)
    ax1.set_title('Source Function f', fontsize=14, fontweight='bold')
    
    # Plot solution
    ax2 = axes[1]
    scatter2 = ax2.scatter(points[:, 0], points[:, 1], c=solution,
                          cmap='viridis', s=20, alpha=0.7)
    circle2 = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax2.add_patch(circle2)
    plt.colorbar(scatter2, ax=ax2, label='Solution u')
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('u', fontsize=12)
    ax2.set_ylabel('v', fontsize=12)
    ax2.set_title('Solution u (Δ_H u = f)', fontsize=14, fontweight='bold')
    
    # Plot residual
    ax3 = axes[2]
    L = data['laplacian']
    residual = L.dot(solution) - source
    scatter3 = ax3.scatter(points[:, 0], points[:, 1], c=np.abs(residual),
                          cmap='plasma', s=20, alpha=0.7)
    circle3 = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax3.add_patch(circle3)
    plt.colorbar(scatter3, ax=ax3, label='|Residual|')
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_aspect('equal')
    ax3.set_xlabel('u', fontsize=12)
    ax3.set_ylabel('v', fontsize=12)
    ax3.set_title(f'Residual (max={np.max(np.abs(residual)):.4f})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved Laplacian validation to {save_path}")
    plt.close()


def analyze_discretization(save_path='discretization_analysis.png'):
    """Analyze discretization quality."""
    
    print("\nAnalyzing discretization quality...")
    
    n_points_list = [100, 300, 500, 1000]
    errors = []
    times = []
    
    for n_points in n_points_list:
        print(f"  Testing with {n_points} points...")
        
        # Sample points
        points = sample_poincare_disk_uniform_hyperbolic(n_points, R=1.0)
        
        # Build graph and Laplacian
        start_time = time.time()
        adjacency = build_hyperbolic_graph(points, k_neighbors=8, R=1.0)
        L = compute_graph_laplacian(adjacency)
        
        # Generate test problem
        source = generate_smooth_source(points, n_centers=2, R=1.0)
        solution = solve_poisson_hyperbolic(L, source)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        # Compute residual
        residual = L.dot(solution) - source
        error = np.max(np.abs(residual))
        errors.append(error)
        
        print(f"    Max residual: {error:.6f}, Time: {elapsed:.2f}s")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error vs discretization
    ax1.plot(n_points_list, errors, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Points', fontsize=12)
    ax1.set_ylabel('Max Residual Error', fontsize=12)
    ax1.set_title('Discretization Error', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Time vs discretization
    ax2.plot(n_points_list, times, 's-', linewidth=2, markersize=8, color='tab:orange')
    ax2.set_xlabel('Number of Points', fontsize=12)
    ax2.set_ylabel('Computation Time (s)', fontsize=12)
    ax2.set_title('Computational Cost', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved discretization analysis to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENT 4.2: HYPERBOLIC LAPLACIAN (GRAPH-BASED)")
    print("="*70)
    print()
    
    R = 1.0
    
    # Generate dataset
    data = generate_dataset(n_points=1000, n_samples=500, k_neighbors=8, R=R)
    
    # Save dataset
    print("\nSaving dataset...")
    np.savez('hyperbolic_test_data.npz',
             points=data['points'],
             sources=data['sources'],
             solutions=data['solutions'])
    print("  Saved hyperbolic_test_data.npz")
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_laplacian_validation(data)
    analyze_discretization()
    
    # Statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Number of points: {len(data['points'])}")
    print(f"Number of samples: {len(data['sources'])}")
    print(f"Source range: [{np.min(data['sources']):.4f}, {np.max(data['sources']):.4f}]")
    print(f"Solution range: [{np.min(data['solutions']):.4f}, {np.max(data['solutions']):.4f}]")
    
    print()
    print("="*70)
    print("EXPERIMENT 4.2 COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  - hyperbolic_laplacian.py (this module)")
    print("  - hyperbolic_test_data.npz (500 examples)")
    print("  - laplacian_validation.png")
    print("  - discretization_analysis.png")
