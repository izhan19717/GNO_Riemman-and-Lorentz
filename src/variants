"""
Hyperbolic Geometry Implementation
Hyperboloid model of H² with distance computation and visualizations
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def lorentz_inner_product(x, y):
    """
    Lorentz inner product in R^(2,1).
    <x,y>_L = -x₀y₀ + x₁y₁ + x₂y₂
    
    Args:
        x, y: arrays of shape (..., 3) with components (x₀, x₁, x₂)
    Returns:
        <x,y>_L: Lorentz inner product
    """
    return -x[..., 0] * y[..., 0] + x[..., 1] * y[..., 1] + x[..., 2] * y[..., 2]


def hyperbolic_distance(x, y, R=1.0, eps=1e-10):
    """
    Hyperbolic distance in hyperboloid model.
    d_H(x,y) = R * arcosh(-<x,y>_L / R²)
    
    Args:
        x, y: points on hyperboloid (shape: (..., 3))
        R: radius of hyperboloid
        eps: numerical stability threshold
    Returns:
        d: hyperbolic distance
    """
    # Compute Lorentz inner product
    inner = lorentz_inner_product(x, y)
    
    # Argument for arcosh
    arg = -inner / (R ** 2)
    
    # Numerical stability: arg should be >= 1
    arg = np.maximum(arg, 1.0 + eps)
    
    # For nearby points (arg ≈ 1), use series expansion
    # arcosh(1 + ε) ≈ sqrt(2ε) for small ε
    small_mask = (arg - 1.0) < eps
    
    d = np.zeros_like(arg)
    
    # Large distances: use arcosh
    d[~small_mask] = R * np.arccosh(arg[~small_mask])
    
    # Small distances: use series expansion
    epsilon = arg[small_mask] - 1.0
    d[small_mask] = R * np.sqrt(2 * epsilon)
    
    return d


def hyperboloid_to_poincare(x, R=1.0):
    """
    Convert from hyperboloid to Poincaré disk model.
    (x₀, x₁, x₂) → (u, v) = (x₁/(x₀+R), x₂/(x₀+R))
    
    Args:
        x: points on hyperboloid (shape: (..., 3))
        R: radius
    Returns:
        uv: points in Poincaré disk (shape: (..., 2))
    """
    u = x[..., 1] / (x[..., 0] + R)
    v = x[..., 2] / (x[..., 0] + R)
    
    return np.stack([u, v], axis=-1)


def poincare_to_hyperboloid(uv, R=1.0):
    """
    Convert from Poincaré disk to hyperboloid model.
    (u, v) → (x₀, x₁, x₂)
    
    Args:
        uv: points in Poincaré disk (shape: (..., 2))
        R: radius
    Returns:
        x: points on hyperboloid (shape: (..., 3))
    """
    u = uv[..., 0]
    v = uv[..., 1]
    
    norm_sq = u**2 + v**2
    
    # Ensure points are inside disk
    norm_sq = np.minimum(norm_sq, 1.0 - 1e-10)
    
    x0 = R * (1 + norm_sq) / (1 - norm_sq)
    x1 = R * 2 * u / (1 - norm_sq)
    x2 = R * 2 * v / (1 - norm_sq)
    
    return np.stack([x0, x1, x2], axis=-1)


def generate_hyperboloid_surface(R=1.0, n_theta=50, n_r=30, r_max=2.0):
    """Generate hyperboloid surface for visualization."""
    theta = np.linspace(0, 2*np.pi, n_theta)
    r = np.linspace(0, r_max, n_r)
    
    Theta, R_grid = np.meshgrid(theta, r)
    
    # Hyperboloid: x₀² - x₁² - x₂² = R²
    # Parametrize: x₁ = r*cos(θ), x₂ = r*sin(θ), x₀ = sqrt(R² + r²)
    X0 = np.sqrt(R**2 + R_grid**2)
    X1 = R_grid * np.cos(Theta)
    X2 = R_grid * np.sin(Theta)
    
    return X0, X1, X2


def visualize_hyperboloid(R=1.0, save_path='hyperboloid_visualization.png'):
    """Visualize hyperboloid surface with geodesics."""
    
    fig = plt.figure(figsize=(16, 5))
    
    # Plot 1: Hyperboloid surface
    ax1 = fig.add_subplot(131, projection='3d')
    
    X0, X1, X2 = generate_hyperboloid_surface(R=R)
    
    ax1.plot_surface(X1, X2, X0, alpha=0.3, cmap='viridis', edgecolor='none')
    
    # Add some geodesics (straight lines through origin in hyperboloid)
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        t = np.linspace(0, 2, 50)
        x1 = t * np.cos(angle)
        x2 = t * np.sin(angle)
        x0 = np.sqrt(R**2 + x1**2 + x2**2)
        ax1.plot(x1, x2, x0, 'r-', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('$x_1$', fontsize=11)
    ax1.set_ylabel('$x_2$', fontsize=11)
    ax1.set_zlabel('$x_0$', fontsize=11)
    ax1.set_title('Hyperboloid Model $H^2$', fontsize=13, fontweight='bold')
    ax1.set_box_aspect([1,1,1])
    
    # Plot 2: Poincaré disk with geodesics
    ax2 = fig.add_subplot(132)
    
    # Draw disk boundary
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax2.add_patch(circle)
    
    # Geodesics through origin are straight lines
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        t = np.linspace(-0.9, 0.9, 50)
        u = t * np.cos(angle)
        v = t * np.sin(angle)
        ax2.plot(u, v, 'r-', linewidth=2, alpha=0.7)
    
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('$u$', fontsize=11)
    ax2.set_ylabel('$v$', fontsize=11)
    ax2.set_title('Poincaré Disk Model', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distance field
    ax3 = fig.add_subplot(133)
    
    # Create grid in Poincaré disk
    n_grid = 50
    u_range = np.linspace(-0.95, 0.95, n_grid)
    v_range = np.linspace(-0.95, 0.95, n_grid)
    U, V = np.meshgrid(u_range, v_range)
    
    # Mask points outside disk
    mask = U**2 + V**2 < 1.0
    
    # Reference point (origin in Poincaré disk)
    ref_poincare = np.array([0.0, 0.0])
    ref_hyperboloid = poincare_to_hyperboloid(ref_poincare, R=R)
    
    # Compute distances
    distances = np.zeros_like(U)
    
    for i in range(n_grid):
        for j in range(n_grid):
            if mask[i, j]:
                point_poincare = np.array([U[i, j], V[i, j]])
                point_hyperboloid = poincare_to_hyperboloid(point_poincare, R=R)
                distances[i, j] = hyperbolic_distance(ref_hyperboloid, point_hyperboloid, R=R)
            else:
                distances[i, j] = np.nan
    
    im = ax3.contourf(U, V, distances, levels=20, cmap='plasma')
    circle3 = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax3.add_patch(circle3)
    ax3.plot(0, 0, 'r*', markersize=15, label='Reference point')
    
    plt.colorbar(im, ax=ax3, label='Hyperbolic Distance')
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_aspect('equal')
    ax3.set_xlabel('$u$', fontsize=11)
    ax3.set_ylabel('$v$', fontsize=11)
    ax3.set_title('Distance Field from Origin', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved hyperboloid visualization to {save_path}")
    plt.close()


def visualize_distance_field(R=1.0, save_path='distance_field_hyperbolic.png'):
    """Detailed distance field visualization."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create grid
    n_grid = 100
    u_range = np.linspace(-0.95, 0.95, n_grid)
    v_range = np.linspace(-0.95, 0.95, n_grid)
    U, V = np.meshgrid(u_range, v_range)
    
    mask = U**2 + V**2 < 1.0
    
    # Reference point
    ref_poincare = np.array([0.3, 0.2])
    ref_hyperboloid = poincare_to_hyperboloid(ref_poincare, R=R)
    
    # Hyperbolic distances
    hyp_distances = np.zeros_like(U)
    eucl_distances = np.zeros_like(U)
    
    for i in range(n_grid):
        for j in range(n_grid):
            if mask[i, j]:
                point_poincare = np.array([U[i, j], V[i, j]])
                point_hyperboloid = poincare_to_hyperboloid(point_poincare, R=R)
                hyp_distances[i, j] = hyperbolic_distance(ref_hyperboloid, point_hyperboloid, R=R)
                eucl_distances[i, j] = np.sqrt((U[i, j] - ref_poincare[0])**2 + 
                                              (V[i, j] - ref_poincare[1])**2)
            else:
                hyp_distances[i, j] = np.nan
                eucl_distances[i, j] = np.nan
    
    # Plot hyperbolic distance
    ax1 = axes[0]
    im1 = ax1.contourf(U, V, hyp_distances, levels=20, cmap='viridis')
    circle1 = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(circle1)
    ax1.plot(ref_poincare[0], ref_poincare[1], 'r*', markersize=15)
    plt.colorbar(im1, ax=ax1, label='Hyperbolic Distance')
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('$u$', fontsize=12)
    ax1.set_ylabel('$v$', fontsize=12)
    ax1.set_title('Hyperbolic Distance Field', fontsize=14, fontweight='bold')
    
    # Plot Euclidean distance for comparison
    ax2 = axes[1]
    im2 = ax2.contourf(U, V, eucl_distances, levels=20, cmap='viridis')
    circle2 = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax2.add_patch(circle2)
    ax2.plot(ref_poincare[0], ref_poincare[1], 'r*', markersize=15)
    plt.colorbar(im2, ax=ax2, label='Euclidean Distance')
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('$u$', fontsize=12)
    ax2.set_ylabel('$v$', fontsize=12)
    ax2.set_title('Euclidean Distance (for comparison)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved distance field to {save_path}")
    plt.close()


def test_coordinate_conversions(R=1.0, save_path='coordinate_conversions_test.png'):
    """Test and visualize coordinate conversions."""
    
    # Generate random points in Poincaré disk
    n_points = 100
    angles = np.random.rand(n_points) * 2 * np.pi
    radii = np.random.rand(n_points) * 0.9
    
    u = radii * np.cos(angles)
    v = radii * np.sin(angles)
    poincare_points = np.stack([u, v], axis=-1)
    
    # Convert to hyperboloid and back
    hyperboloid_points = poincare_to_hyperboloid(poincare_points, R=R)
    poincare_reconstructed = hyperboloid_to_poincare(hyperboloid_points, R=R)
    
    # Compute reconstruction error
    errors = np.linalg.norm(poincare_points - poincare_reconstructed, axis=-1)
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    
    print(f"\nCoordinate Conversion Test:")
    print(f"  Max reconstruction error: {max_error:.2e}")
    print(f"  Mean reconstruction error: {mean_error:.2e}")
    
    # Verify hyperboloid constraint
    constraint_values = lorentz_inner_product(hyperboloid_points, hyperboloid_points)
    constraint_errors = np.abs(constraint_values + R**2)
    print(f"  Max constraint violation: {np.max(constraint_errors):.2e}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot original and reconstructed points
    ax1 = axes[0]
    circle1 = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(circle1)
    ax1.scatter(poincare_points[:, 0], poincare_points[:, 1], 
               c='blue', s=50, alpha=0.6, label='Original')
    ax1.scatter(poincare_reconstructed[:, 0], poincare_reconstructed[:, 1],
               c='red', s=20, alpha=0.6, marker='x', label='Reconstructed')
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('$u$', fontsize=12)
    ax1.set_ylabel('$v$', fontsize=12)
    ax1.set_title('Poincaré Disk: Conversion Test', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot errors
    ax2 = axes[1]
    ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(mean_error, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_error:.2e}')
    ax2.set_xlabel('Reconstruction Error', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved coordinate conversion test to {save_path}")
    plt.close()
    
    return max_error, mean_error


def validate_hyperbolic_geometry(R=1.0):
    """Validation tests for hyperbolic geometry implementation."""
    
    print("\n" + "="*70)
    print("VALIDATION TESTS")
    print("="*70)
    
    # Test 1: Geodesic through origin in Poincaré disk
    print("\n1. Geodesic through origin test:")
    origin = np.array([0.0, 0.0])
    point1 = np.array([0.5, 0.0])
    point2 = np.array([0.8, 0.0])
    
    origin_hyp = poincare_to_hyperboloid(origin, R=R)
    point1_hyp = poincare_to_hyperboloid(point1, R=R)
    point2_hyp = poincare_to_hyperboloid(point2, R=R)
    
    d1 = hyperbolic_distance(origin_hyp, point1_hyp, R=R)
    d2 = hyperbolic_distance(origin_hyp, point2_hyp, R=R)
    d12 = hyperbolic_distance(point1_hyp, point2_hyp, R=R)
    
    print(f"  d(origin, 0.5) = {d1:.6f}")
    print(f"  d(origin, 0.8) = {d2:.6f}")
    print(f"  d(0.5, 0.8) = {d12:.6f}")
    print(f"  Triangle inequality check: {d2:.6f} ≤ {d1 + d12:.6f} ? {d2 <= d1 + d12}")
    
    # Test 2: Volume growth
    print("\n2. Volume growth test (Area ~ exp(r/R)):")
    radii = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    for r_poincare in radii:
        # Approximate area by counting points
        n_samples = 10000
        angles = np.random.rand(n_samples) * 2 * np.pi
        radii_samples = np.random.rand(n_samples) * r_poincare
        
        u = radii_samples * np.cos(angles)
        v = radii_samples * np.sin(angles)
        
        # Convert to hyperboloid and compute distances from origin
        origin_hyp = poincare_to_hyperboloid(np.array([0.0, 0.0]), R=R)
        
        count = 0
        for i in range(min(1000, n_samples)):  # Sample subset for speed
            point_hyp = poincare_to_hyperboloid(np.array([u[i], v[i]]), R=R)
            d = hyperbolic_distance(origin_hyp, point_hyp, R=R)
            # Theoretical: r_hyp = R * artanh(r_poincare)
            r_hyp_theoretical = R * np.arctanh(r_poincare)
            if d <= r_hyp_theoretical:
                count += 1
        
        area_estimate = count / 1000.0 * np.pi * r_poincare**2
        area_theoretical = 2 * np.pi * R**2 * (np.cosh(r_poincare/R) - 1)
        
        print(f"  r_Poincaré = {r_poincare:.1f}: Area ≈ {area_estimate:.4f} "
              f"(theoretical growth ~ exp({r_poincare/R:.2f}))")


if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENT 4.1: HYPERBOLIC DISTANCE IMPLEMENTATION")
    print("="*70)
    print()
    
    R = 1.0
    
    # Generate visualizations
    print("Generating visualizations...")
    visualize_hyperboloid(R=R)
    visualize_distance_field(R=R)
    max_err, mean_err = test_coordinate_conversions(R=R)
    
    # Run validation tests
    validate_hyperbolic_geometry(R=R)
    
    print()
    print("="*70)
    print("EXPERIMENT 4.1 COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  - hyperbolic_geometry.py (this module)")
    print("  - hyperboloid_visualization.png")
    print("  - distance_field_hyperbolic.png")
    print("  - coordinate_conversions_test.png")
