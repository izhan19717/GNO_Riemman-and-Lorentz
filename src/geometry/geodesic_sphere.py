"""
Geodesic Distance on Sphere (S^2)
Implements great circle distance computation with numerical stability
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def geodesic_distance_sphere(x, y, R=1.0, input_type='cartesian'):
    """
    Compute geodesic (great circle) distance on S^2.
    
    Args:
        x: Point(s) on sphere
        y: Point(s) on sphere
        R: Radius of sphere
        input_type: 'cartesian' for (x,y,z) or 'spherical' for (theta, phi)
        
    Returns:
        d: Geodesic distance d_S2(x,y) = R * arccos(x·y/R²)
    """
    if input_type == 'spherical':
        # Convert (theta, phi) to Cartesian
        x = spherical_to_cartesian(x, R)
        y = spherical_to_cartesian(y, R)
    
    # Ensure x and y are arrays
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    
    # Normalize to ensure points are on sphere of radius R
    x_norm = x / np.linalg.norm(x, axis=-1, keepdims=True) * R
    y_norm = y / np.linalg.norm(y, axis=-1, keepdims=True) * R
    
    # Compute dot product
    dot_prod = np.sum(x_norm * y_norm, axis=-1)
    
    # Numerical stability: clamp to [-R², R²]
    # Then normalize by R² to get cos(theta)
    cos_theta = dot_prod / (R ** 2)
    
    # Critical: clamp to [-1+eps, 1-eps] to avoid NaN in arccos gradient
    eps = 1e-7
    cos_theta = np.clip(cos_theta, -1.0 + eps, 1.0 - eps)
    
    # Geodesic distance
    d = R * np.arccos(cos_theta)
    
    return d


def spherical_to_cartesian(coords, R=1.0):
    """
    Convert spherical (theta, phi) to Cartesian (x, y, z).
    
    Args:
        coords: [..., 2] array of (theta, phi)
        R: Radius
        
    Returns:
        xyz: [..., 3] array of (x, y, z)
    """
    coords = np.atleast_2d(coords)
    theta = coords[..., 0]
    phi = coords[..., 1]
    
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    
    return np.stack([x, y, z], axis=-1)


def cartesian_to_spherical(xyz):
    """
    Convert Cartesian (x, y, z) to spherical (theta, phi).
    
    Args:
        xyz: [..., 3] array of (x, y, z)
        
    Returns:
        coords: [..., 2] array of (theta, phi)
    """
    xyz = np.atleast_2d(xyz)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    
    r = np.linalg.norm(xyz, axis=-1)
    theta = np.arccos(np.clip(z / (r + 1e-10), -1, 1))
    phi = np.arctan2(y, x)
    
    return np.stack([theta, phi], axis=-1)


def validate_geodesic_distance():
    """
    Validation tests for geodesic distance computation.
    """
    R = 1.0
    
    print("="*70)
    print("GEODESIC DISTANCE VALIDATION TESTS")
    print("="*70)
    print()
    
    # Test 1: North pole to south pole
    print("Test 1: North Pole to South Pole")
    north = np.array([0, 0, R])
    south = np.array([0, 0, -R])
    d_ns = geodesic_distance_sphere(north, south, R)
    d_ns_val = float(d_ns) if isinstance(d_ns, np.ndarray) else d_ns
    expected_ns = np.pi * R
    print(f"  Computed: {d_ns_val:.6f}")
    print(f"  Expected: {expected_ns:.6f}")
    print(f"  Error: {np.abs(d_ns_val - expected_ns):.2e}\n")
    
    # Test 2: Antipodal points (random)
    print("Test 2: Random Antipodal Points")
    theta = np.pi / 3
    phi = np.pi / 4
    p1 = spherical_to_cartesian(np.array([theta, phi]), R)
    p2 = -p1  # Antipodal
    d_antipodal = geodesic_distance_sphere(p1, p2, R)
    d_antipodal_val = float(d_antipodal) if isinstance(d_antipodal, np.ndarray) else d_antipodal
    expected_antipodal = np.pi * R
    print(f"  Computed: {d_antipodal_val:.6f}")
    print(f"  Expected: {expected_antipodal:.6f}")
    print(f"  Error: {np.abs(d_antipodal_val - expected_antipodal):.2e}\n")
    
    # Test 3: Points on equator
    print("Test 3: Points on Equator")
    alpha = np.pi / 6  # 30 degrees
    p1_eq = spherical_to_cartesian(np.array([np.pi/2, 0]), R)
    p2_eq = spherical_to_cartesian(np.array([np.pi/2, alpha]), R)
    d_equator = geodesic_distance_sphere(p1_eq, p2_eq, R)
    d_equator_val = float(d_equator) if isinstance(d_equator, np.ndarray) else d_equator
    expected_equator = R * alpha
    print(f"  Angle: {np.degrees(alpha):.2f}°")
    print(f"  Computed: {d_equator_val:.6f}")
    print(f"  Expected: {expected_equator:.6f}")
    print(f"  Error: {np.abs(d_equator_val - expected_equator):.2e}\n")
    
    # Test 4: Same point
    print("Test 4: Same Point (Numerical Stability)")
    p_same = spherical_to_cartesian(np.array([np.pi/4, np.pi/3]), R)
    d_same = geodesic_distance_sphere(p_same, p_same, R)
    d_same_val = float(d_same) if isinstance(d_same, np.ndarray) else d_same
    print(f"  Computed: {d_same_val:.2e}")
    print(f"  Expected: 0.0")
    print(f"  Error: {d_same_val:.2e}\n")


def visualize_distance_field(save_path='distance_field_visualization.png'):
    """
    Visualize geodesic distance field on sphere.
    """
    R = 1.0
    
    # Create sphere mesh
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    Theta, Phi = np.meshgrid(theta, phi)
    
    X = R * np.sin(Theta) * np.cos(Phi)
    Y = R * np.sin(Theta) * np.sin(Phi)
    Z = R * np.cos(Theta)
    
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Single reference point
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Reference point
    x0_spherical = np.array([np.pi/4, np.pi/4])
    x0 = spherical_to_cartesian(x0_spherical, R).flatten()  # Ensure 1D array
    
    # Compute distance field
    points_cart = spherical_to_cartesian(np.stack([Theta, Phi], axis=-1), R)
    distances = np.zeros_like(Theta)
    
    for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            p = points_cart[i, j]
            distances[i, j] = geodesic_distance_sphere(x0, p, R)
    
    # Normalize for color mapping
    norm_dist = (distances - distances.min()) / (distances.max() - distances.min() + 1e-10)
    
    # Plot sphere with distance field
    surf1 = ax1.plot_surface(X, Y, Z, facecolors=cm.viridis(norm_dist), alpha=0.9)
    
    # Mark reference point
    ax1.scatter(x0[0], x0[1], x0[2], color='red', s=100, marker='*', label='Reference')
    
    ax1.set_title('Distance Field: Single Reference', fontsize=12)
    ax1.set_box_aspect([1,1,1])
    ax1.axis('off')
    
    # Plot 2: Multiple reference points
    ax2 = fig.add_subplot(132, projection='3d')
    
    # 5 random reference points
    np.random.seed(42)
    n_refs = 5
    ref_points_spherical = np.random.rand(n_refs, 2) * np.array([np.pi, 2*np.pi])
    ref_points = spherical_to_cartesian(ref_points_spherical, R)
    
    # Compute minimum distance to any reference
    min_distances = np.full_like(Theta, np.inf)
    
    for ref in ref_points:
        for i in range(Theta.shape[0]):
            for j in range(Theta.shape[1]):
                p = points_cart[i, j]
                d = geodesic_distance_sphere(ref, p, R)
                min_distances[i, j] = min(min_distances[i, j], d)
    
    # Normalize
    norm_min_dist = (min_distances - min_distances.min()) / (min_distances.max() - min_distances.min() + 1e-10)
    
    surf2 = ax2.plot_surface(X, Y, Z, facecolors=cm.plasma(norm_min_dist), alpha=0.9)
    
    # Mark reference points
    ax2.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], 
                color='red', s=100, marker='*', label='References')
    
    ax2.set_title('Distance Field: 5 References', fontsize=12)
    ax2.set_box_aspect([1,1,1])
    ax2.axis('off')
    
    # Plot 3: Distance gradient visualization
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Use single reference again but show gradient
    gradient_colors = cm.coolwarm(norm_dist)
    surf3 = ax3.plot_surface(X, Y, Z, facecolors=gradient_colors, alpha=0.9)
    ax3.scatter(x0[0], x0[1], x0[2], color='yellow', s=150, marker='*', edgecolors='black', linewidths=2)
    
    ax3.set_title('Distance Gradient (Coolwarm)', fontsize=12)
    ax3.set_box_aspect([1,1,1])
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved distance field visualization to {save_path}")
    plt.close()


def generate_comparison_dataset(n_samples=1000, save_path='geodesic_test_data.npz'):
    """
    Generate dataset comparing geodesic vs Euclidean distances.
    """
    R = 1.0
    
    # Generate random point pairs on sphere
    np.random.seed(42)
    
    # Random spherical coordinates
    theta1 = np.random.rand(n_samples) * np.pi
    phi1 = np.random.rand(n_samples) * 2 * np.pi
    theta2 = np.random.rand(n_samples) * np.pi
    phi2 = np.random.rand(n_samples) * 2 * np.pi
    
    points1_spherical = np.stack([theta1, phi1], axis=-1)
    points2_spherical = np.stack([theta2, phi2], axis=-1)
    
    # Convert to Cartesian
    points1 = spherical_to_cartesian(points1_spherical, R)
    points2 = spherical_to_cartesian(points2_spherical, R)
    
    # Compute geodesic distances
    geodesic_dists = np.array([
        geodesic_distance_sphere(p1, p2, R) 
        for p1, p2 in zip(points1, points2)
    ])
    
    # Compute Euclidean distances in embedding space
    euclidean_dists = np.linalg.norm(points1 - points2, axis=-1)
    
    # Save dataset
    np.savez(save_path, 
             points1=points1,
             points2=points2,
             geodesic_distances=geodesic_dists,
             euclidean_distances=euclidean_dists)
    
    print(f"Saved test dataset to {save_path}")
    
    return geodesic_dists, euclidean_dists


def plot_distance_comparison(geodesic_dists, euclidean_dists, 
                             save_path='distance_comparison.png'):
    """
    Plot comparison between geodesic and Euclidean distances.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Scatter plot
    axes[0].scatter(euclidean_dists, geodesic_dists, alpha=0.5, s=10)
    axes[0].plot([0, 2], [0, np.pi], 'r--', label='Max geodesic (π)')
    axes[0].set_xlabel('Euclidean Distance', fontsize=12)
    axes[0].set_ylabel('Geodesic Distance', fontsize=12)
    axes[0].set_title('Geodesic vs Euclidean Distance', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Histogram of differences
    differences = geodesic_dists - euclidean_dists
    axes[1].hist(differences, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(differences), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(differences):.3f}')
    axes[1].set_xlabel('Geodesic - Euclidean', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Distance Difference Distribution', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Ratio analysis
    # Ensure both arrays are 1D before division to avoid broadcasting issues
    geo_arr = np.asarray(geodesic_dists).ravel()
    euc_arr = np.asarray(euclidean_dists).ravel()
    ratios_arr = geo_arr / (euc_arr + 1e-10)
    
    axes[2].scatter(euc_arr, ratios_arr, alpha=0.5, s=10, c=geo_arr, cmap='viridis')
    axes[2].axhline(1.0, color='red', linestyle='--', label='Ratio = 1')
    axes[2].set_xlabel('Euclidean Distance', fontsize=12)
    axes[2].set_ylabel('Geodesic / Euclidean', fontsize=12)
    axes[2].set_title('Distance Ratio vs Euclidean', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved distance comparison plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENT 1.2: GEODESIC DISTANCE ON SPHERE")
    print("="*70)
    print()
    
    # Validation tests
    validate_geodesic_distance()
    
    # Visualizations
    print("Generating Visualizations:")
    print("-" * 70)
    visualize_distance_field()
    
    # Generate comparison dataset
    print("\nGenerating Comparison Dataset:")
    print("-" * 70)
    geodesic_dists, euclidean_dists = generate_comparison_dataset()
    
    # Statistics
    print(f"\nDataset Statistics (n=1000):")
    print(f"  Geodesic distances: mean={np.mean(geodesic_dists):.4f}, std={np.std(geodesic_dists):.4f}")
    print(f"  Euclidean distances: mean={np.mean(euclidean_dists):.4f}, std={np.std(euclidean_dists):.4f}")
    print(f"  Difference: mean={np.mean(geodesic_dists - euclidean_dists):.4f}")
    
    # Comparison plot
    plot_distance_comparison(geodesic_dists, euclidean_dists)
    
    print()
    print("="*70)
    print("EXPERIMENT 1.2 COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  - geodesic_sphere.py (this module)")
    print("  - distance_field_visualization.png")
    print("  - distance_comparison.png")
    print("  - geodesic_test_data.npz")
