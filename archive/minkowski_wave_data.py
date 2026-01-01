"""
Wave Equation in 1+1 Minkowski Space
Implements d'Alembert solution with causal structure visualization
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm


def generate_initial_conditions(x, n_modes=5):
    """
    Generate random smooth initial conditions.
    u₀(x) = Σ aₙ sin(2πnx/L)
    v₀(x) = Σ bₙ sin(2πnx/L)
    """
    L = x[-1] - x[0]
    
    u0 = np.zeros_like(x)
    v0 = np.zeros_like(x)
    
    for n in range(1, n_modes + 1):
        # Coefficients decay as 1/n²
        a_n = np.random.randn() / (n ** 2)
        b_n = np.random.randn() / (n ** 2)
        
        u0 += a_n * np.sin(2 * np.pi * n * x / L)
        v0 += b_n * np.sin(2 * np.pi * n * x / L)
    
    return u0, v0


def dalembert_solution(u0, v0, x, t, L):
    """
    d'Alembert solution for 1+1 wave equation with periodic boundaries.
    u(t,x) = [u₀(x-t) + u₀(x+t)]/2 + (1/2)∫[x-t to x+t] v₀(s) ds
    """
    # Periodic extension
    def periodic_interp(values, positions):
        """Interpolate with periodic boundary conditions."""
        # Wrap positions to [0, L]
        positions_wrapped = positions % L
        
        # Linear interpolation
        result = np.interp(positions_wrapped, x, values, period=L)
        return result
    
    # u₀(x-t) + u₀(x+t)
    u_left = periodic_interp(u0, x - t)
    u_right = periodic_interp(u0, x + t)
    
    # Integral term: ∫[x-t to x+t] v₀(s) ds
    # Approximate using trapezoidal rule
    integral = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        # Integration bounds
        s_min = xi - t
        s_max = xi + t
        
        # Sample points for integration
        n_samples = 50
        s_vals = np.linspace(s_min, s_max, n_samples)
        v_vals = periodic_interp(v0, s_vals)
        
        # Trapezoidal integration
        integral[i] = np.trapz(v_vals, s_vals)
    
    u = (u_left + u_right) / 2 + integral / 2
    
    return u


def compute_causal_features(t, x, t0, x0):
    """
    Compute causal features for point (t, x) relative to event (t0, x0).
    
    Returns:
    - proper_time: τ = √|Δt² - Δx²|
    - light_cone_coords: (u, v) = (t-x, t+x)
    - causal_indicator: sign(Δt² - Δx²)
    """
    dt = t - t0
    dx = x - x0
    
    # Minkowski interval
    interval = dt**2 - dx**2
    
    # Proper time (for timelike separated events)
    proper_time = np.sqrt(np.abs(interval))
    
    # Light cone coordinates
    u = t - x
    v = t + x
    
    # Causal indicator: +1 (timelike), 0 (lightlike), -1 (spacelike)
    causal_indicator = np.sign(interval)
    
    return proper_time, (u, v), causal_indicator


def is_in_past_light_cone(t, x, t0, x0):
    """Check if (t, x) is in past light cone of (t0, x0)."""
    return (t < t0) and (np.abs(x - x0) <= (t0 - t))


def generate_dataset(n_samples=1000, nx=100, nt=10, L=1.0, T=1.0):
    """
    Generate complete dataset for wave equation.
    
    Returns:
        initial_u0: (n_samples, nx) initial positions
        initial_v0: (n_samples, nx) initial velocities
        solutions: (n_samples, nt, nx) solutions at different times
        x: (nx,) spatial grid
        t: (nt,) temporal grid
    """
    x = np.linspace(0, L, nx)
    t = np.linspace(0.1, T, nt)
    
    initial_u0 = []
    initial_v0 = []
    solutions = []
    
    print(f"Generating {n_samples} wave equation solutions...")
    
    for i in range(n_samples):
        if i % 100 == 0:
            print(f"  Progress: {i}/{n_samples}")
        
        # Generate initial conditions
        u0, v0 = generate_initial_conditions(x, n_modes=5)
        
        initial_u0.append(u0)
        initial_v0.append(v0)
        
        # Compute solution at each time
        u_t = []
        for ti in t:
            u = dalembert_solution(u0, v0, x, ti, L)
            u_t.append(u)
        
        solutions.append(u_t)
    
    return (np.array(initial_u0), np.array(initial_v0), 
            np.array(solutions), x, t)


def plot_spacetime_diagram(u0, v0, x, t, L, save_path='spacetime_diagram.png'):
    """Create spacetime diagram showing light cones."""
    
    # Compute solution on fine grid
    nt_fine = 50
    t_fine = np.linspace(0, t[-1], nt_fine)
    
    solution = np.zeros((nt_fine, len(x)))
    for i, ti in enumerate(t_fine):
        solution[i] = dalembert_solution(u0, v0, x, ti, L)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Spacetime diagram with solution
    ax1 = axes[0]
    
    X, T = np.meshgrid(x, t_fine)
    im = ax1.pcolormesh(X, T, solution, cmap='RdBu_r', shading='auto')
    
    # Draw light cones from several events
    events = [(0.3, L/4), (0.5, L/2), (0.7, 3*L/4)]
    
    for t0, x0 in events:
        # Future light cone
        t_cone = np.linspace(t0, t[-1], 20)
        x_left = x0 - (t_cone - t0)
        x_right = x0 + (t_cone - t0)
        
        ax1.plot(x_left, t_cone, 'k--', linewidth=1.5, alpha=0.7)
        ax1.plot(x_right, t_cone, 'k--', linewidth=1.5, alpha=0.7)
        
        # Mark event
        ax1.plot(x0, t0, 'ko', markersize=8)
    
    ax1.set_xlabel('Position x', fontsize=12)
    ax1.set_ylabel('Time t', fontsize=12)
    ax1.set_title('Spacetime Diagram with Light Cones', fontsize=14)
    plt.colorbar(im, ax=ax1, label='u(t,x)')
    ax1.set_xlim(0, L)
    ax1.set_ylim(0, t[-1])
    
    # Plot 2: Solution at different times
    ax2 = axes[1]
    
    time_slices = [0, 0.3, 0.6, 0.9]
    colors = cm.viridis(np.linspace(0, 1, len(time_slices)))
    
    for ti, color in zip(time_slices, colors):
        if ti == 0:
            ax2.plot(x, u0, color=color, linewidth=2, label=f't={ti:.1f}')
        else:
            u_ti = dalembert_solution(u0, v0, x, ti, L)
            ax2.plot(x, u_ti, color=color, linewidth=2, label=f't={ti:.1f}')
    
    ax2.set_xlabel('Position x', fontsize=12)
    ax2.set_ylabel('u(t,x)', fontsize=12)
    ax2.set_title('Wave Propagation', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved spacetime diagram to {save_path}")
    plt.close()


def plot_causality_visualization(u0, v0, x, t, L, save_path='causality_visualization.png'):
    """Visualize past light cone and domain of dependence."""
    
    # Event of interest
    t0, x0 = 0.5, L/2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Past light cone
    ax1 = axes[0]
    
    # Create grid
    nt_grid = 50
    nx_grid = 100
    t_grid = np.linspace(0, 1.0, nt_grid)
    x_grid = np.linspace(0, L, nx_grid)
    
    X_grid, T_grid = np.meshgrid(x_grid, t_grid)
    
    # Compute causal indicator for each point
    causal_map = np.zeros_like(X_grid)
    in_past_cone = np.zeros_like(X_grid, dtype=bool)
    
    for i in range(nt_grid):
        for j in range(nx_grid):
            t_ij = T_grid[i, j]
            x_ij = X_grid[i, j]
            
            _, _, causal_ind = compute_causal_features(t_ij, x_ij, t0, x0)
            causal_map[i, j] = causal_ind
            
            in_past_cone[i, j] = is_in_past_light_cone(t_ij, x_ij, t0, x0)
    
    # Plot causal structure
    im1 = ax1.pcolormesh(X_grid, T_grid, in_past_cone.astype(float), 
                         cmap='Blues', alpha=0.5, shading='auto')
    
    # Draw light cone boundaries
    t_cone = np.linspace(0, t0, 50)
    x_left = x0 - (t0 - t_cone)
    x_right = x0 + (t0 - t_cone)
    
    ax1.plot(x_left, t_cone, 'r-', linewidth=2, label='Past light cone')
    ax1.plot(x_right, t_cone, 'r-', linewidth=2)
    
    # Mark event
    ax1.plot(x0, t0, 'ro', markersize=12, label=f'Event ({x0:.2f}, {t0:.2f})')
    
    # Fill past light cone
    vertices = np.array([[x0, 0]] + 
                       list(zip(x_left[::-1], t_cone[::-1])) + 
                       [[x0, t0]] +
                       list(zip(x_right, t_cone)) + 
                       [[x0, 0]])
    polygon = Polygon(vertices, alpha=0.3, facecolor='yellow', 
                     edgecolor='red', linewidth=2, label='Domain of dependence')
    ax1.add_patch(polygon)
    
    ax1.set_xlabel('Position x', fontsize=12)
    ax1.set_ylabel('Time t', fontsize=12)
    ax1.set_title('Past Light Cone and Domain of Dependence', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, L)
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Causal features
    ax2 = axes[1]
    
    # Compute proper time for points on a line
    x_line = np.linspace(0, L, 100)
    t_line = np.ones_like(x_line) * 0.3
    
    proper_times = []
    for xi, ti in zip(x_line, t_line):
        tau, _, _ = compute_causal_features(ti, xi, t0, x0)
        proper_times.append(tau)
    
    ax2.plot(x_line, proper_times, linewidth=2)
    ax2.axvline(x0, color='r', linestyle='--', label=f'Event position x₀={x0:.2f}')
    ax2.set_xlabel('Position x', fontsize=12)
    ax2.set_ylabel('Proper Time τ', fontsize=12)
    ax2.set_title(f'Proper Time from Event at t={t_line[0]:.1f}', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved causality visualization to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENT 2.1: WAVE EQUATION IN 1+1 MINKOWSKI SPACE")
    print("="*70)
    print()
    
    # Parameters
    L = 1.0  # Spatial domain
    T = 1.0  # Time domain
    nx = 100  # Spatial grid points
    nt = 10   # Time snapshots
    
    # Generate dataset
    print("Generating dataset...")
    initial_u0, initial_v0, solutions, x, t = generate_dataset(
        n_samples=1000, nx=nx, nt=nt, L=L, T=T
    )
    
    print(f"\nDataset shapes:")
    print(f"  Initial u0: {initial_u0.shape}")
    print(f"  Initial v0: {initial_v0.shape}")
    print(f"  Solutions: {solutions.shape}")
    
    # Split train/test
    n_train = 800
    
    # Save datasets
    print("\nSaving datasets...")
    np.savez('train_wave_minkowski.npz',
             initial_u0=initial_u0[:n_train],
             initial_v0=initial_v0[:n_train],
             solutions=solutions[:n_train],
             x=x,
             t=t)
    print("  Saved train_wave_minkowski.npz")
    
    np.savez('test_wave_minkowski.npz',
             initial_u0=initial_u0[n_train:],
             initial_v0=initial_v0[n_train:],
             solutions=solutions[n_train:],
             x=x,
             t=t)
    print("  Saved test_wave_minkowski.npz")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Use first example for visualization
    plot_spacetime_diagram(initial_u0[0], initial_v0[0], x, t, L)
    plot_causality_visualization(initial_u0[0], initial_v0[0], x, t, L)
    
    print()
    print("="*70)
    print("EXPERIMENT 2.1 COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  - minkowski_wave_data.py (this module)")
    print("  - spacetime_diagram.png")
    print("  - causality_visualization.png")
    print("  - train_wave_minkowski.npz (800 examples)")
    print("  - test_wave_minkowski.npz (200 examples)")
