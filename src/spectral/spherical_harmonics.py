"""
Spherical Harmonics Implementation for Geometric Neural Operators
Implements forward/inverse transforms and validation tests
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from scipy.special import sph_harm
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SphericalHarmonics:
    """
    Spherical Harmonics basis functions for functions on S^2.
    
    Implements:
    - Y_l^m(theta, phi) computation
    - Forward transform: function → coefficients
    - Inverse transform: coefficients → function
    - Orthonormality validation
    """
    
    def __init__(self, L_max=10):
        """
        Initialize spherical harmonics up to degree L_max.
        
        Args:
            L_max: Maximum degree l
        """
        self.L_max = L_max
        self.num_coeffs = (L_max + 1) ** 2
        
    def Y(self, l, m, theta, phi):
        """
        Compute spherical harmonic Y_l^m(theta, phi).
        
        Args:
            l: Degree (0 <= l <= L_max)
            m: Order (-l <= m <= l)
            theta: Colatitude [0, pi]
            phi: Azimuth [0, 2pi]
            
        Returns:
            Complex-valued Y_l^m
        """
        # scipy uses (m, l, phi, theta) convention
        return sph_harm(m, l, phi, theta)
    
    def forward_transform(self, func, n_theta=100, n_phi=200):
        """
        Forward transform: function → coefficients.
        
        Computes: c_l^m = ∫∫ f(theta, phi) Y_l^m*(theta, phi) sin(theta) dtheta dphi
        
        Args:
            func: Function f(theta, phi) to decompose
            n_theta: Number of theta samples
            n_phi: Number of phi samples
            
        Returns:
            coeffs: Dictionary {(l, m): c_l^m}
        """
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
        
        # Evaluate function on grid
        f_vals = func(Theta, Phi)
        
        coeffs = {}
        for l in range(self.L_max + 1):
            for m in range(-l, l + 1):
                # Compute Y_l^m on grid
                Y_lm = self.Y(l, m, Theta, Phi)
                
                # Numerical integration using trapezoidal rule
                # ∫∫ f * Y* sin(theta) dtheta dphi
                integrand = f_vals * np.conj(Y_lm) * np.sin(Theta)
                
                # Integrate over theta and phi
                c_lm = np.trapz(np.trapz(integrand, phi, axis=1), theta, axis=0)
                coeffs[(l, m)] = c_lm
                
        return coeffs
    
    def inverse_transform(self, coeffs, theta, phi):
        """
        Inverse transform: coefficients → function.
        
        Reconstructs: f(theta, phi) = Σ c_l^m Y_l^m(theta, phi)
        
        Args:
            coeffs: Dictionary {(l, m): c_l^m}
            theta: Colatitude points
            phi: Azimuth points
            
        Returns:
            f_reconstructed: Function values at (theta, phi)
        """
        # Ensure theta and phi are arrays
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        
        # Create meshgrid if needed
        if theta.ndim == 1 and phi.ndim == 1:
            Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
        else:
            Theta, Phi = theta, phi
            
        f_recon = np.zeros_like(Theta, dtype=complex)
        
        for (l, m), c_lm in coeffs.items():
            Y_lm = self.Y(l, m, Theta, Phi)
            f_recon += c_lm * Y_lm
            
        return np.real(f_recon)
    
    def validate_orthonormality(self, l_max=5, n_theta=100, n_phi=200):
        """
        Validate orthonormality: ∫ Y_l^m * Y_l'^m' dΩ = δ_ll' δ_mm'
        
        Args:
            l_max: Maximum degree to test
            n_theta: Theta grid points
            n_phi: Phi grid points
            
        Returns:
            max_error: Maximum deviation from orthonormality
        """
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
        
        max_error = 0.0
        
        for l1 in range(l_max + 1):
            for m1 in range(-l1, l1 + 1):
                Y1 = self.Y(l1, m1, Theta, Phi)
                
                for l2 in range(l_max + 1):
                    for m2 in range(-l2, l2 + 1):
                        Y2 = self.Y(l2, m2, Theta, Phi)
                        
                        # Compute inner product
                        integrand = Y1 * np.conj(Y2) * np.sin(Theta)
                        inner_prod = np.trapz(np.trapz(integrand, phi, axis=1), theta, axis=0)
                        
                        # Expected value
                        expected = 1.0 if (l1 == l2 and m1 == m2) else 0.0
                        
                        error = np.abs(inner_prod - expected)
                        max_error = max(max_error, error)
                        
        return max_error


def test_constant_function():
    """Test 1: Constant function u(x) = 1"""
    sh = SphericalHarmonics(L_max=10)
    
    def constant(theta, phi):
        return np.ones_like(theta)
    
    # Forward transform
    coeffs = sh.forward_transform(constant)
    
    # Analytical: only c_0^0 = sqrt(4π) should be non-zero
    analytical_c00 = np.sqrt(4 * np.pi)
    numerical_c00 = coeffs[(0, 0)]
    
    print(f"Constant Function Test:")
    print(f"  Analytical c_0^0: {analytical_c00:.6f}")
    print(f"  Numerical c_0^0:  {numerical_c00:.6f}")
    print(f"  Error: {np.abs(numerical_c00 - analytical_c00):.2e}")
    
    # Check other coefficients are near zero
    max_other = max(np.abs(coeffs[(l, m)]) for (l, m) in coeffs if (l, m) != (0, 0))
    print(f"  Max other coefficient: {max_other:.2e}\n")
    
    return coeffs


def test_dipole_function():
    """Test 2: Dipole u(x) = cos(theta)"""
    sh = SphericalHarmonics(L_max=10)
    
    def dipole(theta, phi):
        return np.cos(theta)
    
    coeffs = sh.forward_transform(dipole)
    
    # Analytical: cos(theta) = sqrt(4π/3) Y_1^0
    analytical_c10 = np.sqrt(4 * np.pi / 3)
    numerical_c10 = coeffs[(1, 0)]
    
    print(f"Dipole Function Test:")
    print(f"  Analytical c_1^0: {analytical_c10:.6f}")
    print(f"  Numerical c_1^0:  {numerical_c10:.6f}")
    print(f"  Error: {np.abs(numerical_c10 - analytical_c10):.2e}\n")
    
    return coeffs


def test_quadrupole_function():
    """Test 3: Quadrupole u(x) = 3cos²(theta) - 1"""
    sh = SphericalHarmonics(L_max=10)
    
    def quadrupole(theta, phi):
        return 3 * np.cos(theta)**2 - 1
    
    coeffs = sh.forward_transform(quadrupole)
    
    print(f"Quadrupole Function Test:")
    print(f"  Numerical c_2^0: {coeffs[(2, 0)]:.6f}")
    print(f"  Numerical c_0^0: {coeffs[(0, 0)]:.6f}\n")
    
    return coeffs


def test_gaussian_bump():
    """Test 4: Gaussian bump centered at north pole"""
    sh = SphericalHarmonics(L_max=10)
    
    sigma = 0.3
    
    def gaussian(theta, phi):
        # Distance from north pole (theta=0)
        return np.exp(-theta**2 / (2 * sigma**2))
    
    coeffs = sh.forward_transform(gaussian)
    
    print(f"Gaussian Bump Test:")
    print(f"  c_0^0: {coeffs[(0, 0)]:.6f}")
    print(f"  c_1^0: {coeffs[(1, 0)]:.6f}")
    print(f"  c_2^0: {coeffs[(2, 0)]:.6f}\n")
    
    return coeffs


def plot_spherical_harmonics(L_max=3, save_path='validation_plots.png'):
    """Plot first 16 spherical harmonics on sphere (4x4 grid)"""
    sh = SphericalHarmonics(L_max=L_max)
    
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    Theta, Phi = np.meshgrid(theta, phi)
    
    # Convert to Cartesian for plotting
    X = np.sin(Theta) * np.cos(Phi)
    Y = np.sin(Theta) * np.sin(Phi)
    Z = np.cos(Theta)
    
    fig = plt.figure(figsize=(16, 16))
    
    plot_idx = 1
    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            if plot_idx > 16:
                break
                
            ax = fig.add_subplot(4, 4, plot_idx, projection='3d')
            
            # Compute Y_l^m
            Y_lm = np.real(sh.Y(l, m, Theta, Phi))
            
            # Plot on sphere
            surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.seismic(
                (Y_lm - Y_lm.min()) / (Y_lm.max() - Y_lm.min() + 1e-10)
            ), alpha=0.9)
            
            ax.set_title(f'$Y_{{{l}}}^{{{m}}}$', fontsize=12)
            ax.set_box_aspect([1,1,1])
            ax.axis('off')
            
            plot_idx += 1
            
        if plot_idx > 16:
            break
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved spherical harmonics plot to {save_path}")
    plt.close()


def convergence_analysis(save_csv='convergence_analysis.csv'):
    """Analyze reconstruction error vs number of modes"""
    sh_full = SphericalHarmonics(L_max=15)
    
    # Test function: Gaussian bump
    sigma = 0.3
    def test_func(theta, phi):
        return np.exp(-theta**2 / (2 * sigma**2))
    
    # Evaluation grid
    theta_eval = np.linspace(0, np.pi, 50)
    phi_eval = np.linspace(0, 2*np.pi, 100)
    Theta_eval, Phi_eval = np.meshgrid(theta_eval, phi_eval, indexing='ij')
    f_true = test_func(Theta_eval, Phi_eval)
    
    # Compute full coefficients
    coeffs_full = sh_full.forward_transform(test_func)
    
    L_values = range(1, 16)
    errors = []
    num_modes = []
    
    for L in L_values:
        # Truncate coefficients
        coeffs_trunc = {(l, m): c for (l, m), c in coeffs_full.items() if l <= L}
        
        # Reconstruct
        f_recon = sh_full.inverse_transform(coeffs_trunc, Theta_eval, Phi_eval)
        
        # Compute L2 error
        error = np.sqrt(np.mean((f_recon - f_true)**2))
        errors.append(error)
        num_modes.append((L + 1)**2)
        
    # Save to CSV
    with open(save_csv, 'w') as f:
        f.write("L_max,num_modes,l2_error\n")
        for L, n, e in zip(L_values, num_modes, errors):
            f.write(f"{L},{n},{e:.6e}\n")
    
    print(f"Saved convergence analysis to {save_csv}")
    
    return L_values, errors


def generate_validation_plots():
    """Generate comprehensive validation plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Orthonormality validation
    sh = SphericalHarmonics(L_max=10)
    l_values = range(6)
    ortho_errors = []
    
    for l_max in l_values:
        error = sh.validate_orthonormality(l_max=l_max, n_theta=50, n_phi=100)
        ortho_errors.append(error)
    
    axes[0, 0].semilogy(l_values, ortho_errors, 'o-', linewidth=2)
    axes[0, 0].set_xlabel('$L_{max}$', fontsize=12)
    axes[0, 0].set_ylabel('Max Orthonormality Error', fontsize=12)
    axes[0, 0].set_title('Orthonormality Validation', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Convergence analysis
    L_values, errors = convergence_analysis()
    axes[0, 1].loglog([l**2 for l in L_values], errors, 'o-', linewidth=2, label='Gaussian bump')
    axes[0, 1].set_xlabel('Number of Modes', fontsize=12)
    axes[0, 1].set_ylabel('$L_2$ Error', fontsize=12)
    axes[0, 1].set_title('Reconstruction Convergence', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Coefficient comparison (Constant function)
    coeffs_const = test_constant_function()
    l_vals = []
    c_vals = []
    for (l, m), c in sorted(coeffs_const.items())[:20]:
        l_vals.append(l + m/10)  # Offset for visualization
        c_vals.append(np.abs(c))
    
    axes[1, 0].semilogy(l_vals, c_vals, 'o', markersize=6)
    axes[1, 0].set_xlabel('Degree $l$ (with order offset)', fontsize=12)
    axes[1, 0].set_ylabel('$|c_l^m|$', fontsize=12)
    axes[1, 0].set_title('Coefficients: Constant Function', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Coefficient comparison (Dipole)
    coeffs_dipole = test_dipole_function()
    l_vals_d = []
    c_vals_d = []
    for (l, m), c in sorted(coeffs_dipole.items())[:20]:
        l_vals_d.append(l + m/10)
        c_vals_d.append(np.abs(c))
    
    axes[1, 1].semilogy(l_vals_d, c_vals_d, 'o', markersize=6, color='C1')
    axes[1, 1].set_xlabel('Degree $l$ (with order offset)', fontsize=12)
    axes[1, 1].set_ylabel('$|c_l^m|$', fontsize=12)
    axes[1, 1].set_title('Coefficients: Dipole Function', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('validation_plots.png', dpi=200, bbox_inches='tight')
    print("Saved validation plots to validation_plots.png")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("PHASE 1: SPHERICAL HARMONICS IMPLEMENTATION & VALIDATION")
    print("="*70)
    print()
    
    # Test functions
    print("Testing Analytical Functions:")
    print("-" * 70)
    test_constant_function()
    test_dipole_function()
    test_quadrupole_function()
    test_gaussian_bump()
    
    # Validate orthonormality
    print("Validating Orthonormality:")
    print("-" * 70)
    sh = SphericalHarmonics(L_max=10)
    max_error = sh.validate_orthonormality(l_max=5)
    print(f"Maximum orthonormality error (L_max=5): {max_error:.2e}\n")
    
    # Generate plots
    print("Generating Visualizations:")
    print("-" * 70)
    plot_spherical_harmonics(L_max=3, save_path='spherical_harmonics_basis.png')
    generate_validation_plots()
    
    print()
    print("="*70)
    print("PHASE 1 COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  - spherical_harmonics.py (this module)")
    print("  - spherical_harmonics_basis.png (4x4 grid of Y_l^m)")
    print("  - validation_plots.png (4 validation subplots)")
    print("  - convergence_analysis.csv (error vs modes)")
