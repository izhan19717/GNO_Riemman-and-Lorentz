"""
Improved Figure 1: Framework with Real Geometric Visualizations
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Wedge
import matplotlib.patches as mpatches

# Publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 300

COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'red': '#CC78BC',
    'green': '#029E73'
}


def create_improved_figure1(save_path='figure1_framework_improved.pdf'):
    """Create improved framework figure with real visualizations."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.05, right=0.95, top=0.93, bottom=0.05)
    
    # ============ ROW 1: GEOMETRIES ============
    
    # Sphere (3D)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, alpha=0.3, color=COLORS['blue'], edgecolor='none')
    
    # Add some geodesics
    for angle in [0, np.pi/3, 2*np.pi/3]:
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(np.cos(theta)*np.cos(angle), np.cos(theta)*np.sin(angle), 
                np.sin(theta), 'r-', linewidth=2, alpha=0.7)
    
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)
    ax1.set_box_aspect([1,1,1])
    ax1.axis('off')
    ax1.set_title('Sphere $S^2$ (K > 0)\nRiemannian', fontsize=11, fontweight='bold', pad=10)
    ax1.text2D(0.5, -0.05, 'SO(3) Equivariance', transform=ax1.transAxes,
              ha='center', fontsize=9, style='italic')
    
    # Hyperbolic (Poincaré disk)
    ax2 = fig.add_subplot(gs[0, 1])
    circle = Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax2.add_patch(circle)
    
    # Geodesics (straight lines through origin)
    for angle in np.linspace(0, np.pi, 8, endpoint=False):
        t = np.linspace(-0.95, 0.95, 50)
        x = t * np.cos(angle)
        y = t * np.sin(angle)
        ax2.plot(x, y, color=COLORS['red'], linewidth=1.5, alpha=0.6)
    
    # Hyperbolic circles
    for r in [0.3, 0.6, 0.9]:
        circle_h = Circle((0, 0), r, fill=False, edgecolor=COLORS['red'], 
                         linewidth=1, alpha=0.4, linestyle='--')
        ax2.add_patch(circle_h)
    
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Hyperbolic $H^2$ (K < 0)\nPoincaré Disk', fontsize=11, fontweight='bold')
    ax2.text(0, -1.4, 'Möbius Invariance', ha='center', fontsize=9, style='italic')
    
    # Minkowski (spacetime diagram)
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Light cone
    x_range = np.linspace(-1, 1, 100)
    ax3.fill_between(x_range, -np.abs(x_range), np.abs(x_range), 
                     alpha=0.2, color=COLORS['orange'], label='Causal Region')
    ax3.plot(x_range, x_range, 'k-', linewidth=2.5, label='Light Cone')
    ax3.plot(x_range, -x_range, 'k-', linewidth=2.5)
    
    # Event
    ax3.plot(0, 0, 'ro', markersize=10, zorder=5)
    
    # Worldlines
    for x0 in [-0.5, 0.3]:
        t = np.linspace(-1, 1, 50)
        ax3.plot([x0]*len(t), t, '--', color=COLORS['green'], linewidth=1.5, alpha=0.7)
    
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_aspect('equal')
    ax3.set_xlabel('Space (x)', fontsize=9)
    ax3.set_ylabel('Time (t)', fontsize=9)
    ax3.set_title('Minkowski (Lorentzian)\nSpacetime', fontsize=11, fontweight='bold')
    ax3.text(0, -1.4, 'Causality Preservation', ha='center', fontsize=9, style='italic')
    ax3.grid(True, alpha=0.3)
    
    # ============ ROW 2: BRANCH NETWORKS ============
    
    branch_configs = [
        ('Spectral Branch', ['Spherical Harmonics', 'Y_lm coefficients', '[36] features', 
                            'Dense [128]', 'Dense [128]', 'Latent [64]'], COLORS['blue']),
        ('Graph Branch', ['Function values', 'at 200 points', 'Graph encoding',
                         'Dense [128]', 'Dense [128]', 'Latent [64]'], COLORS['red']),
        ('Fourier Branch', ['Initial conditions', 'FFT [10 modes]', 'Fourier coeffs',
                           'Dense [128]', 'Dense [128]', 'Latent [64]'], COLORS['orange'])
    ]
    
    for i, (title, layers, color) in enumerate(branch_configs):
        ax = fig.add_subplot(gs[1, i])
        ax.text(0.5, 0.95, title, ha='center', fontsize=10, fontweight='bold',
               transform=ax.transAxes)
        
        y_start = 0.85
        y_step = 0.13
        
        for j, layer in enumerate(layers):
            y = y_start - j * y_step
            width = 0.7 - j * 0.05 if j < 3 else 0.55
            
            if j < 3:  # Input layers
                rect = FancyBboxPatch((0.5 - width/2, y - 0.05), width, 0.08,
                                     boxstyle="round,pad=0.01",
                                     edgecolor=color, facecolor='lightyellow',
                                     linewidth=1.5, transform=ax.transAxes)
            else:  # Network layers
                rect = FancyBboxPatch((0.5 - width/2, y - 0.05), width, 0.08,
                                     boxstyle="round,pad=0.01",
                                     edgecolor=color, facecolor='white',
                                     linewidth=1.5, transform=ax.transAxes)
            
            ax.add_patch(rect)
            ax.text(0.5, y, layer, ha='center', va='center', fontsize=7.5,
                   transform=ax.transAxes)
            
            # Arrow
            if j < len(layers) - 1:
                ax.annotate('', xy=(0.5, y - 0.08), xytext=(0.5, y - 0.05),
                           xycoords='axes fraction', textcoords='axes fraction',
                           arrowprops=dict(arrowstyle='->', lw=1.5, color=color))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # ============ ROW 3: TRUNK NETWORKS ============
    
    trunk_configs = [
        ('Geometric Trunk', 
         ['Query (θ, φ)', 'Geodesic distances\nto 10 refs', 'SH features',
          'Dense [128]', 'Dense [128]', 'Output [64]'], COLORS['blue']),
        ('Hyperbolic Trunk',
         ['Query (u, v)', 'Hyperbolic dist\nto 10 refs', 'Depth + curvature',
          'Dense [128]', 'Dense [128]', 'Output [64]'], COLORS['red']),
        ('Causal Trunk',
         ['Query (t, x)', 'Light cone coords', 'Proper time',
          'Dense [128]', 'Dense [128]', 'Output [64]'], COLORS['orange'])
    ]
    
    for i, (title, layers, color) in enumerate(trunk_configs):
        ax = fig.add_subplot(gs[2, i])
        ax.text(0.5, 0.95, title, ha='center', fontsize=10, fontweight='bold',
               transform=ax.transAxes)
        
        y_start = 0.85
        y_step = 0.13
        
        for j, layer in enumerate(layers):
            y = y_start - j * y_step
            width = 0.7 - j * 0.05 if j < 3 else 0.55
            
            if j < 3:  # Feature layers
                rect = FancyBboxPatch((0.5 - width/2, y - 0.05), width, 0.08,
                                     boxstyle="round,pad=0.01",
                                     edgecolor=color, facecolor='lightgreen',
                                     linewidth=1.5, transform=ax.transAxes)
            else:  # Network layers
                rect = FancyBboxPatch((0.5 - width/2, y - 0.05), width, 0.08,
                                     boxstyle="round,pad=0.01",
                                     edgecolor=color, facecolor='white',
                                     linewidth=1.5, transform=ax.transAxes)
            
            ax.add_patch(rect)
            ax.text(0.5, y, layer, ha='center', va='center', fontsize=7.5,
                   transform=ax.transAxes)
            
            # Arrow
            if j < len(layers) - 1:
                ax.annotate('', xy=(0.5, y - 0.08), xytext=(0.5, y - 0.05),
                           xycoords='axes fraction', textcoords='axes fraction',
                           arrowprops=dict(arrowstyle='->', lw=1.5, color=color))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Main title
    fig.suptitle('Geometric Neural Operators: Framework Across Geometries',
                fontsize=14, fontweight='bold', y=0.97)
    
    # Add legend for output
    fig.text(0.5, 0.015, 'Output: u(y) = ⟨Branch(f), Trunk(y)⟩  (Inner product in latent space)',
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved improved {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Creating improved Figure 1...")
    create_improved_figure1()
    print("Done!")
