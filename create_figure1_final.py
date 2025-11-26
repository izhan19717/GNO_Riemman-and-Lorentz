"""
Professional Publication-Quality Figure 1
Clean, clear framework visualization
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Wedge, Arc
import matplotlib.patches as mpatches

# Professional publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.titlesize': 14,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Clean color palette
COLORS = {
    'sphere': '#2E86AB',      # Blue
    'hyperbolic': '#A23B72',  # Purple
    'minkowski': '#F18F01',   # Orange
    'neutral': '#4A4A4A'      # Dark gray
}


def draw_network_layer(ax, x, y, width, height, text, color, fill=True):
    """Draw a single network layer box."""
    facecolor = 'white' if not fill else f'{color}20'
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.01",
                          edgecolor=color, facecolor=facecolor,
                          linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=9,
           transform=ax.transAxes, weight='bold' if fill else 'normal')


def create_professional_figure1(save_path='figure1_framework_final.pdf'):
    """Create professional publication-quality framework figure."""
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.3,
                  left=0.08, right=0.92, top=0.90, bottom=0.08)
    
    # ==================== ROW 1: GEOMETRY VISUALIZATIONS ====================
    
    # --- SPHERE ---
    ax_sphere = fig.add_subplot(gs[0, 0])
    
    # Draw sphere outline
    theta = np.linspace(0, 2*np.pi, 100)
    ax_sphere.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2.5)
    ax_sphere.fill(np.cos(theta), np.sin(theta), color=COLORS['sphere'], alpha=0.15)
    
    # Latitude lines
    for lat in [-0.6, 0, 0.6]:
        width = np.sqrt(1 - lat**2)
        theta_lat = np.linspace(0, 2*np.pi, 50)
        ax_sphere.plot(width * np.cos(theta_lat), lat + 0*theta_lat, 
                      color=COLORS['sphere'], linewidth=1, alpha=0.6)
    
    # Longitude lines
    for lon in np.linspace(0, np.pi, 6, endpoint=False):
        theta_lon = np.linspace(-np.pi/2, np.pi/2, 50)
        ax_sphere.plot(np.cos(theta_lon) * np.cos(lon), np.sin(theta_lon),
                      color=COLORS['sphere'], linewidth=1, alpha=0.6)
    
    ax_sphere.set_xlim(-1.3, 1.3)
    ax_sphere.set_ylim(-1.3, 1.3)
    ax_sphere.set_aspect('equal')
    ax_sphere.axis('off')
    ax_sphere.set_title('Sphere $\\mathbf{S^2}$\n$K > 0$ (Positive Curvature)', 
                       fontsize=11, weight='bold', color=COLORS['sphere'], pad=15)
    ax_sphere.text(0, -1.55, 'SO(3) Equivariance', ha='center', fontsize=9, 
                  style='italic', color=COLORS['sphere'])
    
    # --- HYPERBOLIC ---
    ax_hyp = fig.add_subplot(gs[0, 1])
    
    # Poincaré disk boundary
    circle = Circle((0, 0), 1, fill=True, facecolor=f'{COLORS["hyperbolic"]}15',
                   edgecolor='k', linewidth=2.5)
    ax_hyp.add_patch(circle)
    
    # Geodesics (diameters)
    for angle in np.linspace(0, 150, 4):
        angle_rad = np.radians(angle)
        ax_hyp.plot([-0.95*np.cos(angle_rad), 0.95*np.cos(angle_rad)],
                   [-0.95*np.sin(angle_rad), 0.95*np.sin(angle_rad)],
                   color=COLORS['hyperbolic'], linewidth=1.5, alpha=0.7)
    
    # Concentric circles
    for r in [0.4, 0.7]:
        circ = Circle((0, 0), r, fill=False, edgecolor=COLORS['hyperbolic'],
                     linewidth=1, alpha=0.5, linestyle='--')
        ax_hyp.add_patch(circ)
    
    ax_hyp.set_xlim(-1.3, 1.3)
    ax_hyp.set_ylim(-1.3, 1.3)
    ax_hyp.set_aspect('equal')
    ax_hyp.axis('off')
    ax_hyp.set_title('Hyperbolic $\\mathbf{H^2}$\n$K < 0$ (Negative Curvature)',
                    fontsize=11, weight='bold', color=COLORS['hyperbolic'])
    ax_hyp.text(0, -1.5, 'Möbius Invariance', ha='center', fontsize=9,
               style='italic', color=COLORS['hyperbolic'])
    
    # --- MINKOWSKI ---
    ax_mink = fig.add_subplot(gs[0, 2])
    
    # Light cone
    x = np.linspace(-1, 1, 100)
    ax_mink.fill_between(x, -np.abs(x), np.abs(x), 
                        color=COLORS['minkowski'], alpha=0.15)
    ax_mink.plot(x, x, 'k-', linewidth=2.5)
    ax_mink.plot(x, -x, 'k-', linewidth=2.5)
    
    # Event point
    ax_mink.plot(0, 0, 'o', color=COLORS['minkowski'], markersize=10, 
                markeredgecolor='k', markeredgewidth=1.5, zorder=5)
    
    # Grid
    for t in [-0.8, -0.4, 0.4, 0.8]:
        ax_mink.axhline(t, color='gray', linewidth=0.5, alpha=0.3, linestyle=':')
        ax_mink.axvline(t, color='gray', linewidth=0.5, alpha=0.3, linestyle=':')
    
    ax_mink.set_xlim(-1.1, 1.1)
    ax_mink.set_ylim(-1.1, 1.1)
    ax_mink.set_aspect('equal')
    ax_mink.set_xlabel('Space $x$', fontsize=10)
    ax_mink.set_ylabel('Time $t$', fontsize=10)
    ax_mink.set_title('Minkowski Spacetime\nLorentzian Signature',
                     fontsize=11, weight='bold', color=COLORS['minkowski'])
    ax_mink.text(0, -1.35, 'Causality Preservation', ha='center', fontsize=9,
                style='italic', color=COLORS['minkowski'])
    
    # ==================== ROW 2: INPUT ENCODING ====================
    
    encodings = [
        ('Spectral Encoding', COLORS['sphere'], 
         ['Spherical Harmonics $Y_\\ell^m$', '36 coefficients']),
        ('Graph Encoding', COLORS['hyperbolic'],
         ['200 sample points', 'Graph structure']),
        ('Fourier Encoding', COLORS['minkowski'],
         ['Initial conditions', '10 Fourier modes'])
    ]
    
    for i, (title, color, items) in enumerate(encodings):
        ax = fig.add_subplot(gs[1, i])
        ax.text(0.5, 0.9, title, ha='center', fontsize=10, weight='bold',
               transform=ax.transAxes, color=color)
        
        for j, item in enumerate(items):
            y = 0.65 - j * 0.25
            draw_network_layer(ax, 0.5, y, 0.75, 0.15, item, color, fill=True)
        
        # Arrow down
        ax.annotate('', xy=(0.5, 0.15), xytext=(0.5, 0.35),
                   xycoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', lw=2, color=color))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # ==================== ROW 3: BRANCH NETWORKS ====================
    
    for i, color in enumerate([COLORS['sphere'], COLORS['hyperbolic'], COLORS['minkowski']]):
        ax = fig.add_subplot(gs[2, i])
        ax.text(0.5, 0.95, 'Branch Network', ha='center', fontsize=10, weight='bold',
               transform=ax.transAxes)
        
        layers = ['Dense [128]', 'Dense [128]', 'Latent $p=64$']
        for j, layer in enumerate(layers):
            y = 0.7 - j * 0.25
            draw_network_layer(ax, 0.5, y, 0.7, 0.12, layer, color, fill=(j==2))
            
            if j < len(layers) - 1:
                ax.annotate('', xy=(0.5, y - 0.18), xytext=(0.5, y - 0.06),
                           xycoords='axes fraction',
                           arrowprops=dict(arrowstyle='->', lw=1.5, color=color))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # ==================== ROW 4: TRUNK NETWORKS ====================
    
    trunk_features = [
        ('Geodesic Features', COLORS['sphere'],
         ['10 reference distances', 'Coordinates $(\\theta, \\phi)$']),
        ('Hyperbolic Features', COLORS['hyperbolic'],
         ['10 reference distances', 'Depth + curvature']),
        ('Causal Features', COLORS['minkowski'],
         ['Light cone coordinates', 'Proper time $\\tau$'])
    ]
    
    for i, (title, color, features) in enumerate(trunk_features):
        ax = fig.add_subplot(gs[3, i])
        ax.text(0.5, 0.95, 'Trunk Network', ha='center', fontsize=10, weight='bold',
               transform=ax.transAxes)
        
        # Feature box
        y_feat = 0.75
        for j, feat in enumerate(features):
            draw_network_layer(ax, 0.5, y_feat - j*0.15, 0.75, 0.12, feat, color, fill=True)
        
        # Arrow
        ax.annotate('', xy=(0.5, 0.38), xytext=(0.5, 0.52),
                   xycoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', lw=1.5, color=color))
        
        # Network layers
        layers = ['Dense [128]', 'Output $p=64$']
        for j, layer in enumerate(layers):
            y = 0.28 - j * 0.2
            draw_network_layer(ax, 0.5, y, 0.7, 0.12, layer, color, fill=(j==1))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # ==================== TITLE AND FOOTER ====================
    
    fig.suptitle('Geometric DeepONet Framework Across Three Geometries',
                fontsize=14, weight='bold', y=0.965)
    
    # Formula at bottom
    fig.text(0.5, 0.02, 
            '$u(y) = \\langle \\mathrm{Branch}(f), \\mathrm{Trunk}(y) \\rangle$ '
            '(inner product in latent space)',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
    
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✓ Saved professional figure: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Creating professional publication-quality Figure 1...")
    create_professional_figure1()
    print("Done!")
