"""
Final Clean Publication Figure 1
Simplified, clear, professional
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Wedge
import matplotlib.patches as mpatches

# Clean publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 300,
})

# Simple, clear colors
BLUE = '#1f77b4'
PURPLE = '#9467bd'
ORANGE = '#ff7f0e'


def create_simple_figure1(save_path='figure1_simple.pdf'):
    """Create simple, clear figure focusing on key concepts."""
    
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.25,
                  left=0.06, right=0.94, top=0.88, bottom=0.12)
    
    # ============ TOP ROW: GEOMETRIES ============
    
    # SPHERE
    ax1 = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Main circle
    ax1.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=3)
    ax1.fill(np.cos(theta), np.sin(theta), color=BLUE, alpha=0.1)
    
    # Grid lines
    for lat in [-0.7, 0, 0.7]:
        w = np.sqrt(max(0, 1 - lat**2))
        t = np.linspace(0, 2*np.pi, 50)
        ax1.plot(w * np.cos(t), lat + 0*t, color=BLUE, linewidth=1.5, alpha=0.5)
    
    for lon in [0, np.pi/3, 2*np.pi/3]:
        t = np.linspace(-np.pi/2, np.pi/2, 50)
        ax1.plot(np.cos(t) * np.cos(lon), np.sin(t), color=BLUE, linewidth=1.5, alpha=0.5)
    
    ax1.set_xlim(-1.4, 1.4)
    ax1.set_ylim(-1.4, 1.4)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Sphere $S^2$\\n$K > 0$', fontsize=14, weight='bold', pad=10)
    
    # HYPERBOLIC
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Disk
    circle = Circle((0, 0), 1, fill=True, facecolor=f'{PURPLE}15',
                   edgecolor='k', linewidth=3)
    ax2.add_patch(circle)
    
    # Geodesics
    for angle in [0, 45, 90, 135]:
        a = np.radians(angle)
        ax2.plot([-0.95*np.cos(a), 0.95*np.cos(a)],
                [-0.95*np.sin(a), 0.95*np.sin(a)],
                color=PURPLE, linewidth=2, alpha=0.6)
    
    # Circles
    for r in [0.5, 0.75]:
        c = Circle((0, 0), r, fill=False, edgecolor=PURPLE,
                  linewidth=1.5, alpha=0.4, linestyle='--')
        ax2.add_patch(c)
    
    ax2.set_xlim(-1.4, 1.4)
    ax2.set_ylim(-1.4, 1.4)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Hyperbolic $H^2$\\n$K < 0$', fontsize=14, weight='bold', pad=10)
    
    # MINKOWSKI
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Light cone
    x = np.linspace(-1, 1, 100)
    ax3.fill_between(x, -np.abs(x), np.abs(x), color=ORANGE, alpha=0.15)
    ax3.plot(x, x, 'k-', linewidth=3)
    ax3.plot(x, -x, 'k-', linewidth=3)
    
    # Event
    ax3.plot(0, 0, 'o', color=ORANGE, markersize=12, 
            markeredgecolor='k', markeredgewidth=2, zorder=5)
    
    # Grid
    for val in [-0.75, -0.25, 0.25, 0.75]:
        ax3.axhline(val, color='gray', linewidth=0.5, alpha=0.3, linestyle=':')
        ax3.axvline(val, color='gray', linewidth=0.5, alpha=0.3, linestyle=':')
    
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_aspect('equal')
    ax3.set_xlabel('Space', fontsize=11)
    ax3.set_ylabel('Time', fontsize=11)
    ax3.set_title('Minkowski\\nLorentzian', fontsize=14, weight='bold', pad=10)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # ============ BOTTOM ROW: DEEPONET ARCHITECTURE ============
    
    configs = [
        ('Spectral Branch\\n+ Geodesic Trunk', BLUE),
        ('Graph Branch\\n+ Hyperbolic Trunk', PURPLE),
        ('Fourier Branch\\n+ Causal Trunk', ORANGE)
    ]
    
    for i, (label, color) in enumerate(configs):
        ax = fig.add_subplot(gs[1, i])
        
        # Simple box diagram
        box_width = 0.7
        box_height = 0.15
        
        # Branch
        rect1 = FancyBboxPatch((0.5 - box_width/2, 0.7 - box_height/2), 
                              box_width, box_height,
                              boxstyle="round,pad=0.02",
                              edgecolor=color, facecolor='white',
                              linewidth=2.5, transform=ax.transAxes)
        ax.add_patch(rect1)
        ax.text(0.5, 0.7, 'Branch', ha='center', va='center', 
               fontsize=11, weight='bold', transform=ax.transAxes)
        
        # Arrow
        ax.annotate('', xy=(0.5, 0.45), xytext=(0.5, 0.62),
                   xycoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', lw=3, color=color))
        
        # Latent
        rect2 = FancyBboxPatch((0.5 - box_width/2, 0.35 - box_height/2),
                              box_width, box_height,
                              boxstyle="round,pad=0.02",
                              edgecolor=color, facecolor=f'{color}30',
                              linewidth=2.5, transform=ax.transAxes)
        ax.add_patch(rect2)
        ax.text(0.5, 0.35, 'Latent $p=64$', ha='center', va='center',
               fontsize=11, weight='bold', transform=ax.transAxes)
        
        # Arrow
        ax.annotate('', xy=(0.5, 0.10), xytext=(0.5, 0.27),
                   xycoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', lw=3, color=color))
        
        # Trunk
        rect3 = FancyBboxPatch((0.5 - box_width/2, 0.0),
                              box_width, box_height,
                              boxstyle="round,pad=0.02",
                              edgecolor=color, facecolor='white',
                              linewidth=2.5, transform=ax.transAxes)
        ax.add_patch(rect3)
        ax.text(0.5, 0.075, 'Trunk', ha='center', va='center',
               fontsize=11, weight='bold', transform=ax.transAxes)
        
        # Title
        ax.text(0.5, 0.95, label, ha='center', va='top',
               fontsize=12, weight='bold', color=color, transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Main title
    fig.suptitle('Geometric DeepONet Framework', fontsize=16, weight='bold', y=0.96)
    
    # Formula
    fig.text(0.5, 0.02, 
            '$u(y) = \\langle \\mathrm{Branch}(f), \\mathrm{Trunk}(y) \\rangle$',
            ha='center', fontsize=13,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.2))
    
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"✓ Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Creating simplified publication figure...")
    create_simple_figure1()
    print("Done!")
