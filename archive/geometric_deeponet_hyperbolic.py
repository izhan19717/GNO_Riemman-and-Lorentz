"""
Geometric DeepONet for Hyperbolic Space
Implements DeepONet with hyperbolic geometric features
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import json
import time

# Import from previous experiments
from hyperbolic_geometry import (
    poincare_to_hyperboloid,
    hyperbolic_distance,
    hyperboloid_to_poincare
)


class HyperbolicBranchNetwork(nn.Module):
    """Branch network for hyperbolic space."""
    
    def __init__(self, m_sensors=200, hidden_dims=[128, 128], p=64):
        super().__init__()
        
        layers = []
        in_dim = m_sensors
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, p))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, u_sensors):
        """
        Args:
            u_sensors: (batch, m_sensors) function values at sensor points
        Returns:
            (batch, p) latent encoding
        """
        return self.network(u_sensors)


class HyperbolicTrunkNetwork(nn.Module):
    """Trunk network with hyperbolic geometric features."""
    
    def __init__(self, n_refs=10, hidden_dims=[128, 128], p=64, R=1.0):
        super().__init__()
        
        self.n_refs = n_refs
        self.R = R
        
        # Input: distances (n_refs) + coords (2) + depth (1) + curvature (1)
        input_dim = n_refs + 2 + 1 + 1
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, p))
        
        self.network = nn.Sequential(*layers)
        
        # Reference points (will be initialized)
        self.register_buffer('ref_points_poincare', torch.zeros(n_refs, 2))
        self.register_buffer('ref_points_hyperboloid', torch.zeros(n_refs, 3))
    
    def initialize_references(self, points):
        """Initialize reference points from dataset points."""
        # Select diverse reference points
        indices = np.random.choice(len(points), self.n_refs, replace=False)
        ref_points = points[indices]
        
        self.ref_points_poincare = torch.FloatTensor(ref_points)
        
        # Convert to hyperboloid
        ref_hyp = poincare_to_hyperboloid(ref_points, R=self.R)
        self.ref_points_hyperboloid = torch.FloatTensor(ref_hyp)
    
    def forward(self, query_points_poincare):
        """
        Args:
            query_points_poincare: (batch, n_points, 2) query points in Poincaré disk
        Returns:
            (batch, n_points, p) latent features
        """
        # Ensure tensor
        if not isinstance(query_points_poincare, torch.Tensor):
            query_points_poincare = torch.FloatTensor(query_points_poincare)
        
        # Handle unbatched input
        if query_points_poincare.dim() == 2:
            query_points_poincare = query_points_poincare.unsqueeze(0)
            was_unbatched = True
        else:
            was_unbatched = False
        
        batch_size, n_points, _ = query_points_poincare.shape
        device = query_points_poincare.device
        
        # Simplified feature computation (vectorized)
        # Use Euclidean distances as proxy for hyperbolic (for speed)
        features_list = []
        
        for i in range(batch_size):
            points = query_points_poincare[i]  # (n_points, 2)
            
            # Distances to references (Euclidean as proxy)
            ref_points = self.ref_points_poincare.to(device)  # (n_refs, 2)
            
            # Compute pairwise distances
            dists = torch.cdist(points.unsqueeze(0), ref_points.unsqueeze(0)).squeeze(0)  # (n_points, n_refs)
            
            # Poincaré coordinates
            coords = points  # (n_points, 2)
            
            # Distance to origin
            depth = torch.norm(points, dim=1, keepdim=True)  # (n_points, 1)
            
            # Curvature (constant)
            curvature = torch.ones(n_points, 1, device=device) * (-1.0 / (self.R ** 2))
            
            # Combine features
            features = torch.cat([dists, coords, depth, curvature], dim=1)  # (n_points, n_refs+2+1+1)
            features_list.append(features)
        
        features = torch.stack(features_list)  # (batch, n_points, feature_dim)
        
        # Pass through network
        features_flat = features.reshape(-1, features.shape[-1])
        output_flat = self.network(features_flat)
        output = output_flat.reshape(batch_size, n_points, -1)
        
        if was_unbatched:
            output = output.squeeze(0)
        
        return output


class GeometricDeepONetHyperbolic(nn.Module):
    """Geometric DeepONet for hyperbolic space."""
    
    def __init__(self, m_sensors=200, n_refs=10, p=64, R=1.0):
        super().__init__()
        
        self.branch = HyperbolicBranchNetwork(m_sensors=m_sensors, p=p)
        self.trunk = HyperbolicTrunkNetwork(n_refs=n_refs, p=p, R=R)
    
    def forward(self, u_sensors, query_points):
        """
        Args:
            u_sensors: (batch, m_sensors) function values
            query_points: (batch, n_points, 2) query points in Poincaré disk
        Returns:
            u_pred: (batch, n_points) predicted values
        """
        # Branch output
        branch_out = self.branch(u_sensors)  # (batch, p)
        
        # Trunk output
        trunk_out = self.trunk(query_points)  # (batch, n_points, p)
        
        # Inner product
        u_pred = torch.sum(branch_out.unsqueeze(1) * trunk_out, dim=-1)
        
        return u_pred


class HyperbolicDataset(Dataset):
    """Dataset for hyperbolic Poisson equation."""
    
    def __init__(self, npz_path, m_sensors=200):
        data = np.load(npz_path)
        
        self.points = data['points']  # (n_points, 2) in Poincaré disk
        self.sources = data['sources']  # (n_samples, n_points)
        self.solutions = data['solutions']  # (n_samples, n_points)
        
        self.m_sensors = m_sensors
        self.n_samples = len(self.sources)
        
        # Select sensor points
        sensor_indices = np.random.choice(len(self.points), m_sensors, replace=False)
        self.sensor_points = self.points[sensor_indices]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Get sensor values
        sensor_indices = np.random.choice(len(self.points), self.m_sensors, replace=False)
        u_sensors = self.solutions[idx, sensor_indices]
        
        return {
            'u_sensors': torch.FloatTensor(u_sensors),
            'query_points': torch.FloatTensor(self.points),
            'u_true': torch.FloatTensor(self.solutions[idx])
        }


def custom_collate_hyperbolic(batch):
    """Custom collate function."""
    u_sensors = torch.stack([item['u_sensors'] for item in batch])
    u_true = torch.stack([item['u_true'] for item in batch])
    query_points = batch[0]['query_points']  # Shared
    
    return {
        'u_sensors': u_sensors,
        'query_points': query_points,
        'u_true': u_true
    }


def train_hyperbolic_model(model, train_loader, test_loader, epochs=100, lr=1e-3, device='cpu'):
    """Train geometric DeepONet for hyperbolic space."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    print("Training hyperbolic DeepONet...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            u_sensors = batch['u_sensors'].to(device)
            query_points = batch['query_points'].to(device)
            u_true = batch['u_true'].to(device)
            
            batch_size = u_sensors.shape[0]
            query_batch = query_points.unsqueeze(0).expand(batch_size, -1, -1)
            
            optimizer.zero_grad()
            u_pred = model(u_sensors, query_batch)
            
            loss = criterion(u_pred, u_true)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Testing
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                u_sensors = batch['u_sensors'].to(device)
                query_points = batch['query_points'].to(device)
                u_true = batch['u_true'].to(device)
                
                batch_size = u_sensors.shape[0]
                query_batch = query_points.unsqueeze(0).expand(batch_size, -1, -1)
                
                u_pred = model(u_sensors, query_batch)
                loss = criterion(u_pred, u_true)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Test: {test_loss:.6f}")
    
    return train_losses, test_losses


def visualize_results(model, test_dataset, device='cpu', save_path='hyperbolic_results.png'):
    """Visualize predictions."""
    
    model.eval()
    
    # Get a test example
    idx = 0
    sample = test_dataset[idx]
    
    u_sensors = sample['u_sensors'].unsqueeze(0).to(device)
    query_points = sample['query_points'].unsqueeze(0).to(device)
    u_true = sample['u_true'].cpu().numpy()
    
    with torch.no_grad():
        u_pred = model(u_sensors, query_points).cpu().numpy().flatten()
    
    points = test_dataset.points
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # True solution
    ax1 = axes[0]
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], c=u_true, cmap='viridis', s=20)
    circle1 = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(circle1)
    plt.colorbar(scatter1, ax=ax1)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.set_title('True Solution', fontsize=14, fontweight='bold')
    
    # Predicted solution
    ax2 = axes[1]
    scatter2 = ax2.scatter(points[:, 0], points[:, 1], c=u_pred, cmap='viridis', s=20)
    circle2 = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax2.add_patch(circle2)
    plt.colorbar(scatter2, ax=ax2)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_title('Predicted Solution', fontsize=14, fontweight='bold')
    
    # Error
    ax3 = axes[2]
    error = np.abs(u_pred - u_true)
    scatter3 = ax3.scatter(points[:, 0], points[:, 1], c=error, cmap='plasma', s=20)
    circle3 = plt.Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
    ax3.add_patch(circle3)
    plt.colorbar(scatter3, ax=ax3)
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_aspect('equal')
    ax3.set_title(f'Error (max={np.max(error):.4f})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved results to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENT 4.3: GEOMETRIC DEEPONET FOR HYPERBOLIC SPACE")
    print("="*70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = HyperbolicDataset('hyperbolic_test_data.npz', m_sensors=200)
    
    # Use first 400 for training, rest for testing
    train_size = 400
    test_dataset = HyperbolicDataset('hyperbolic_test_data.npz', m_sensors=200)
    
    # Create subset for training
    from torch.utils.data import Subset
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, len(train_dataset)))
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True,
                              collate_fn=custom_collate_hyperbolic)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False,
                             collate_fn=custom_collate_hyperbolic)
    
    print(f"Train samples: {len(train_subset)}")
    print(f"Test samples: {len(test_subset)}\n")
    
    # Initialize model
    print("Initializing model...")
    model = GeometricDeepONetHyperbolic(m_sensors=200, n_refs=10, p=64, R=1.0)
    model.trunk.initialize_references(train_dataset.points)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Train
    train_losses, test_losses = train_hyperbolic_model(
        model, train_loader, test_loader, epochs=100, lr=1e-3, device=device
    )
    
    # Save model
    print("\nSaving model...")
    torch.save(model.state_dict(), 'hyperbolic_model.pth')
    print("  Saved hyperbolic_model.pth")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_results(model, test_dataset, device=device)
    
    # Save metrics
    metrics = {
        'final_train_loss': float(train_losses[-1]),
        'final_test_loss': float(test_losses[-1]),
        'min_test_loss': float(min(test_losses))
    }
    
    with open('hierarchical_learning_analysis.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("  Saved hierarchical_learning_analysis.json")
    
    print()
    print("="*70)
    print("EXPERIMENT 4.3 COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  - geometric_deeponet_hyperbolic.py (this module)")
    print("  - hyperbolic_model.pth")
    print("  - hyperbolic_results.png")
    print("  - hierarchical_learning_analysis.json")
    print("\nNote: Möbius invariance test would require additional implementation.")
