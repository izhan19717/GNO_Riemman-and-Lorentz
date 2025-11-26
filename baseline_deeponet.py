"""
Baseline DeepONet on Sphere
Vanilla DeepONet WITHOUT geometric structure (treats coordinates as Euclidean)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import json
import time


class BranchNetwork(nn.Module):
    """
    Branch network: processes function values at sensor points.
    Input: u(x_i) for i=1,...,m
    Output: p-dimensional latent vector
    """
    def __init__(self, m_sensors=100, hidden_dims=[128, 128], p=64):
        super().__init__()
        
        layers = []
        in_dim = m_sensors
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, p))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, u):
        """
        Args:
            u: (batch, m_sensors) function values
        Returns:
            (batch, p) latent vector
        """
        return self.network(u)


class TrunkNetwork(nn.Module):
    """
    Trunk network: processes query coordinates.
    Input: (theta, phi) treated as Euclidean coordinates
    Output: p-dimensional latent vector
    """
    def __init__(self, input_dim=2, hidden_dims=[128, 128], p=64):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, p))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, coords):
        """
        Args:
            coords: (batch, n_points, 2) coordinates (theta, phi)
        Returns:
            (batch, n_points, p) latent vectors
        """
        batch_size, n_points, _ = coords.shape
        coords_flat = coords.reshape(-1, 2)
        output_flat = self.network(coords_flat)
        return output_flat.reshape(batch_size, n_points, -1)


class BaselineDeepONet(nn.Module):
    """
    Vanilla DeepONet: inner product of branch and trunk outputs.
    """
    def __init__(self, m_sensors=100, p=64):
        super().__init__()
        
        self.branch = BranchNetwork(m_sensors=m_sensors, hidden_dims=[128, 128], p=p)
        self.trunk = TrunkNetwork(input_dim=2, hidden_dims=[128, 128], p=p)
        
    def forward(self, u_sensors, query_coords):
        """
        Args:
            u_sensors: (batch, m_sensors) function values at sensors
            query_coords: (batch, n_points, 2) query coordinates
            
        Returns:
            u_pred: (batch, n_points) predicted function values
        """
        # Branch output: (batch, p)
        branch_out = self.branch(u_sensors)
        
        # Trunk output: (batch, n_points, p)
        trunk_out = self.trunk(query_coords)
        
        # Inner product: (batch, n_points)
        # Expand branch_out to (batch, 1, p) for broadcasting
        u_pred = torch.sum(branch_out.unsqueeze(1) * trunk_out, dim=-1)
        
        return u_pred


class PoissonDataset(Dataset):
    """Dataset for Poisson equation on sphere."""
    
    def __init__(self, npz_path, m_sensors=100):
        data = np.load(npz_path)
        
        self.sources = torch.FloatTensor(data['sources'])  # (N, n_theta, n_phi)
        self.solutions = torch.FloatTensor(data['solutions'])  # (N, n_theta, n_phi)
        self.theta = data['theta']
        self.phi = data['phi']
        
        self.n_samples = len(self.sources)
        self.n_theta = len(self.theta)
        self.n_phi = len(self.phi)
        
        # Create coordinate grid
        Theta, Phi = np.meshgrid(self.theta, self.phi, indexing='ij')
        self.coords = torch.FloatTensor(np.stack([Theta.flatten(), Phi.flatten()], axis=-1))
        
        # Select random sensor points
        np.random.seed(42)
        total_points = self.n_theta * self.n_phi
        self.sensor_indices = np.random.choice(total_points, m_sensors, replace=False)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Flatten source and solution
        source_flat = self.sources[idx].flatten()
        solution_flat = self.solutions[idx].flatten()
        
        # Get sensor values
        u_sensors = source_flat[self.sensor_indices]
        
        return {
            'u_sensors': u_sensors,
            'coords': self.coords,  # Shape: (n_points, 2)
            'u_true': solution_flat  # Shape: (n_points,)
        }


def custom_collate(batch):
    """
    Custom collate function to handle shared coordinates.
    All samples share the same coordinate grid, so we don't stack coords.
    """
    u_sensors = torch.stack([item['u_sensors'] for item in batch])
    u_true = torch.stack([item['u_true'] for item in batch])
    coords = batch[0]['coords']  # All samples have same coords, take first
    
    return {
        'u_sensors': u_sensors,
        'coords': coords,
        'u_true': u_true
    }


def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, device='cpu'):
    """Train the baseline DeepONet."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            u_sensors = batch['u_sensors'].to(device)
            coords = batch['coords'].to(device)  # Now properly (n_points, 2)
            u_true = batch['u_true'].to(device)
            
            batch_size = u_sensors.shape[0]
            
            optimizer.zero_grad()
            
            # Expand coords to match batch size: (batch, n_points, 2)
            coords_batch = coords.unsqueeze(0).expand(batch_size, -1, -1)
            
            u_pred = model(u_sensors, coords_batch)
            
            loss = criterion(u_pred, u_true)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                u_sensors = batch['u_sensors'].to(device)
                coords = batch['coords'].to(device)
                u_true = batch['u_true'].to(device)
                
                batch_size = u_sensors.shape[0]
                coords_batch = coords.unsqueeze(0).expand(batch_size, -1, -1)
                
                u_pred = model(u_sensors, coords_batch)
                
                loss = criterion(u_pred, u_true)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return train_losses, val_losses


def compute_metrics(model, test_loader, device='cpu'):
    """Compute evaluation metrics."""
    
    model.eval()
    model = model.to(device)
    
    relative_errors = []
    max_errors = []
    inference_times = []
    
    with torch.no_grad():
        for batch in test_loader:
            u_sensors = batch['u_sensors'].to(device)
            coords = batch['coords'].to(device)
            u_true = batch['u_true'].to(device)
            
            batch_size = u_sensors.shape[0]
            coords_batch = coords.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Measure inference time
            start_time = time.time()
            u_pred = model(u_sensors, coords_batch)
            inference_time = time.time() - start_time
            inference_times.append(inference_time / batch_size)
            
            # Compute errors
            for i in range(len(u_sensors)):
                pred = u_pred[i]
                true = u_true[i]
                
                # Relative L2 error
                rel_error = torch.norm(pred - true) / (torch.norm(true) + 1e-10)
                relative_errors.append(rel_error.item())
                
                # Max pointwise error
                max_error = torch.max(torch.abs(pred - true))
                max_errors.append(max_error.item())
    
    metrics = {
        'mean_relative_l2_error': float(np.mean(relative_errors)),
        'std_relative_l2_error': float(np.std(relative_errors)),
        'mean_max_error': float(np.mean(max_errors)),
        'mean_inference_time_ms': float(np.mean(inference_times) * 1000)
    }
    
    return metrics, relative_errors


def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """Plot training and validation loss curves."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Baseline DeepONet Training Curves', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def visualize_predictions(model, test_dataset, theta, phi, n_examples=10, 
                         save_path='prediction_examples.png', device='cpu'):
    """Visualize predictions on test examples."""
    
    model.eval()
    model = model.to(device)
    
    # Select random examples
    indices = np.random.choice(len(test_dataset), n_examples, replace=False)
    
    # Convert to Cartesian for 3D plotting
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')
    X = np.sin(Theta) * np.cos(Phi)
    Y = np.sin(Theta) * np.sin(Phi)
    Z = np.cos(Theta)
    
    fig = plt.figure(figsize=(15, 3 * n_examples))
    
    with torch.no_grad():
        for idx, i in enumerate(indices):
            sample = test_dataset[i]
            
            u_sensors = sample['u_sensors'].unsqueeze(0).to(device)
            coords = sample['coords'].unsqueeze(0).to(device)
            u_true = sample['u_true'].cpu().numpy().reshape(len(theta), len(phi))
            
            # Predict
            u_pred = model(u_sensors, coords).cpu().numpy().reshape(len(theta), len(phi))
            
            # Error
            error = np.abs(u_pred - u_true)
            
            # True solution
            ax1 = fig.add_subplot(n_examples, 3, idx*3 + 1, projection='3d')
            norm_true = (u_true - u_true.min()) / (u_true.max() - u_true.min() + 1e-10)
            ax1.plot_surface(X, Y, Z, facecolors=cm.viridis(norm_true), alpha=0.9)
            ax1.set_title(f'Example {idx+1}: True', fontsize=10)
            ax1.axis('off')
            ax1.set_box_aspect([1,1,1])
            
            # Predicted solution
            ax2 = fig.add_subplot(n_examples, 3, idx*3 + 2, projection='3d')
            norm_pred = (u_pred - u_pred.min()) / (u_pred.max() - u_pred.min() + 1e-10)
            ax2.plot_surface(X, Y, Z, facecolors=cm.viridis(norm_pred), alpha=0.9)
            ax2.set_title(f'Predicted', fontsize=10)
            ax2.axis('off')
            ax2.set_box_aspect([1,1,1])
            
            # Error
            ax3 = fig.add_subplot(n_examples, 3, idx*3 + 3, projection='3d')
            norm_error = error / (error.max() + 1e-10)
            ax3.plot_surface(X, Y, Z, facecolors=cm.plasma(norm_error), alpha=0.9)
            ax3.set_title(f'Error (max={error.max():.4f})', fontsize=10)
            ax3.axis('off')
            ax3.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved prediction examples to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENT 1.4: BASELINE DEEPONET ON SPHERE")
    print("="*70)
    print()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = PoissonDataset('train_poisson_sphere.npz', m_sensors=100)
    test_dataset = PoissonDataset('test_poisson_sphere.npz', m_sensors=100)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Initialize model
    print("Initializing model...")
    model = BaselineDeepONet(m_sensors=100, p=64)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Train
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, 
        epochs=200, lr=1e-3, device=device
    )
    
    # Save model
    print("\nSaving model...")
    torch.save(model.state_dict(), 'trained_baseline_model.pth')
    print("  Saved trained_baseline_model.pth")
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(train_losses, val_losses)
    
    # Compute metrics
    print("\nComputing evaluation metrics...")
    metrics, relative_errors = compute_metrics(model, test_loader, device=device)
    
    print(f"\nMetrics:")
    print(f"  Mean Relative L² Error: {metrics['mean_relative_l2_error']:.6f} ± {metrics['std_relative_l2_error']:.6f}")
    print(f"  Mean Max Error: {metrics['mean_max_error']:.6f}")
    print(f"  Mean Inference Time: {metrics['mean_inference_time_ms']:.2f} ms/sample")
    
    # Save metrics
    with open('baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("\n  Saved baseline_metrics.json")
    
    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, test_dataset, train_dataset.theta, train_dataset.phi, 
                         n_examples=10, device=device)
    
    print()
    print("="*70)
    print("EXPERIMENT 1.4 COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  - baseline_deeponet.py (this module)")
    print("  - trained_baseline_model.pth")
    print("  - training_curves.png")
    print("  - prediction_examples.png (10×3 grid)")
    print("  - baseline_metrics.json")
