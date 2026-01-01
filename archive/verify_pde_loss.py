
import torch
import numpy as np
import torch.nn as nn
from optimized_geometric_deeponet import OptimizedGeometricDeepONet

def test_pde_loss():
    print("Testing compute_laplace_beltrami...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    L_max = 5
    n_coeffs = (L_max + 1) ** 2
    n_points = 100
    batch_size = 4
    
    model = OptimizedGeometricDeepONet(L_max=L_max, n_refs=10, p=64, R=1.0).to(device)
    
    # Mock data
    coeffs = torch.randn(batch_size, n_coeffs).to(device).requires_grad_(True)
    
    # Mock grid
    theta = torch.linspace(0.1, np.pi-0.1, n_points).to(device) # Avoid poles
    phi = torch.linspace(0, 2*np.pi, n_points).to(device)
    
    # Initialize references
    model.trunk.initialize_references(theta.cpu().numpy(), phi.cpu().numpy())
    
    print("Computing Laplacian...")
    try:
        laplacian = model.compute_laplace_beltrami(coeffs, theta, phi)
        print(f"Laplacian shape: {laplacian.shape}")
        
        # Check gradients
        loss = torch.mean(laplacian**2)
        print("Backward pass...")
        loss.backward()
        print("Backward pass successful.")
        
        if coeffs.grad is not None:
            print("Gradients computed for coeffs.")
        else:
            print("WARNING: No gradients for coeffs!")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pde_loss()
