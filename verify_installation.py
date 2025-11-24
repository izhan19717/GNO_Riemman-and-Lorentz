import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from src.geometry.sphere import Sphere
from src.geometry.minkowski import Minkowski
from src.models.branch import BranchNet
from src.models.trunk import TrunkNet
from src.models.gno import GeometricDeepONet

def test_geometry():
    print("Testing Geometry...")
    
    # Sphere
    sphere = Sphere(radius=1.0)
    p1 = torch.tensor([0.0, 0.0, 1.0]) # North pole
    p2 = torch.tensor([0.0, 0.0, -1.0]) # South pole
    d = sphere.dist(p1, p2)
    print(f"Sphere distance (N-S): {d.item()} (Expected: pi = {3.14159})")
    assert torch.abs(d - 3.14159) < 1e-3
    
    # Minkowski
    mink = Minkowski()
    e1 = torch.tensor([0.0, 0.0]) # Origin
    e2 = torch.tensor([1.0, 0.0]) # Time 1
    s2 = mink.dist(e1, e2)
    print(f"Minkowski interval (0,0)-(1,0): {s2.item()} (Expected: -1)")
    assert torch.abs(s2 - (-1.0)) < 1e-3
    
    print("Geometry OK.")

def test_model():
    print("Testing Model...")
    
    branch_input = 10
    trunk_input = 3
    hidden = 32
    output = 16
    
    branch = BranchNet(branch_input, hidden, output)
    trunk = TrunkNet(trunk_input, hidden, output)
    model = GeometricDeepONet(branch, trunk)
    
    batch_size = 5
    num_points = 20
    
    u_coeffs = torch.randn(batch_size, branch_input)
    y_points = torch.randn(batch_size, num_points, trunk_input)
    
    res = model(u_coeffs, y_points)
    print(f"Model output shape: {res.shape} (Expected: [{batch_size}, {num_points}, 1])")
    assert res.shape == (batch_size, num_points, 1)
    
    print("Model OK.")

if __name__ == "__main__":
    test_geometry()
    test_model()
