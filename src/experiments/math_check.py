import torch
import numpy as np
import sys
import os

# Add project root
sys.path.append(os.getcwd())

from src.geometry.sphere import Sphere
from src.geometry.hyperbolic import Hyperboloid
from src.geometry.minkowski import Minkowski

def check_inverse_consistency(manifold, name="Manifold"):
    print(f"\n--- Checking {name} Inverse Consistency (Exp/Log) ---")
    
    # 1. Sample random points x
    if hasattr(manifold, 'random_point'):
        x = manifold.random_point(10)
    elif isinstance(manifold, Sphere):
        # Random point on sphere
        x = torch.randn(10, 3)
        x = x / torch.norm(x, dim=-1, keepdim=True) * manifold.radius
    elif isinstance(manifold, Minkowski):
        x = torch.randn(10, 2)
    else:
        print("Skipping: No random sampling implemented.")
        return

    # 2. Sample random tangent vectors v
    # For Sphere/Hyperboloid, project random vector
    v_raw = torch.randn_like(x)
    if hasattr(manifold, 'proj_tan'):
        v = manifold.proj_tan(x, v_raw)
    else:
        v = v_raw # Flat space
        
    # Scale v to be reasonable
    v = v * 0.5
    
    # 3. y = Exp_x(v)
    y = manifold.exp(x, v)
    
    # 4. v_rec = Log_x(y)
    v_rec = manifold.log(x, y)
    
    # 5. Error
    error = torch.norm(v - v_rec, dim=-1).mean().item()
    print(f"Exp -> Log Reconstruction Error: {error:.2e}")
    
    if error < 1e-4:
        print(">> PASSED")
    else:
        print(">> FAILED")

def check_distance_axioms(manifold, name="Manifold"):
    print(f"\n--- Checking {name} Distance Axioms ---")
    
    if hasattr(manifold, 'random_point'):
        p1 = manifold.random_point(5)
        p2 = manifold.random_point(5)
    elif isinstance(manifold, Sphere):
        p1 = torch.randn(5, 3); p1 = p1/p1.norm(dim=-1, keepdim=True)
        p2 = torch.randn(5, 3); p2 = p2/p2.norm(dim=-1, keepdim=True)
    elif isinstance(manifold, Minkowski):
        p1 = torch.randn(5, 2)
        p2 = torch.randn(5, 2)
        
    # Symmetry
    d12 = manifold.dist(p1, p2)
    d21 = manifold.dist(p2, p1)
    
    sym_err = torch.abs(d12 - d21).mean().item()
    print(f"Symmetry Error: {sym_err:.2e}")
    
    if sym_err < 1e-5:
        print(">> PASSED")
    else:
        print(">> FAILED")

def check_hyperbolic_invariants():
    print("\n--- Checking Hyperboloid Invariants ---")
    H = Hyperboloid()
    
    # 1. Check if points are actually on the hyperboloid
    p = H.random_point(100)
    # <p, p>_L should be -1
    norm_sq = H.minkowski_dot(p, p)
    err = torch.abs(norm_sq + 1.0).mean().item()
    print(f"Manifold Constraint Error (<p,p>_L = -1): {err:.2e}")
    
    if err < 1e-5:
        print(">> PASSED")
    else:
        print(">> FAILED")

if __name__ == "__main__":
    print("Running Mathematical Verification Suite...")
    
    # Sphere
    S2 = Sphere()
    check_inverse_consistency(S2, "Sphere")
    check_distance_axioms(S2, "Sphere")
    
    # Hyperboloid
    H2 = Hyperboloid()
    check_inverse_consistency(H2, "Hyperboloid")
    check_distance_axioms(H2, "Hyperboloid")
    check_hyperbolic_invariants()
    
    # Minkowski
    # Note: Minkowski distance is interval, can be negative. Log map not fully defined for all pairs (spacelike/timelike separation).
    # We skip full Exp/Log check for Minkowski in this generic suite unless we handle causality.
    print("\nSkipping Minkowski Exp/Log check (requires causal handling).")
