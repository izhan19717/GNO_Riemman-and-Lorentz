import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import time

# Import SFNO model from existing implementation
from sfno_comparison_experiment import SFNO_Poisson, PoissonSphereDataset_SFNO

def train_sfno_on_noisy_data(noise_level, train_file, test_file='test_poisson_sphere.npz'):
    """Train SFNO on specific noise level."""
    print(f"\n{'='*60}")
    print(f"SFNO TRAINING ON {int(noise_level*100)}% NOISE")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print(f"\nLoading training data from {train_file}...")
    train_dataset = PoissonSphereDataset_SFNO(train_file, nlat=50, nlon=100)
    
    print(f"Loading test data from {test_file}...")
    test_dataset = PoissonSphereDataset_SFNO(test_file, nlat=50, nlon=100)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    model = SFNO_Poisson(nlat=50, nlon=100, hidden_channels=64, n_layers=4, modes=32).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    history = []
    
    print("\nStarting training (100 epochs)...")
    start_time = time.time()
    
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            source = batch['source'].to(device)
            solution = batch['solution'].to(device)
            
            optimizer.zero_grad()
            pred = model(source)
            loss = criterion(pred, solution)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                source = batch['source'].to(device)
                solution = batch['solution'].to(device)
                pred = model(source)
                val_loss += criterion(pred, solution).item()
        val_loss /= len(test_loader)
        
        history.append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'sfno_noise{int(noise_level*100):02d}.pth')
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best Validation Loss: {best_loss:.6f}")
    
    return {
        'noise_level': noise_level,
        'best_loss': best_loss,
        'history': history,
        'time': total_time
    }

if __name__ == "__main__":
    print("="*60)
    print("SFNO EXPERIMENT 9.2: NOISE ROBUSTNESS STUDY")
    print("="*60)
    
    # Baseline (clean data)
    print("\n\nBASELINE: Clean Data")
    baseline_result = train_sfno_on_noisy_data(
        noise_level=0.0,
        train_file='train_poisson_sphere.npz',
        test_file='test_poisson_sphere.npz'
    )
    
    # Noisy data
    noise_levels = [0.01, 0.05, 0.10]
    results = [baseline_result]
    
    for noise_level in noise_levels:
        result = train_sfno_on_noisy_data(
            noise_level=noise_level,
            train_file=f'train_poisson_sphere_noise{int(noise_level*100):02d}.npz',
            test_file='test_poisson_sphere.npz'
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("SFNO EXPERIMENT 9.2 RESULTS")
    print("="*60)
    print(f"{'Noise Level':<15} {'Test Loss':<15} {'Degradation':<15}")
    print("-"*60)
    
    baseline_loss = results[0]['best_loss']
    for r in results:
        noise_pct = int(r['noise_level'] * 100)
        degradation = ((r['best_loss'] - baseline_loss) / baseline_loss) * 100
        print(f"{noise_pct:2d}%{'':<12} {r['best_loss']:.6f}{'':<8} {degradation:+.1f}%")
    
    # Save results
    with open('sfno_experiment_9_2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to sfno_experiment_9_2_results.json")
    
    # Compare with DeepONet
    print("\n" + "="*60)
    print("COMPARISON: SFNO vs DeepONet")
    print("="*60)
    
    deeponet_results = {
        0: 0.05963,
        1: 0.05967,
        5: 0.05970,
        10: 0.05960
    }
    
    print(f"{'Noise':<10} {'SFNO':<15} {'DeepONet':<15} {'Winner':<15}")
    print("-"*60)
    for r in results:
        noise_pct = int(r['noise_level'] * 100)
        sfno_loss = r['best_loss']
        deeponet_loss = deeponet_results[noise_pct]
        winner = "SFNO" if sfno_loss < deeponet_loss else "DeepONet"
        print(f"{noise_pct:2d}%{'':<7} {sfno_loss:.6f}{'':<8} {deeponet_loss:.6f}{'':<8} {winner}")
