"""
Experiment 9.2: Noise Robustness Study
Generate noisy versions of the training data to test robustness.
"""

import numpy as np
import json

def add_noise_to_dataset(clean_file, noise_levels=[0.01, 0.05, 0.10]):
    """
    Add Gaussian noise to source functions at different levels.
    
    Args:
        clean_file: Path to clean data (train_poisson_sphere.npz)
        noise_levels: List of noise standard deviations (as fraction of signal std)
    """
    print(f"Loading clean data from {clean_file}...")
    data = np.load(clean_file)
    
    sources = data['sources']
    solutions = data['solutions']
    theta = data['theta']
    phi = data['phi']
    
    # Compute signal statistics
    signal_std = np.std(sources)
    print(f"Signal std: {signal_std:.4f}")
    
    results = {}
    
    for noise_level in noise_levels:
        print(f"\nGenerating {int(noise_level*100)}% noise dataset...")
        
        # Add Gaussian noise
        noise_std = noise_level * signal_std
        noise = noise_std * np.random.randn(*sources.shape)
        noisy_sources = sources + noise
        
        # Calculate SNR
        snr = 20 * np.log10(signal_std / noise_std)
        print(f"  Noise std: {noise_std:.4f}")
        print(f"  SNR: {snr:.2f} dB")
        
        # Save noisy dataset
        output_file = f'train_poisson_sphere_noise{int(noise_level*100):02d}.npz'
        np.savez(
            output_file,
            sources=noisy_sources,
            solutions=solutions,  # Ground truth unchanged
            theta=theta,
            phi=phi
        )
        print(f"  Saved to {output_file}")
        
        results[noise_level] = {
            'noise_std': float(noise_std),
            'snr_db': float(snr),
            'output_file': output_file
        }
    
    # Save metadata
    with open('noise_experiment_metadata.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("NOISE DATASETS GENERATED")
    print("="*60)
    for level, info in results.items():
        print(f"{int(level*100):2d}% noise: SNR = {info['snr_db']:.1f} dB")
    
    return results

if __name__ == "__main__":
    print("="*60)
    print("EXPERIMENT 9.2: NOISE ROBUSTNESS DATA GENERATION")
    print("="*60)
    
    # Generate noisy versions of training data
    results = add_noise_to_dataset(
        'train_poisson_sphere.npz',
        noise_levels=[0.01, 0.05, 0.10]
    )
    
    print("\nDone!")
