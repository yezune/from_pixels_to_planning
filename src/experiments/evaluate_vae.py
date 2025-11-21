#!/usr/bin/env python3
"""
Evaluate trained VAE on Atari observations.

This script evaluates a trained VAE model by:
1. Computing reconstruction quality metrics
2. Visualizing latent space structure
3. Testing on multiple Atari games

Usage:
    python src/experiments/evaluate_vae.py --model_path outputs/vae_training/best_model.pt
"""

import argparse
import os
from pathlib import Path
import sys

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.vae import VAE
from src.envs.atari_env import AtariPixelEnv


def collect_samples(env_name: str, num_samples: int, device: str):
    """Collect sample observations from environment."""
    env = AtariPixelEnv(env_id=env_name, device=device)
    samples = []
    
    while len(samples) < num_samples:
        obs, _ = env.reset()
        samples.append(obs.cpu())
        
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            samples.append(obs.cpu())
            
            if terminated or truncated:
                break
            if len(samples) >= num_samples:
                break
    
    return torch.stack(samples[:num_samples])


@torch.no_grad()
def compute_reconstruction_metrics(model: VAE, samples: torch.Tensor, device: str):
    """Compute reconstruction quality metrics."""
    model.eval()
    samples = samples.to(device)
    
    # Forward pass
    recon, mu, logvar = model(samples)
    
    # MSE
    mse = torch.mean((recon - samples) ** 2).item()
    
    # PSNR
    psnr = 10 * np.log10(1.0 / mse)
    
    # Per-pixel accuracy (within 0.1)
    accuracy = torch.mean((torch.abs(recon - samples) < 0.1).float()).item()
    
    return {
        'mse': mse,
        'psnr': psnr,
        'accuracy': accuracy
    }


@torch.no_grad()
def visualize_reconstructions(
    model: VAE,
    samples: torch.Tensor,
    num_samples: int,
    device: str,
    save_path: str
):
    """Visualize original and reconstructed samples."""
    model.eval()
    
    indices = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)
    selected = samples[indices].to(device)
    
    recon, _, _ = model(selected)
    
    fig, axes = plt.subplots(2, len(indices), figsize=(len(indices) * 2, 4))
    if len(indices) == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(len(indices)):
        # Original
        orig = selected[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(orig)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # Reconstruction
        rec = recon[i].cpu().permute(1, 2, 0).numpy()
        rec = np.clip(rec, 0, 1)
        axes[1, i].imshow(rec)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstruction')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def visualize_latent_space(
    model: VAE,
    samples: torch.Tensor,
    device: str,
    save_path: str
):
    """Visualize latent space structure using first 2 dimensions."""
    model.eval()
    samples = samples.to(device)
    
    # Encode
    mu, logvar = model.encode(samples)
    mu = mu.cpu().numpy()
    
    # Plot first 2 dimensions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Scatter plot
    axes[0].scatter(mu[:, 0], mu[:, 1], alpha=0.5, s=10)
    axes[0].set_xlabel('Latent Dim 0')
    axes[0].set_ylabel('Latent Dim 1')
    axes[0].set_title('Latent Space (First 2 Dims)')
    axes[0].grid(True)
    
    # Histogram of first dimension
    axes[1].hist(mu[:, 0], bins=50, alpha=0.7)
    axes[1].set_xlabel('Latent Dim 0')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Latent Dim 0')
    axes[1].grid(True)
    
    # Histogram of all dimensions' means
    dim_means = np.mean(mu, axis=0)
    axes[2].bar(range(len(dim_means)), dim_means)
    axes[2].set_xlabel('Latent Dimension')
    axes[2].set_ylabel('Mean Value')
    axes[2].set_title('Mean Values Across Latent Dimensions')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def sample_from_latent(
    model: VAE,
    num_samples: int,
    device: str,
    save_path: str
):
    """Sample from latent space and decode."""
    model.eval()
    
    # Sample from standard normal
    z = torch.randn(num_samples, model.latent_dim).to(device)
    
    # Decode
    samples = model.decode(z)
    
    # Plot
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        img = samples[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')
    
    plt.suptitle('Samples from Prior')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained VAE')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--env_name', type=str, default='Breakout',
                       help='Atari environment name')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to evaluate on')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/mps/auto)')
    parser.add_argument('--output_dir', type=str, default='outputs/vae_evaluation',
                       help='Output directory')
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = VAE(input_shape=(3, 64, 64), latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Collect samples
    print(f"Collecting {args.num_samples} samples from {args.env_name}...")
    samples = collect_samples(args.env_name, args.num_samples, device)
    print(f"Collected {len(samples)} samples")
    
    # Compute metrics
    print("\nComputing reconstruction metrics...")
    metrics = compute_reconstruction_metrics(model, samples, device)
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  Accuracy (±0.1): {metrics['accuracy']*100:.2f}%")
    
    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Environment: {args.env_name}\n")
        f.write(f"Num Samples: {args.num_samples}\n")
        f.write(f"\nReconstruction Metrics:\n")
        f.write(f"  MSE: {metrics['mse']:.6f}\n")
        f.write(f"  PSNR: {metrics['psnr']:.2f} dB\n")
        f.write(f"  Accuracy (±0.1): {metrics['accuracy']*100:.2f}%\n")
    
    # Visualize reconstructions
    print("\nGenerating visualizations...")
    visualize_reconstructions(
        model, samples, num_samples=8, device=device,
        save_path=output_dir / 'reconstructions.png'
    )
    print(f"  Saved reconstructions to {output_dir / 'reconstructions.png'}")
    
    # Visualize latent space
    visualize_latent_space(
        model, samples, device=device,
        save_path=output_dir / 'latent_space.png'
    )
    print(f"  Saved latent space visualization to {output_dir / 'latent_space.png'}")
    
    # Sample from prior
    sample_from_latent(
        model, num_samples=8, device=device,
        save_path=output_dir / 'prior_samples.png'
    )
    print(f"  Saved prior samples to {output_dir / 'prior_samples.png'}")
    
    print(f"\nEvaluation complete! All outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
