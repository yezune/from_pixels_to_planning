#!/usr/bin/env python3
"""
Evaluate trained 3-Level Hierarchical RGM.

This script evaluates the hierarchical model by:
1. Testing reconstruction quality at each level
2. Measuring temporal abstraction effectiveness
3. Visualizing hierarchical predictions
4. Comparing with flat (single-level) model

Usage:
    python src/experiments/evaluate_hierarchical_model.py \
        --config_path outputs/hierarchical_training/hierarchical_config.pt \
        --model_dir outputs/hierarchical_training
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.vae import VAE
from src.models.transition import TransitionModel
from src.envs.atari_env import AtariPixelEnv
from train_hierarchical_model import SimpleVAE


def load_hierarchical_model(config_path: str, model_dir: Path, device: str):
    """Load all components of the hierarchical model."""
    
    config = torch.load(config_path, weights_only=False, map_location=device)
    
    # Level 0
    level0_vae = VAE(input_shape=(3, 64, 64), latent_dim=config['level0_latent_dim']).to(device)
    level0_vae.load_state_dict(torch.load(config['level0_vae_path'], weights_only=True, map_location=device))
    
    # Level 1
    level1_vae = SimpleVAE(
        input_dim=config['level0_latent_dim'],
        latent_dim=config['level1_latent_dim'],
        hidden_dim=128
    ).to(device)
    level1_vae.load_state_dict(torch.load(model_dir / 'level1_vae_best.pt', weights_only=True, map_location=device))
    
    level1_transition = TransitionModel(
        latent_dim=config['level1_latent_dim'],
        action_dim=config['action_dim'],
        hidden_dim=64
    ).to(device)
    level1_transition.load_state_dict(torch.load(model_dir / 'level1_transition_best.pt', weights_only=True, map_location=device))
    
    # Level 2
    level2_vae = SimpleVAE(
        input_dim=config['level1_latent_dim'],
        latent_dim=config['level2_latent_dim'],
        hidden_dim=64
    ).to(device)
    level2_vae.load_state_dict(torch.load(model_dir / 'level2_vae_best.pt', weights_only=True, map_location=device))
    
    level2_transition = TransitionModel(
        latent_dim=config['level2_latent_dim'],
        action_dim=config['action_dim'],
        hidden_dim=32
    ).to(device)
    level2_transition.load_state_dict(torch.load(model_dir / 'level2_transition_best.pt', weights_only=True, map_location=device))
    
    return {
        'level0_vae': level0_vae,
        'level1_vae': level1_vae,
        'level1_transition': level1_transition,
        'level2_vae': level2_vae,
        'level2_transition': level2_transition,
        'config': config
    }


@torch.no_grad()
def evaluate_reconstruction(models: dict, env_name: str, num_episodes: int, device: str):
    """Evaluate reconstruction quality through the hierarchy."""
    
    env = AtariPixelEnv(env_id=env_name, device=device)
    
    level0_vae = models['level0_vae']
    level1_vae = models['level1_vae']
    level2_vae = models['level2_vae']
    
    level0_vae.eval()
    level1_vae.eval()
    level2_vae.eval()
    
    mse_level0 = []
    mse_level1 = []
    mse_level2_to_0 = []  # Full reconstruction from Level 2
    
    print(f"Evaluating reconstruction on {num_episodes} episodes...")
    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        obs_batch = obs.unsqueeze(0).to(device)
        
        # Level 0: Pixel → z0 → recon0
        mu0, logvar0 = level0_vae.encode(obs_batch)
        z0 = level0_vae.reparameterize(mu0, logvar0)
        recon0 = level0_vae.decode(z0)
        mse_level0.append(torch.mean((recon0 - obs_batch) ** 2).item())
        
        # Level 1: z0 → z1 → recon_z0
        recon_z0, z1, (mu1, logvar1) = level1_vae(z0)
        mse_level1.append(torch.mean((recon_z0 - z0) ** 2).item())
        
        # Level 2: z1 → z2 → recon_z1 → recon_z0 → recon0
        recon_z1, z2, (mu2, logvar2) = level2_vae(z1)
        recon_z0_from_2 = level1_vae.decode(recon_z1)
        recon0_from_2 = level0_vae.decode(recon_z0_from_2)
        mse_level2_to_0.append(torch.mean((recon0_from_2 - obs_batch) ** 2).item())
    
    results = {
        'level0_mse': np.mean(mse_level0),
        'level1_mse': np.mean(mse_level1),
        'level2_to_0_mse': np.mean(mse_level2_to_0)
    }
    
    return results


@torch.no_grad()
def evaluate_temporal_abstraction(models: dict, env_name: str, num_episodes: int, device: str):
    """Evaluate temporal prediction at different resolutions."""
    
    env = AtariPixelEnv(env_id=env_name, device=device)
    config = models['config']
    
    level1_transition = models['level1_transition']
    level2_transition = models['level2_transition']
    level1_vae = models['level1_vae']
    level2_vae = models['level2_vae']
    level0_vae = models['level0_vae']
    
    level0_vae.eval()
    level1_vae.eval()
    level2_vae.eval()
    level1_transition.eval()
    level2_transition.eval()
    
    tau1 = config['level1_temporal_resolution']
    tau2 = config['level2_temporal_resolution']
    
    # Collect predictions
    level1_errors = []
    level2_errors = []
    
    print(f"\nEvaluating temporal abstraction (τ1={tau1}, τ2={tau2})...")
    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        
        # Encode to all levels
        obs_batch = obs.unsqueeze(0).to(device)
        _, z0, _ = level0_vae(obs_batch)
        _, z1, _ = level1_vae(z0)
        _, z2, _ = level2_vae(z1)
        
        # Collect trajectory
        trajectory_z1 = [z1]
        trajectory_z2 = [z2]
        actions = []
        
        for step in range(32):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            actions.append(action)
            
            obs_batch = obs.unsqueeze(0).to(device)
            mu0, logvar0 = level0_vae.encode(obs_batch)
            z0 = level0_vae.reparameterize(mu0, logvar0)
            _, z1, _ = level1_vae(z0)
            _, z2, _ = level2_vae(z1)
            
            trajectory_z1.append(z1)
            trajectory_z2.append(z2)
            
            if terminated or truncated:
                break
        
        # Evaluate Level 1 prediction (τ=4)
        for i in range(0, len(trajectory_z1) - tau1, tau1):
            z1_current = trajectory_z1[i]
            z1_target = trajectory_z1[i + tau1]
            action_tensor = torch.tensor([actions[i]], dtype=torch.long).to(device)
            
            (z1_pred, _), _ = level1_transition(z1_current, action_tensor)
            error = torch.mean((z1_pred - z1_target) ** 2).item()
            level1_errors.append(error)
        
        # Evaluate Level 2 prediction (τ=16)
        for i in range(0, len(trajectory_z2) - tau2, tau2):
            z2_current = trajectory_z2[i]
            z2_target = trajectory_z2[i + tau2]
            action_tensor = torch.tensor([actions[i]], dtype=torch.long).to(device)
            
            (z2_pred, _), _ = level2_transition(z2_current, action_tensor)
            error = torch.mean((z2_pred - z2_target) ** 2).item()
            level2_errors.append(error)
    
    return {
        'level1_prediction_mse': np.mean(level1_errors) if level1_errors else 0.0,
        'level2_prediction_mse': np.mean(level2_errors) if level2_errors else 0.0
    }


def visualize_hierarchy(models: dict, env_name: str, device: str, save_path: str):
    """Visualize hierarchical encoding/decoding."""
    
    env = AtariPixelEnv(env_id=env_name, device=device)
    
    level0_vae = models['level0_vae']
    level1_vae = models['level1_vae']
    level2_vae = models['level2_vae']
    
    level0_vae.eval()
    level1_vae.eval()
    level2_vae.eval()
    
    # Get sample observation
    obs, _ = env.reset()
    obs_batch = obs.unsqueeze(0).to(device)
    
    # Full forward pass
    with torch.no_grad():
        # Level 0
        mu0, logvar0 = level0_vae.encode(obs_batch)
        z0 = level0_vae.reparameterize(mu0, logvar0)
        recon0 = level0_vae.decode(z0)
        
        # Level 1
        recon_z0_from_1, z1, _ = level1_vae(z0)
        recon0_from_1 = level0_vae.decode(recon_z0_from_1)
        
        # Level 2
        recon_z1_from_2, z2, _ = level2_vae(z1)
        recon_z0_from_2 = level1_vae.decode(recon_z1_from_2)
        recon0_from_2 = level0_vae.decode(recon_z0_from_2)
    
    # Visualize
    fig = plt.figure(figsize=(16, 4))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)
    
    # Original
    ax1 = fig.add_subplot(gs[0, 0])
    img = obs.cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax1.imshow(img)
    ax1.set_title('Original', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Level 0 reconstruction
    ax2 = fig.add_subplot(gs[0, 1])
    img = recon0.cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax2.imshow(img)
    ax2.set_title(f'Level 0 Recon\n(32D)', fontsize=11)
    ax2.axis('off')
    
    # Level 1 → 0 reconstruction
    ax3 = fig.add_subplot(gs[0, 2])
    img = recon0_from_1.cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax3.imshow(img)
    ax3.set_title(f'Level 1 → 0\n(32D→16D→32D)', fontsize=11)
    ax3.axis('off')
    
    # Level 2 → 1 → 0 reconstruction
    ax4 = fig.add_subplot(gs[0, 3])
    img = recon0_from_2.cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax4.imshow(img)
    ax4.set_title(f'Level 2 → 1 → 0\n(32D→16D→8D→16D→32D)', fontsize=11)
    ax4.axis('off')
    
    plt.suptitle('Hierarchical Reconstruction', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hierarchical Model')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to hierarchical config')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--env_name', type=str, default='Breakout',
                       help='Atari environment')
    parser.add_argument('--num_episodes', type=int, default=50,
                       help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/mps/auto)')
    parser.add_argument('--output_dir', type=str, default='outputs/hierarchical_evaluation',
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
    
    # Load models
    print("Loading hierarchical model...")
    model_dir = Path(args.model_dir)
    models = load_hierarchical_model(args.config_path, model_dir, device)
    print("Models loaded!")
    
    config = models['config']
    print(f"\nModel Configuration:")
    print(f"  Level 0: {config['level0_latent_dim']}D (τ=1)")
    print(f"  Level 1: {config['level1_latent_dim']}D (τ={config['level1_temporal_resolution']})")
    print(f"  Level 2: {config['level2_latent_dim']}D (τ={config['level2_temporal_resolution']})")
    
    # Evaluate reconstruction
    print("\n" + "="*60)
    print("RECONSTRUCTION QUALITY")
    print("="*60)
    recon_results = evaluate_reconstruction(models, args.env_name, args.num_episodes, device)
    print(f"Level 0 MSE: {recon_results['level0_mse']:.6f}")
    print(f"Level 1 MSE: {recon_results['level1_mse']:.6f}")
    print(f"Level 2→0 MSE: {recon_results['level2_to_0_mse']:.6f}")
    
    # Evaluate temporal abstraction
    print("\n" + "="*60)
    print("TEMPORAL ABSTRACTION")
    print("="*60)
    temporal_results = evaluate_temporal_abstraction(models, args.env_name, args.num_episodes, device)
    print(f"Level 1 (τ={config['level1_temporal_resolution']}) Prediction MSE: {temporal_results['level1_prediction_mse']:.6f}")
    print(f"Level 2 (τ={config['level2_temporal_resolution']}) Prediction MSE: {temporal_results['level2_prediction_mse']:.6f}")
    
    # Visualize
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    visualize_hierarchy(models, args.env_name, device, output_dir / 'hierarchical_visualization.png')
    
    # Save results
    results = {
        **recon_results,
        **temporal_results
    }
    
    with open(output_dir / 'evaluation_results.txt', 'w') as f:
        f.write("Hierarchical Model Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Level 0: {config['level0_latent_dim']}D (τ=1)\n")
        f.write(f"  Level 1: {config['level1_latent_dim']}D (τ={config['level1_temporal_resolution']})\n")
        f.write(f"  Level 2: {config['level2_latent_dim']}D (τ={config['level2_temporal_resolution']})\n\n")
        f.write(f"Reconstruction Quality:\n")
        f.write(f"  Level 0 MSE: {recon_results['level0_mse']:.6f}\n")
        f.write(f"  Level 1 MSE: {recon_results['level1_mse']:.6f}\n")
        f.write(f"  Level 2→0 MSE: {recon_results['level2_to_0_mse']:.6f}\n\n")
        f.write(f"Temporal Prediction:\n")
        f.write(f"  Level 1 MSE: {temporal_results['level1_prediction_mse']:.6f}\n")
        f.write(f"  Level 2 MSE: {temporal_results['level2_prediction_mse']:.6f}\n")
    
    print(f"\nAll results saved to {output_dir}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
