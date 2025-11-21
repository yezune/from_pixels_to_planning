#!/usr/bin/env python3
"""
Evaluate trained Transition Model.

This script evaluates transition model prediction accuracy
and visualizes prediction quality over multiple timesteps.

Usage:
    python src/experiments/evaluate_transition.py \
        --vae_path outputs/vae_full_training/best_model.pt \
        --transition_path outputs/transition_full_training/best_model.pt
"""

import argparse
import os
from pathlib import Path
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.vae import VAE
from src.models.transition import TransitionModel
from src.envs.atari_env import AtariPixelEnv


def collect_test_trajectories(
    env_name: str,
    vae: VAE,
    num_episodes: int,
    max_steps: int,
    device: str
):
    """Collect test trajectories with latent states."""
    env = AtariPixelEnv(env_id=env_name, device=device)
    vae.eval()
    trajectories = []
    
    print(f"Collecting {num_episodes} test trajectories...")
    for _ in tqdm(range(num_episodes)):
        trajectory = {'z': [], 'actions': []}
        obs, _ = env.reset()
        
        with torch.no_grad():
            obs_batch = obs.unsqueeze(0).to(device)
            _, z, _ = vae(obs_batch)
            trajectory['z'].append(z.squeeze(0).cpu())
        
        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            
            with torch.no_grad():
                obs_batch = obs.unsqueeze(0).to(device)
                _, z, _ = vae(obs_batch)
                trajectory['z'].append(z.squeeze(0).cpu())
                trajectory['actions'].append(action)
            
            if terminated or truncated:
                break
        
        if len(trajectory['z']) > 1:
            trajectories.append(trajectory)
    
    print(f"Collected {len(trajectories)} trajectories")
    return trajectories


@torch.no_grad()
def evaluate_one_step_prediction(
    model: TransitionModel,
    trajectories: list,
    device: str
):
    """Evaluate one-step ahead prediction accuracy."""
    model.eval()
    
    errors = []
    for traj in trajectories:
        z_states = torch.stack(traj['z']).to(device)
        actions = torch.tensor(traj['actions'], dtype=torch.long).to(device)
        
        for t in range(len(actions)):
            z_t = z_states[t:t+1]
            action = actions[t:t+1]
            z_next_true = z_states[t+1]
            
            (z_pred, _), _ = model(z_t, action)
            error = torch.mean((z_pred.squeeze(0) - z_next_true) ** 2).item()
            errors.append(error)
    
    return {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'median': np.median(errors),
        'min': np.min(errors),
        'max': np.max(errors)
    }


@torch.no_grad()
def evaluate_multi_step_prediction(
    model: TransitionModel,
    trajectories: list,
    device: str,
    steps: int = 10
):
    """Evaluate multi-step prediction accuracy."""
    model.eval()
    
    errors_by_step = [[] for _ in range(steps)]
    
    for traj in tqdm(trajectories, desc="Multi-step eval"):
        if len(traj['z']) < steps + 1:
            continue
        
        z_states = torch.stack(traj['z']).to(device)
        actions = torch.tensor(traj['actions'], dtype=torch.long).to(device)
        
        # Start from beginning
        z_t = z_states[0:1]
        hidden = None
        
        for step in range(steps):
            action = actions[step:step+1]
            z_true = z_states[step+1]
            
            (z_pred, _), hidden = model(z_t, action, hidden)
            error = torch.mean((z_pred.squeeze(0) - z_true) ** 2).item()
            errors_by_step[step].append(error)
            
            # Use prediction for next step
            z_t = z_pred
    
    results = []
    for step in range(steps):
        if errors_by_step[step]:
            results.append({
                'step': step + 1,
                'mean': np.mean(errors_by_step[step]),
                'std': np.std(errors_by_step[step])
            })
    
    return results


def plot_multi_step_results(results: list, save_path: str):
    """Plot multi-step prediction errors."""
    steps = [r['step'] for r in results]
    means = [r['mean'] for r in results]
    stds = [r['std'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(steps, means, 'b-', linewidth=2, label='Mean Error')
    ax.fill_between(steps, 
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.3, label='±1 Std')
    
    ax.set_xlabel('Prediction Steps', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Multi-Step Prediction Error', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Transition Model')
    parser.add_argument('--vae_path', type=str, required=True,
                       help='Path to trained VAE')
    parser.add_argument('--transition_path', type=str, required=True,
                       help='Path to trained Transition Model')
    parser.add_argument('--env_name', type=str, default='Breakout',
                       help='Atari environment name')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--num_episodes', type=int, default=50,
                       help='Number of test episodes')
    parser.add_argument('--max_steps', type=int, default=100,
                       help='Maximum steps per episode')
    parser.add_argument('--multi_step', type=int, default=20,
                       help='Number of steps for multi-step prediction')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/mps/auto)')
    parser.add_argument('--output_dir', type=str, default='outputs/transition_evaluation',
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
    
    # Load VAE
    print(f"Loading VAE from {args.vae_path}...")
    vae = VAE(input_shape=(3, 64, 64), latent_dim=args.latent_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_path, weights_only=True, map_location=device))
    vae.eval()
    
    # Load Transition Model
    print(f"Loading Transition Model from {args.transition_path}...")
    env = AtariPixelEnv(env_id=args.env_name, device=device)
    action_dim = env.action_space.n
    
    transition = TransitionModel(
        latent_dim=args.latent_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    transition.load_state_dict(torch.load(args.transition_path, weights_only=True, map_location=device))
    transition.eval()
    print("Models loaded successfully!")
    
    # Collect test trajectories
    trajectories = collect_test_trajectories(
        env_name=args.env_name,
        vae=vae,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        device=device
    )
    
    # Evaluate one-step prediction
    print("\nEvaluating one-step prediction...")
    one_step_results = evaluate_one_step_prediction(
        transition, trajectories, device
    )
    
    print("One-Step Prediction Results:")
    print(f"  Mean MSE: {one_step_results['mean']:.6f}")
    print(f"  Std MSE: {one_step_results['std']:.6f}")
    print(f"  Median MSE: {one_step_results['median']:.6f}")
    print(f"  Min MSE: {one_step_results['min']:.6f}")
    print(f"  Max MSE: {one_step_results['max']:.6f}")
    
    # Evaluate multi-step prediction
    print(f"\nEvaluating {args.multi_step}-step prediction...")
    multi_step_results = evaluate_multi_step_prediction(
        transition, trajectories, device, steps=args.multi_step
    )
    
    print("Multi-Step Prediction Results:")
    for result in multi_step_results[:5]:  # Print first 5
        print(f"  Step {result['step']}: {result['mean']:.6f} ± {result['std']:.6f}")
    if len(multi_step_results) > 5:
        print(f"  ...")
        print(f"  Step {multi_step_results[-1]['step']}: {multi_step_results[-1]['mean']:.6f} ± {multi_step_results[-1]['std']:.6f}")
    
    # Plot results
    plot_multi_step_results(
        multi_step_results,
        output_dir / 'multi_step_prediction.png'
    )
    
    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write(f"VAE: {args.vae_path}\n")
        f.write(f"Transition Model: {args.transition_path}\n")
        f.write(f"Environment: {args.env_name}\n")
        f.write(f"Test Episodes: {args.num_episodes}\n")
        f.write(f"\nOne-Step Prediction:\n")
        f.write(f"  Mean MSE: {one_step_results['mean']:.6f}\n")
        f.write(f"  Std MSE: {one_step_results['std']:.6f}\n")
        f.write(f"  Median MSE: {one_step_results['median']:.6f}\n")
        f.write(f"\nMulti-Step Prediction:\n")
        for result in multi_step_results:
            f.write(f"  Step {result['step']}: {result['mean']:.6f} ± {result['std']:.6f}\n")
    
    print(f"\nAll outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
