#!/usr/bin/env python3
"""
Test integrated VAE + Transition Model as a world model.

This script demonstrates using the trained VAE and Transition Model
together for multi-step prediction in the latent space.

Usage:
    python src/experiments/test_integrated_world_model.py
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.vae import VAE
from src.models.transition import TransitionModel
from src.envs.atari_env import AtariPixelEnv


class IntegratedWorldModel:
    """World model combining VAE and Transition Model."""
    
    def __init__(
        self,
        vae: VAE,
        transition: TransitionModel,
        device: str = 'cpu'
    ):
        self.vae = vae
        self.transition = transition
        self.device = device
        
        self.vae.eval()
        self.transition.eval()
    
    @torch.no_grad()
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent state."""
        _, z, _ = self.vae(obs)
        return z
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent state to observation."""
        recon = self.vae.decode(z)
        return recon
    
    @torch.no_grad()
    def predict_next(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor = None
    ):
        """Predict next latent state given current state and action."""
        (z_next, _), hidden_next = self.transition(z, action, hidden)
        return z_next, hidden_next
    
    @torch.no_grad()
    def simulate_trajectory(
        self,
        initial_obs: torch.Tensor,
        actions: list,
        return_images: bool = True
    ):
        """
        Simulate trajectory in latent space and optionally decode to images.
        
        Args:
            initial_obs: Initial observation [1, C, H, W]
            actions: List of actions to take
            return_images: Whether to decode predictions to images
        
        Returns:
            predicted_states: List of predicted latent states
            predicted_images: List of predicted observations (if return_images=True)
        """
        # Encode initial observation
        z = self.encode(initial_obs)
        
        predicted_states = [z]
        predicted_images = []
        
        if return_images:
            predicted_images.append(self.decode(z))
        
        # Predict future states
        hidden = None
        for action in actions:
            action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
            z, hidden = self.predict_next(z, action_tensor, hidden)
            predicted_states.append(z)
            
            if return_images:
                predicted_images.append(self.decode(z))
        
        return predicted_states, predicted_images


def visualize_trajectory_comparison(
    real_frames: list,
    predicted_frames: list,
    save_path: str,
    num_frames: int = 10
):
    """Visualize real vs predicted trajectory."""
    num_frames = min(num_frames, len(real_frames), len(predicted_frames))
    
    fig = plt.figure(figsize=(20, 4))
    gs = GridSpec(2, num_frames, figure=fig, hspace=0.3, wspace=0.05)
    
    for i in range(num_frames):
        # Real frame
        ax1 = fig.add_subplot(gs[0, i])
        real = real_frames[i].cpu().squeeze(0).permute(1, 2, 0).numpy()
        real = np.clip(real, 0, 1)
        ax1.imshow(real)
        ax1.axis('off')
        if i == 0:
            ax1.set_title('Real', fontsize=12, fontweight='bold')
        else:
            ax1.set_title(f't={i}', fontsize=10)
        
        # Predicted frame
        ax2 = fig.add_subplot(gs[1, i])
        pred = predicted_frames[i].cpu().squeeze(0).permute(1, 2, 0).numpy()
        pred = np.clip(pred, 0, 1)
        ax2.imshow(pred)
        ax2.axis('off')
        if i == 0:
            ax2.set_title('Predicted', fontsize=12, fontweight='bold')
    
    plt.suptitle('Real vs Predicted Trajectory', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory comparison to {save_path}")


def compute_prediction_metrics(
    real_frames: list,
    predicted_frames: list
):
    """Compute metrics comparing real and predicted frames."""
    metrics = []
    
    for i, (real, pred) in enumerate(zip(real_frames, predicted_frames)):
        mse = torch.mean((real - pred) ** 2).item()
        psnr = 10 * torch.log10(torch.tensor(1.0 / (mse + 1e-10))).item()
        metrics.append({'step': i, 'mse': mse, 'psnr': psnr})
    
    return metrics


def main():
    # Configuration
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = Path('outputs/integrated_world_model')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("Loading models...")
    vae = VAE(input_shape=(3, 64, 64), latent_dim=32).to(device)
    vae.load_state_dict(torch.load(
        'outputs/vae_full_training/best_model.pt',
        weights_only=True,
        map_location=device
    ))
    
    env = AtariPixelEnv(env_id='Breakout', device=device)
    action_dim = env.action_space.n
    
    transition = TransitionModel(latent_dim=32, action_dim=action_dim, hidden_dim=64).to(device)
    transition.load_state_dict(torch.load(
        'outputs/transition_full_training/best_model.pt',
        weights_only=True,
        map_location=device
    ))
    print("Models loaded!")
    
    # Create integrated world model
    world_model = IntegratedWorldModel(vae, transition, device)
    
    # Test 1: Single trajectory prediction
    print("\nTest 1: Predicting 20-step trajectory...")
    obs, _ = env.reset()
    real_frames = [obs]
    actions = []
    
    # Collect real trajectory
    for _ in range(20):
        action = env.action_space.sample()
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        real_frames.append(obs)
        
        if terminated or truncated:
            break
    
    # Predict trajectory
    initial_obs = real_frames[0].unsqueeze(0).to(device)
    predicted_states, predicted_frames = world_model.simulate_trajectory(
        initial_obs, actions[:len(real_frames)-1], return_images=True
    )
    
    # Visualize
    visualize_trajectory_comparison(
        real_frames[:11], predicted_frames[:11],
        output_dir / 'trajectory_comparison.png',
        num_frames=11
    )
    
    # Compute metrics
    real_tensors = [f.unsqueeze(0).to(device) for f in real_frames]
    metrics = compute_prediction_metrics(real_tensors, predicted_frames)
    
    print("Prediction Metrics:")
    for m in metrics[:5]:
        print(f"  Step {m['step']}: MSE={m['mse']:.6f}, PSNR={m['psnr']:.2f} dB")
    print(f"  ...")
    if len(metrics) > 5:
        print(f"  Step {metrics[-1]['step']}: MSE={metrics[-1]['mse']:.6f}, PSNR={metrics[-1]['psnr']:.2f} dB")
    
    # Test 2: Multiple random trajectories
    print("\nTest 2: Testing on 10 random trajectories...")
    all_metrics = []
    
    for episode in range(10):
        obs, _ = env.reset()
        real_frames = [obs]
        actions = []
        
        for _ in range(10):
            action = env.action_space.sample()
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            real_frames.append(obs)
            
            if terminated or truncated:
                break
        
        initial_obs = real_frames[0].unsqueeze(0).to(device)
        _, predicted_frames = world_model.simulate_trajectory(
            initial_obs, actions[:len(real_frames)-1], return_images=True
        )
        
        real_tensors = [f.unsqueeze(0).to(device) for f in real_frames]
        metrics = compute_prediction_metrics(real_tensors, predicted_frames)
        all_metrics.extend(metrics)
    
    # Aggregate metrics by step
    step_metrics = {}
    for m in all_metrics:
        step = m['step']
        if step not in step_metrics:
            step_metrics[step] = {'mse': [], 'psnr': []}
        step_metrics[step]['mse'].append(m['mse'])
        step_metrics[step]['psnr'].append(m['psnr'])
    
    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    steps = sorted(step_metrics.keys())
    mse_means = [np.mean(step_metrics[s]['mse']) for s in steps]
    mse_stds = [np.std(step_metrics[s]['mse']) for s in steps]
    psnr_means = [np.mean(step_metrics[s]['psnr']) for s in steps]
    psnr_stds = [np.std(step_metrics[s]['psnr']) for s in steps]
    
    ax1.plot(steps, mse_means, 'b-', linewidth=2)
    ax1.fill_between(steps, 
                     [m - s for m, s in zip(mse_means, mse_stds)],
                     [m + s for m, s in zip(mse_means, mse_stds)],
                     alpha=0.3)
    ax1.set_xlabel('Prediction Step', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('Prediction MSE vs Step', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps, psnr_means, 'g-', linewidth=2)
    ax2.fill_between(steps,
                     [m - s for m, s in zip(psnr_means, psnr_stds)],
                     [m + s for m, s in zip(psnr_means, psnr_stds)],
                     alpha=0.3)
    ax2.set_xlabel('Prediction Step', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('Prediction PSNR vs Step', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'integrated_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved integrated metrics to {output_dir / 'integrated_metrics.png'}")
    
    # Print summary
    print("\nIntegrated World Model Summary:")
    print(f"  Average 1-step PSNR: {psnr_means[1]:.2f} ± {psnr_stds[1]:.2f} dB")
    print(f"  Average 5-step PSNR: {psnr_means[min(5, len(psnr_means)-1)]:.2f} ± {psnr_stds[min(5, len(psnr_stds)-1)]:.2f} dB")
    if len(psnr_means) > 10:
        print(f"  Average 10-step PSNR: {psnr_means[10]:.2f} ± {psnr_stds[10]:.2f} dB")
    
    # Save summary
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("Integrated World Model Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write("Models:\n")
        f.write("  VAE: outputs/vae_full_training/best_model.pt\n")
        f.write("  Transition: outputs/transition_full_training/best_model.pt\n\n")
        f.write("Performance:\n")
        for i, (step, mse_m, mse_s, psnr_m, psnr_s) in enumerate(
            zip(steps, mse_means, mse_stds, psnr_means, psnr_stds)
        ):
            if i < 5 or i == len(steps) - 1:
                f.write(f"  Step {step}: MSE={mse_m:.6f}±{mse_s:.6f}, PSNR={psnr_m:.2f}±{psnr_s:.2f} dB\n")
            elif i == 5:
                f.write("  ...\n")
    
    print(f"\nAll outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
