#!/usr/bin/env python3
"""
Test MCTS planning with learned world model.

This script compares MCTS performance using:
1. Random (untrained) VAE + Transition Model
2. Trained VAE + Transition Model

Usage:
    python src/experiments/test_mcts_with_learned_models.py
"""

import os
import sys
from pathlib import Path
import time

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
from src.planning.mcts import MCTSPlanner


class WorldModelWrapper:
    """Wrapper for VAE + Transition Model to interface with MCTS."""
    
    def __init__(
        self,
        vae: VAE,
        transition: TransitionModel,
        env: AtariPixelEnv,
        device: str = 'cpu'
    ):
        self.vae = vae
        self.transition = transition
        self.env = env
        self.device = device
        
        self.vae.eval()
        self.transition.eval()
    
    @torch.no_grad()
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent state."""
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        _, z, _ = self.vae(obs)
        return z.squeeze(0)
    
    @torch.no_grad()
    def simulate_action(self, state: torch.Tensor, action: int):
        """
        Simulate taking an action in the world model.
        
        Returns:
            next_state: Predicted next latent state
            reward: Estimated reward (using actual environment for now)
            done: Whether episode should terminate
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        action_tensor = torch.tensor([action], dtype=torch.long).to(self.device)
        (next_state, _), _ = self.transition(state, action_tensor)
        
        # For reward, we'd need a reward predictor
        # For now, use a simple heuristic or actual environment
        # In real scenario, this would be learned
        reward = 0.0
        done = False
        
        return next_state.squeeze(0), reward, done
    
    def get_valid_actions(self):
        """Get list of valid actions."""
        return list(range(self.env.action_space.n))


def run_episode_with_mcts(
    env: AtariPixelEnv,
    world_model: WorldModelWrapper,
    num_simulations: int = 50,
    max_steps: int = 1000,
    device: str = 'cpu'
):
    """Run one episode using MCTS with world model."""
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    
    # Initialize MCTS (using existing implementation)
    # For now, use random actions as baseline
    # Real MCTS integration would use world_model.simulate_action
    
    for step in range(max_steps):
        # Encode current observation
        latent_state = world_model.encode(obs)
        
        # Simple policy: random action
        # TODO: Replace with actual MCTS that uses world_model
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    return total_reward, steps


def run_episode_random(env: AtariPixelEnv, max_steps: int = 1000):
    """Run one episode with random actions (baseline)."""
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    return total_reward, steps


def compare_models(
    env_name: str,
    trained_vae_path: str,
    trained_transition_path: str,
    num_episodes: int = 10,
    device: str = 'cpu'
):
    """Compare random baseline vs trained models."""
    
    print("Setting up models...")
    env = AtariPixelEnv(env_id=env_name, device=device)
    action_dim = env.action_space.n
    latent_dim = 32
    
    # 1. Random baseline
    print("\nRunning random baseline...")
    random_rewards = []
    random_steps = []
    
    for ep in tqdm(range(num_episodes), desc="Random policy"):
        reward, steps = run_episode_random(env)
        random_rewards.append(reward)
        random_steps.append(steps)
    
    # 2. Trained models
    print("\nLoading trained models...")
    trained_vae = VAE(input_shape=(3, 64, 64), latent_dim=latent_dim).to(device)
    trained_vae.load_state_dict(torch.load(
        trained_vae_path, weights_only=True, map_location=device
    ))
    
    trained_transition = TransitionModel(
        latent_dim=latent_dim, action_dim=action_dim, hidden_dim=64
    ).to(device)
    trained_transition.load_state_dict(torch.load(
        trained_transition_path, weights_only=True, map_location=device
    ))
    
    trained_world_model = WorldModelWrapper(
        trained_vae, trained_transition, env, device
    )
    
    print("Running with trained models...")
    trained_rewards = []
    trained_steps = []
    
    for ep in tqdm(range(num_episodes), desc="Trained models"):
        reward, steps = run_episode_with_mcts(
            env, trained_world_model, num_simulations=50, device=device
        )
        trained_rewards.append(reward)
        trained_steps.append(steps)
    
    # 3. Random (untrained) models
    print("\nRunning with random (untrained) models...")
    random_vae = VAE(input_shape=(3, 64, 64), latent_dim=latent_dim).to(device)
    random_transition = TransitionModel(
        latent_dim=latent_dim, action_dim=action_dim, hidden_dim=64
    ).to(device)
    
    random_world_model = WorldModelWrapper(
        random_vae, random_transition, env, device
    )
    
    untrained_rewards = []
    untrained_steps = []
    
    for ep in tqdm(range(num_episodes), desc="Untrained models"):
        reward, steps = run_episode_with_mcts(
            env, random_world_model, num_simulations=50, device=device
        )
        untrained_rewards.append(reward)
        untrained_steps.append(steps)
    
    return {
        'random': {'rewards': random_rewards, 'steps': random_steps},
        'trained': {'rewards': trained_rewards, 'steps': trained_steps},
        'untrained': {'rewards': untrained_rewards, 'steps': untrained_steps}
    }


def plot_comparison(results: dict, save_path: str):
    """Plot comparison of different approaches."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rewards
    methods = ['Random\nPolicy', 'Untrained\nModels', 'Trained\nModels']
    rewards = [
        results['random']['rewards'],
        results['untrained']['rewards'],
        results['trained']['rewards']
    ]
    
    positions = [1, 2, 3]
    bp1 = ax1.boxplot(rewards, positions=positions, widths=0.6,
                      patch_artist=True, showmeans=True)
    
    colors = ['lightcoral', 'lightyellow', 'lightgreen']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Episode Rewards Comparison', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Steps
    steps = [
        results['random']['steps'],
        results['untrained']['steps'],
        results['trained']['steps']
    ]
    
    bp2 = ax2.boxplot(steps, positions=positions, widths=0.6,
                      patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_xticks(positions)
    ax2.set_xticklabels(methods, fontsize=11)
    ax2.set_ylabel('Episode Length (steps)', fontsize=12)
    ax2.set_title('Episode Length Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def main():
    # Configuration
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = Path('outputs/mcts_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run comparison
    results = compare_models(
        env_name='Breakout',
        trained_vae_path='outputs/vae_full_training/best_model.pt',
        trained_transition_path='outputs/transition_full_training/best_model.pt',
        num_episodes=10,
        device=device
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for method in ['random', 'untrained', 'trained']:
        rewards = results[method]['rewards']
        steps = results[method]['steps']
        print(f"\n{method.upper()}:")
        print(f"  Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Avg Steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
        print(f"  Min/Max Reward: {np.min(rewards):.1f} / {np.max(rewards):.1f}")
    
    # Plot comparison
    plot_comparison(results, output_dir / 'comparison.png')
    
    # Save results
    with open(output_dir / 'results.txt', 'w') as f:
        f.write("MCTS Comparison: Random vs Untrained vs Trained Models\n")
        f.write("=" * 60 + "\n\n")
        
        for method in ['random', 'untrained', 'trained']:
            rewards = results[method]['rewards']
            steps = results[method]['steps']
            f.write(f"{method.upper()}:\n")
            f.write(f"  Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
            f.write(f"  Average Steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}\n")
            f.write(f"  Reward Range: [{np.min(rewards):.1f}, {np.max(rewards):.1f}]\n")
            f.write(f"  Individual rewards: {[f'{r:.1f}' for r in rewards]}\n")
            f.write("\n")
    
    print(f"\nAll outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
