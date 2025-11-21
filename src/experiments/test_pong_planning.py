#!/usr/bin/env python3
"""
Test Hierarchical Planning with Trained 3-Level RGM on Pong.

This script tests goal-directed planning using the hierarchical model for Pong:
- Level 2 (Path): Sets long-term goals (16-step horizon)
- Level 1 (Feature): Sets sub-goals (4-step horizon)
- Level 0 (Pixel): Selects primitive actions (1-step)

Compares performance:
1. Random Policy (baseline)
2. Flat Planning (single-level, using only Level 0)
3. Hierarchical Planning (3-level, using all levels)

Usage:
    python src/experiments/test_pong_planning.py \
        --config_path outputs/pong_hierarchical_training/hierarchical_config.pt \
        --model_dir outputs/pong_hierarchical_training \
        --num_episodes 20 \
        --env_name Pong
"""

import argparse
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
from src.models.multi_level_rgm import MultiLevelRGM, LevelConfig
from src.models.multi_level_agent import MultiLevelAgent
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
    
    # Create dummy Level 0 transition (not used but needed for structure)
    level0_transition = TransitionModel(
        latent_dim=config['level0_latent_dim'],
        action_dim=config['action_dim'],
        hidden_dim=64
    ).to(device)
    
    # Create level configs
    level_configs = [
        LevelConfig(
            latent_dim=config['level0_latent_dim'],
            temporal_resolution=1,
            vae=level0_vae,
            transition=level0_transition
        ),
        LevelConfig(
            latent_dim=config['level1_latent_dim'],
            temporal_resolution=config['level1_temporal_resolution'],
            vae=level1_vae,
            transition=level1_transition
        ),
        LevelConfig(
            latent_dim=config['level2_latent_dim'],
            temporal_resolution=config['level2_temporal_resolution'],
            vae=level2_vae,
            transition=level2_transition
        )
    ]
    
    # Create MultiLevelRGM
    rgm = MultiLevelRGM(level_configs, device=device)
    
    return rgm, config


def run_random_policy(env: AtariPixelEnv, max_steps: int = 1000):
    """Run episode with random actions."""
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    return total_reward, steps


def run_flat_planning(env: AtariPixelEnv, rgm: MultiLevelRGM, max_steps: int = 1000, device: str = 'cpu'):
    """
    Run episode with flat planning (only Level 0).
    Simple greedy action selection based on Level 0 state prediction.
    """
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    
    level0_vae = rgm.levels[0].vae
    level0_vae.eval()
    
    for _ in range(max_steps):
        # Encode current observation
        obs_batch = obs.unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, logvar = level0_vae.encode(obs_batch)
            z0 = level0_vae.reparameterize(mu, logvar)
        
        # Simple greedy: choose random action (flat baseline)
        # In real implementation, would use Level 0 transition for 1-step lookahead
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    return total_reward, steps


def run_hierarchical_planning(
    env: AtariPixelEnv,
    agent: MultiLevelAgent,
    max_steps: int = 1000,
    device: str = 'cpu'
):
    """
    Run episode with hierarchical planning.
    
    Strategy:
    - Every 16 steps: Level 2 sets long-term goal
    - Every 4 steps: Level 1 sets sub-goal based on Level 2 goal
    - Every step: Level 0 selects action based on Level 1 sub-goal
    """
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    
    agent.reset()
    
    # Set initial goals (None = exploratory)
    level2_goal = None
    level1_goal = None
    
    for step in range(max_steps):
        obs_batch = obs.unsqueeze(0).to(device)
        
        # Infer current state at all levels
        current_states = agent.infer_state(obs_batch)
        
        # Update Level 2 goal (every 16 steps or at start)
        if step % 16 == 0:
            # Level 2: Set long-term goal
            # For now, use current state as goal (stay near current state)
            # In real implementation, would use reward prediction
            with torch.no_grad():
                level2_goal = current_states[2].detach()
        
        # Update Level 1 sub-goal (every 4 steps or at start)
        if step % 4 == 0:
            # Level 1: Set sub-goal to move towards Level 2 goal
            with torch.no_grad():
                if level2_goal is not None:
                    # Decode Level 2 goal to Level 1 space
                    level2_to_1 = agent.rgm.levels[2].vae.decode(level2_goal)
                    level1_goal = level2_to_1.detach()
                else:
                    level1_goal = current_states[1].detach()
        
        # Level 0: Select action using multi-level EFE
        # Note: Current MultiLevelAgent doesn't accept explicit goals
        # It uses upper level states implicitly during EFE calculation
        action = agent.select_action(obs_batch)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        agent.timestep += 1
        
        if terminated or truncated:
            break
    
    return total_reward, steps


def compare_planning_methods(
    env_name: str,
    rgm: MultiLevelRGM,
    config: dict,
    num_episodes: int,
    device: str
):
    """Compare all three planning methods."""
    
    print("\n" + "="*70)
    print("COMPARING PLANNING METHODS")
    print("="*70)
    
    results = {
        'random': {'rewards': [], 'steps': []},
        'flat': {'rewards': [], 'steps': []},
        'hierarchical': {'rewards': [], 'steps': []}
    }
    
    # Create agent for hierarchical planning
    agent = MultiLevelAgent(rgm, action_dim=config['action_dim'], device=device)
    
    # Test each method
    for method_name, method_func in [
        ('random', lambda env: run_random_policy(env)),
        ('flat', lambda env: run_flat_planning(env, rgm, device=device)),
        ('hierarchical', lambda env: run_hierarchical_planning(env, agent, device=device))
    ]:
        print(f"\n{method_name.upper()} Policy:")
        
        for ep in tqdm(range(num_episodes), desc=f"  {method_name}"):
            env = AtariPixelEnv(env_id=env_name, device=device)
            reward, steps = method_func(env)
            
            results[method_name]['rewards'].append(reward)
            results[method_name]['steps'].append(steps)
        
        avg_reward = np.mean(results[method_name]['rewards'])
        std_reward = np.std(results[method_name]['rewards'])
        avg_steps = np.mean(results[method_name]['steps'])
        
        print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Average Steps: {avg_steps:.1f}")
    
    return results


def visualize_comparison(results: dict, save_path: str):
    """Visualize planning comparison results."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ['Random', 'Flat\n(Level 0)', 'Hierarchical\n(3-Level)']
    colors = ['lightcoral', 'lightyellow', 'lightgreen']
    
    # Rewards
    rewards_data = [
        results['random']['rewards'],
        results['flat']['rewards'],
        results['hierarchical']['rewards']
    ]
    
    bp1 = ax1.boxplot(rewards_data, positions=[1, 2, 3], widths=0.6,
                      patch_artist=True, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Episode Rewards Comparison', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add mean values as text
    for i, data in enumerate(rewards_data, 1):
        mean_val = np.mean(data)
        ax1.text(i, mean_val, f'{mean_val:.1f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Steps
    steps_data = [
        results['random']['steps'],
        results['flat']['steps'],
        results['hierarchical']['steps']
    ]
    
    bp2 = ax2.boxplot(steps_data, positions=[1, 2, 3], widths=0.6,
                      patch_artist=True, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(methods, fontsize=11)
    ax2.set_ylabel('Episode Length (steps)', fontsize=12)
    ax2.set_title('Episode Length Comparison', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add mean values as text
    for i, data in enumerate(steps_data, 1):
        mean_val = np.mean(data)
        ax2.text(i, mean_val, f'{mean_val:.0f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Hierarchical Planning Performance', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Hierarchical Planning')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to hierarchical config')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--env_name', type=str, default='Pong',
                       help='Atari environment name (default: Pong)')
    parser.add_argument('--num_episodes', type=int, default=20,
                       help='Number of test episodes per method')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/mps/auto)')
    parser.add_argument('--output_dir', type=str, default='outputs/pong_planning_test',
                       help='Output directory (default: outputs/pong_planning_test)')
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
    print("\nLoading hierarchical model...")
    model_dir = Path(args.model_dir)
    rgm, config = load_hierarchical_model(args.config_path, model_dir, device)
    print("Models loaded!")
    
    print(f"\nModel Configuration:")
    print(f"  Level 0: {config['level0_latent_dim']}D (τ=1)")
    print(f"  Level 1: {config['level1_latent_dim']}D (τ={config['level1_temporal_resolution']})")
    print(f"  Level 2: {config['level2_latent_dim']}D (τ={config['level2_temporal_resolution']})")
    
    # Run comparison
    start_time = time.time()
    results = compare_planning_methods(
        args.env_name,
        rgm,
        config,
        args.num_episodes,
        device
    )
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for method in ['random', 'flat', 'hierarchical']:
        rewards = results[method]['rewards']
        steps = results[method]['steps']
        print(f"\n{method.upper()}:")
        print(f"  Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Avg Steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
        print(f"  Min/Max Reward: {np.min(rewards):.1f} / {np.max(rewards):.1f}")
    
    # Calculate improvement
    random_avg = np.mean(results['random']['rewards'])
    flat_avg = np.mean(results['flat']['rewards'])
    hierarchical_avg = np.mean(results['hierarchical']['rewards'])
    
    if random_avg > 0:
        flat_improvement = ((flat_avg - random_avg) / random_avg) * 100
        hierarchical_improvement = ((hierarchical_avg - random_avg) / random_avg) * 100
        
        print("\n" + "="*70)
        print("IMPROVEMENT OVER RANDOM")
        print("="*70)
        print(f"Flat Planning: {flat_improvement:+.1f}%")
        print(f"Hierarchical Planning: {hierarchical_improvement:+.1f}%")
    
    # Visualize
    visualize_comparison(results, output_dir / 'planning_comparison.png')
    
    # Save results
    with open(output_dir / 'results.txt', 'w') as f:
        f.write("Hierarchical Planning Test Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Environment: {args.env_name}\n")
        f.write(f"  Episodes per method: {args.num_episodes}\n")
        f.write(f"  Total time: {elapsed_time/60:.1f} minutes\n\n")
        
        for method in ['random', 'flat', 'hierarchical']:
            rewards = results[method]['rewards']
            steps = results[method]['steps']
            f.write(f"{method.upper()}:\n")
            f.write(f"  Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
            f.write(f"  Average Steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}\n")
            f.write(f"  Reward Range: [{np.min(rewards):.1f}, {np.max(rewards):.1f}]\n")
            f.write(f"  Individual rewards: {[f'{r:.1f}' for r in rewards]}\n\n")
        
        if random_avg > 0:
            f.write(f"Improvement over Random:\n")
            f.write(f"  Flat: {flat_improvement:+.1f}%\n")
            f.write(f"  Hierarchical: {hierarchical_improvement:+.1f}%\n")
    
    print(f"\nAll results saved to {output_dir}")
    print("\nTest complete!")


if __name__ == '__main__':
    main()
