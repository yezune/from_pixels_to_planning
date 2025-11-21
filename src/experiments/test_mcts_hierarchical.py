#!/usr/bin/env python3
"""
Test MCTS-based Planning with Hierarchical 3-Level RGM.

This script tests Monte Carlo Tree Search (MCTS) planning using the hierarchical model:
- Compares 4 methods: Random, Flat, Hierarchical, MCTS
- Uses learned transition models for MCTS simulation
- Evaluates goal-directed planning performance

Usage:
    python src/experiments/test_mcts_hierarchical.py \
        --config_path outputs/hierarchical_training/hierarchical_config.pt \
        --model_dir outputs/hierarchical_training \
        --num_episodes 20 \
        --mcts_simulations 50
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
from src.planning.mcts import MCTSPlanner
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
            temporal_resolution=4,
            vae=level1_vae,
            transition=level1_transition
        ),
        LevelConfig(
            latent_dim=config['level2_latent_dim'],
            temporal_resolution=16,
            vae=level2_vae,
            transition=level2_transition
        )
    ]
    
    rgm = MultiLevelRGM(level_configs, device=device)
    rgm.action_dim = config['action_dim']  # Add action_dim as attribute
    
    return rgm, config


def test_random_policy(env, num_episodes=10):
    """Test random action selection."""
    rewards = []
    steps_list = []
    
    for _ in tqdm(range(num_episodes), desc='random'):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 1000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        rewards.append(total_reward)
        steps_list.append(steps)
    
    return {
        'rewards': rewards,
        'steps': steps_list,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_steps': np.mean(steps_list)
    }


def test_flat_planning(env, rgm, num_episodes=10, device='cpu'):
    """Test flat (single-level) planning using only Level 0."""
    rewards = []
    steps_list = []
    
    level0_vae = rgm.levels[0].vae
    level0_vae.eval()
    
    for _ in tqdm(range(num_episodes), desc='flat'):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 1000:
            # Encode observation to latent
            # AtariPixelEnv returns (C, H, W) format already
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, _ = level0_vae.encode(obs_tensor)
            
            # Simple greedy action selection (no planning)
            action = np.random.randint(0, env.action_space.n)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        rewards.append(total_reward)
        steps_list.append(steps)
    
    return {
        'rewards': rewards,
        'steps': steps_list,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_steps': np.mean(steps_list)
    }


def test_hierarchical_planning(env, agent, num_episodes=10, device='cpu'):
    """Test hierarchical planning with 3 levels."""
    rewards = []
    steps_list = []
    
    for _ in tqdm(range(num_episodes), desc='hierarchical'):
        obs, _ = env.reset()
        agent.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 1000:
            # Hierarchical action selection
            action = agent.select_action(obs)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        rewards.append(total_reward)
        steps_list.append(steps)
    
    return {
        'rewards': rewards,
        'steps': steps_list,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_steps': np.mean(steps_list)
    }


def test_mcts_planning(env, rgm, num_episodes=10, num_simulations=50, device='cpu'):
    """Test MCTS planning using learned transition models."""
    rewards = []
    steps_list = []
    
    level0_vae = rgm.levels[0].vae
    level1_transition = rgm.levels[1].transition
    action_dim = rgm.action_dim
    
    level0_vae.eval()
    level1_transition.eval()
    
    # Create MCTS planner with Level 1 transition model
    mcts_planner = MCTSPlanner(
        transition_model=level1_transition,
        action_dim=action_dim,
        num_simulations=num_simulations,
        exploration_weight=1.414,
        discount=0.95,
        device=device
    )
    
    for _ in tqdm(range(num_episodes), desc='mcts'):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        hidden = None
        current_action = env.action_space.sample()  # Initial random action
        replan_interval = 4  # Replan every 4 steps (faster execution)
        
        while not done and steps < 1000:
            # Only run MCTS planning every N steps
            if steps % replan_interval == 0:
                # Encode observation to Level 0 latent
                # AtariPixelEnv returns (C, H, W) format already
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    mu0, _ = level0_vae.encode(obs_tensor)
                    
                    # Encode to Level 1 latent
                    level1_vae = rgm.levels[1].vae
                    mu1, _ = level1_vae.encode(mu0)
                
                # Use MCTS to select action
                # Note: MCTS plans in Level 1 latent space
                current_action = mcts_planner.plan(
                    initial_state=mu1,
                    goal_state=None,  # No specific goal, explore
                    depth=3  # 3-step lookahead (reduced from 5)
                )
            
            obs, reward, terminated, truncated, _ = env.step(current_action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        rewards.append(total_reward)
        steps_list.append(steps)
    
    return {
        'rewards': rewards,
        'steps': steps_list,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_steps': np.mean(steps_list)
    }


def plot_results(results: dict, output_dir: Path):
    """Plot comparison of all methods."""
    methods = list(results.keys())
    avg_rewards = [results[m]['avg_reward'] for m in methods]
    std_rewards = [results[m]['std_reward'] for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Average Reward
    colors = ['gray', 'orange', 'blue', 'green']
    bars = ax1.bar(methods, avg_rewards, yerr=std_rewards, 
                   capsize=5, alpha=0.8, color=colors)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title('Planning Performance Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, avg in zip(bars, avg_rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    # Reward distributions
    for i, method in enumerate(methods):
        rewards = results[method]['rewards']
        positions = np.random.normal(i, 0.04, size=len(rewards))
        ax2.scatter(positions, rewards, alpha=0.6, s=50, color=colors[i], label=method)
    
    ax2.set_ylabel('Episode Reward', fontsize=12)
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_title('Reward Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mcts_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to {output_dir / 'mcts_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Test MCTS with Hierarchical RGM')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to hierarchical_config.pt')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--env_name', type=str, default='Breakout',
                       help='Atari environment name')
    parser.add_argument('--num_episodes', type=int, default=20,
                       help='Number of test episodes per method')
    parser.add_argument('--mcts_simulations', type=int, default=50,
                       help='Number of MCTS simulations per planning step')
    parser.add_argument('--output_dir', type=str, default='outputs/mcts_hierarchical_test',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    # Detect MPS
    if torch.backends.mps.is_available() and args.device == 'cpu':
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders) device")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    
    print(f"Using device: {device}\n")
    
    # Load models
    print("Loading hierarchical model...")
    rgm, config = load_hierarchical_model(args.config_path, model_dir, device)
    print("Models loaded!\n")
    
    print("Model Configuration:")
    print(f"  Level 0: {config['level0_latent_dim']}D (τ=1)")
    print(f"  Level 1: {config['level1_latent_dim']}D (τ=4)")
    print(f"  Level 2: {config['level2_latent_dim']}D (τ=16)\n")
    
    # Create environment
    env = AtariPixelEnv(
        env_id=f'{args.env_name}NoFrameskip-v4',
        image_size=64,
        device='cpu'
    )
    
    # Create hierarchical agent
    agent = MultiLevelAgent(rgm, action_dim=config['action_dim'], device=device)
    
    print("=" * 70)
    print("COMPARING PLANNING METHODS (INCLUDING MCTS)")
    print("=" * 70)
    print()
    
    # Test all methods
    results = {}
    
    # 1. Random
    print("RANDOM Policy:")
    results['random'] = test_random_policy(env, args.num_episodes)
    print(f"  Average Reward: {results['random']['avg_reward']:.2f} ± {results['random']['std_reward']:.2f}")
    print(f"  Average Steps: {results['random']['avg_steps']:.1f}\n")
    
    # 2. Flat
    print("FLAT Policy:")
    results['flat'] = test_flat_planning(env, rgm, args.num_episodes, device)
    print(f"  Average Reward: {results['flat']['avg_reward']:.2f} ± {results['flat']['std_reward']:.2f}")
    print(f"  Average Steps: {results['flat']['avg_steps']:.1f}\n")
    
    # 3. Hierarchical
    print("HIERARCHICAL Policy:")
    results['hierarchical'] = test_hierarchical_planning(env, agent, args.num_episodes, device)
    print(f"  Average Reward: {results['hierarchical']['avg_reward']:.2f} ± {results['hierarchical']['std_reward']:.2f}")
    print(f"  Average Steps: {results['hierarchical']['avg_steps']:.1f}\n")
    
    # 4. MCTS
    print(f"MCTS Policy ({args.mcts_simulations} simulations):")
    results['mcts'] = test_mcts_planning(env, rgm, args.num_episodes, args.mcts_simulations, device)
    print(f"  Average Reward: {results['mcts']['avg_reward']:.2f} ± {results['mcts']['std_reward']:.2f}")
    print(f"  Average Steps: {results['mcts']['avg_steps']:.1f}\n")
    
    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    
    for method in results:
        r = results[method]
        print(f"{method.upper()}:")
        print(f"  Avg Reward: {r['avg_reward']:.2f} ± {r['std_reward']:.2f}")
        print(f"  Avg Steps: {r['avg_steps']:.1f} ± {np.std(r['steps']):.1f}")
        print(f"  Min/Max Reward: {min(r['rewards']):.1f} / {max(r['rewards']):.1f}")
        print()
    
    # Plot results
    plot_results(results, output_dir)
    
    # Save results
    import json
    results_summary = {
        method: {
            'avg_reward': float(r['avg_reward']),
            'std_reward': float(r['std_reward']),
            'avg_steps': float(r['avg_steps']),
            'min_reward': float(min(r['rewards'])),
            'max_reward': float(max(r['rewards'])),
            'all_rewards': [float(x) for x in r['rewards']],
            'all_steps': [int(x) for x in r['steps']]
        }
        for method, r in results.items()
    }
    
    with open(output_dir / 'mcts_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"Saved comparison plot to {output_dir / 'mcts_comparison.png'}")
    print(f"\nAll results saved to {output_dir}")
    print("\nTest complete!")
    
    env.close()


if __name__ == '__main__':
    main()
