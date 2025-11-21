#!/usr/bin/env python3
"""
Train Transition Model on Atari latent space.

This script trains a transition model to learn temporal dynamics:
z_{t+1} = f(z_t, a_t)

The model learns to predict the next latent state given the current
latent state and action, enabling model-based planning.

Usage:
    python src/experiments/train_atari_transition.py \
        --vae_path outputs/vae_full_training/best_model.pt \
        --env_name Breakout \
        --epochs 50
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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


class TransitionDataset(Dataset):
    """Dataset of (z_t, action, z_next) tuples for transition learning."""
    
    def __init__(self, transitions: List[Dict]):
        """
        Args:
            transitions: List of dicts with keys 'z_t', 'action', 'z_next'
        """
        self.z_t = torch.stack([t['z_t'] for t in transitions])
        self.actions = torch.tensor([t['action'] for t in transitions], dtype=torch.long)
        self.z_next = torch.stack([t['z_next'] for t in transitions])
    
    def __len__(self):
        return len(self.z_t)
    
    def __getitem__(self, idx):
        return {
            'z_t': self.z_t[idx],
            'action': self.actions[idx],
            'z_next': self.z_next[idx]
        }


def collect_transitions(
    env_name: str,
    vae: VAE,
    num_episodes: int = 100,
    max_steps: int = 1000,
    device: str = 'cpu'
) -> List[Dict]:
    """
    Collect (z_t, action, z_next) transitions using trained VAE.
    
    Args:
        env_name: Name of Atari environment
        vae: Trained VAE model
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        device: Device to use
    
    Returns:
        List of transition dictionaries
    """
    env = AtariPixelEnv(env_id=env_name, device=device)
    vae.eval()
    transitions = []
    
    print(f"Collecting {num_episodes} episodes of transitions...")
    for ep in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        
        # Encode initial observation
        with torch.no_grad():
            obs_batch = obs.unsqueeze(0).to(device)
            _, z_t, _ = vae(obs_batch)
            z_t = z_t.squeeze(0)
        
        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()
            
            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Encode next observation
            with torch.no_grad():
                next_obs_batch = next_obs.unsqueeze(0).to(device)
                _, z_next, _ = vae(next_obs_batch)
                z_next = z_next.squeeze(0)
            
            # Store transition
            transitions.append({
                'z_t': z_t.cpu(),
                'action': int(action),
                'z_next': z_next.cpu()
            })
            
            # Update current state
            z_t = z_next
            
            if terminated or truncated:
                break
    
    print(f"Collected {len(transitions)} transitions")
    return transitions


def train_epoch(
    model: TransitionModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Transition model
        dataloader: Data loader
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
    
    Returns:
        Dictionary of average losses
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        z_t = batch['z_t'].to(device)
        actions = batch['action'].to(device)
        z_next = batch['z_next'].to(device)
        
        # Forward pass
        (z_mu, z_logvar), _ = model(z_t, actions)
        
        # MSE loss on predicted mean
        loss = nn.functional.mse_loss(z_mu, z_next)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': total_loss / num_batches})
    
    return {'loss': total_loss / num_batches}


@torch.no_grad()
def evaluate(
    model: TransitionModel,
    dataloader: DataLoader,
    device: str
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    
    Args:
        model: Transition model
        dataloader: Data loader
        device: Device to use
    
    Returns:
        Dictionary of average losses
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        z_t = batch['z_t'].to(device)
        actions = batch['action'].to(device)
        z_next = batch['z_next'].to(device)
        
        # Forward pass
        (z_mu, z_logvar), _ = model(z_t, actions)
        
        # MSE loss
        loss = nn.functional.mse_loss(z_mu, z_next)
        
        # Accumulate
        total_loss += loss.item()
        num_batches += 1
    
    return {'loss': total_loss / num_batches}


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str
):
    """
    Plot training curves.
    
    Args:
        history: Dictionary of losses over epochs
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history['train_loss'], label='Train', linewidth=2)
    ax.plot(history['val_loss'], label='Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Transition Model Training', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Transition Model on Atari')
    parser.add_argument('--vae_path', type=str, required=True,
                       help='Path to trained VAE model')
    parser.add_argument('--env_name', type=str, default='Breakout',
                       help='Atari environment name')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension for transition model')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of episodes to collect')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/mps/auto)')
    parser.add_argument('--output_dir', type=str, default='outputs/transition_training',
                       help='Output directory')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                       help='Checkpoint frequency (epochs)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
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
    
    # Load trained VAE
    print(f"Loading VAE from {args.vae_path}...")
    vae = VAE(input_shape=(3, 64, 64), latent_dim=args.latent_dim).to(device)
    vae.load_state_dict(torch.load(args.vae_path, weights_only=True, map_location=device))
    vae.eval()
    print("VAE loaded successfully!")
    
    # Collect transitions
    transitions = collect_transitions(
        env_name=args.env_name,
        vae=vae,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        device=device
    )
    
    # Split into train/val
    num_val = int(len(transitions) * args.val_split)
    indices = np.random.permutation(len(transitions))
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]
    
    train_transitions = [transitions[i] for i in train_indices]
    val_transitions = [transitions[i] for i in val_indices]
    print(f"Train: {len(train_transitions)} transitions, Val: {len(val_transitions)} transitions")
    
    # Create datasets and dataloaders
    train_dataset = TransitionDataset(train_transitions)
    val_dataset = TransitionDataset(val_transitions)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Get action_dim from environment
    env = AtariPixelEnv(env_id=args.env_name, device=device)
    action_dim = env.action_space.n
    print(f"Action space: {action_dim} actions")
    
    # Initialize transition model
    model = TransitionModel(
        latent_dim=args.latent_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        # Evaluate
        val_losses = evaluate(model, val_loader, device)
        
        # Record history
        history['train_loss'].append(train_losses['loss'])
        history['val_loss'].append(val_losses['loss'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_losses['loss']:.6f}")
        print(f"  Val Loss: {val_losses['loss']:.6f}")
        
        # Save best model
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            torch.save(
                model.state_dict(),
                output_dir / 'best_model.pt'
            )
            print(f"  Saved best model (val_loss: {best_val_loss:.6f})")
        
        # Save checkpoint
        if epoch % args.checkpoint_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'args': vars(args)
            }
            torch.save(
                checkpoint,
                output_dir / f'checkpoint_epoch_{epoch}.pt'
            )
            print(f"  Saved checkpoint at epoch {epoch}")
    
    # Training complete
    elapsed = time.time() - start_time
    print(f"\nTraining complete! Total time: {elapsed/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Save final model
    torch.save(
        model.state_dict(),
        output_dir / 'final_model.pt'
    )
    print(f"Saved final model to {output_dir / 'final_model.pt'}")
    
    # Plot training curves
    plot_training_curves(history, output_dir / 'training_curves.png')
    
    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write(f"VAE Model: {args.vae_path}\n")
        f.write(f"Environment: {args.env_name}\n")
        f.write(f"Num Episodes: {args.num_episodes}\n")
        f.write(f"Num Transitions: {len(transitions)}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Training Time: {elapsed/60:.1f} minutes\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"  Best Val Loss: {best_val_loss:.6f}\n")
        f.write(f"  Final Train Loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"  Final Val Loss: {history['val_loss'][-1]:.6f}\n")
    
    print(f"\nAll outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
