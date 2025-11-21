#!/usr/bin/env python3
"""
Train VAE on Atari Pong observations.

This script trains a VAE to encode Pong game frames (64x64 RGB) into a compact latent space.
This is adapted from train_atari_vae.py specifically for the Pong environment.

Usage:
    python src/experiments/train_pong_vae.py --num_episodes 100 --epochs 100
    
    # Quick test
    python src/experiments/train_pong_vae.py --num_episodes 10 --epochs 10
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.vae import VAE
from src.envs.atari_env import AtariPixelEnv


class PongFrameDataset(Dataset):
    """Dataset of Pong frames for VAE training."""
    
    def __init__(self, frames: np.ndarray):
        """
        Args:
            frames: (N, 3, 64, 64) numpy array of frames
        """
        self.frames = torch.from_numpy(frames).float()
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        return self.frames[idx]


def collect_pong_episodes(
    num_episodes: int = 100,
    max_steps: int = 1000,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Collect frames by playing random Pong episodes.
    
    Pong has 6 actions: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
    
    Args:
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        device: Device to use
    
    Returns:
        frames: (N, 3, 64, 64) array of collected frames
    """
    env = AtariPixelEnv(env_id='Pong', device=device)
    frames = []
    
    print(f"Collecting {num_episodes} random Pong episodes...")
    print(f"Action space: {env.env.action_space}")
    print(f"Actions: {env.env.unwrapped.get_action_meanings()}")
    
    for ep in tqdm(range(num_episodes), desc="Episodes"):
        obs, _ = env.reset()
        frames.append(obs.cpu().numpy())
        
        for step in range(max_steps):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(obs.cpu().numpy())
            
            if terminated or truncated:
                break
    
    env.close()
    frames = np.stack(frames)
    print(f"Collected {len(frames)} frames from Pong")
    return frames


def train_epoch(
    model: VAE,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        batch = batch.to(device)
        
        # Forward pass
        recon, mu, logvar = model(batch)
        
        # Compute loss
        loss_dict = model.loss_function(recon, batch, mu, logvar)
        loss = loss_dict['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += loss_dict['recon_loss'].item()
        total_kl_loss += loss_dict['kld_loss'].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / num_batches,
            'recon': total_recon_loss / num_batches,
            'kl': total_kl_loss / num_batches
        })
    
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches
    }


@torch.no_grad()
def evaluate(
    model: VAE,
    dataloader: DataLoader,
    device: str
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        batch = batch.to(device)
        
        # Forward pass
        recon, mu, logvar = model(batch)
        
        # Compute loss
        loss_dict = model.loss_function(recon, batch, mu, logvar)
        
        # Accumulate losses
        total_loss += loss_dict['loss'].item()
        total_recon_loss += loss_dict['recon_loss'].item()
        total_kl_loss += loss_dict['kld_loss'].item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches
    }


def visualize_reconstructions(
    model: VAE,
    dataset: Dataset,
    num_samples: int,
    device: str,
    save_path: str
):
    """Visualize original and reconstructed Pong frames."""
    model.eval()
    
    # Sample random frames
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    frames = torch.stack([dataset[i] for i in indices]).to(device)
    
    # Get reconstructions
    with torch.no_grad():
        recon, _, _ = model(frames)
    
    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    for i in range(num_samples):
        # Original
        orig = frames[i].cpu().permute(1, 2, 0).numpy()
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
    print(f"Saved reconstructions to {save_path}")


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str
):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Reconstruction loss
    axes[1].plot(history['train_recon_loss'], label='Train')
    axes[1].plot(history['val_recon_loss'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # KL divergence
    axes[2].plot(history['train_kl_loss'], label='Train')
    axes[2].plot(history['val_kl_loss'], label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train VAE on Pong observations')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension (default: 32)')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of episodes to collect (default: 100)')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split (default: 0.1)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/mps/auto)')
    parser.add_argument('--output_dir', type=str, default='outputs/pong_vae_training',
                       help='Output directory (default: outputs/pong_vae_training)')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
                       help='Checkpoint frequency in epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
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
    print(f"Training VAE for Pong environment")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect Pong data
    frames = collect_pong_episodes(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        device=device
    )
    
    # Split into train/val
    num_val = int(len(frames) * args.val_split)
    indices = np.random.permutation(len(frames))
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]
    
    train_frames = frames[train_indices]
    val_frames = frames[val_indices]
    print(f"Train: {len(train_frames)} frames, Val: {len(val_frames)} frames")
    
    # Create datasets and dataloaders
    train_dataset = PongFrameDataset(train_frames)
    val_dataset = PongFrameDataset(val_frames)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    model = VAE(
        input_shape=(3, 64, 64),
        latent_dim=args.latent_dim
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training history
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': []
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
        history['train_recon_loss'].append(train_losses['recon_loss'])
        history['train_kl_loss'].append(train_losses['kl_loss'])
        history['val_loss'].append(val_losses['loss'])
        history['val_recon_loss'].append(val_losses['recon_loss'])
        history['val_kl_loss'].append(val_losses['kl_loss'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_losses['loss']:.4f} "
              f"(Recon: {train_losses['recon_loss']:.4f}, "
              f"KL: {train_losses['kl_loss']:.4f})")
        print(f"  Val Loss: {val_losses['loss']:.4f} "
              f"(Recon: {val_losses['recon_loss']:.4f}, "
              f"KL: {val_losses['kl_loss']:.4f})")
        
        # Save best model
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            torch.save(
                model.state_dict(),
                output_dir / 'best_model.pt'
            )
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
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
            print(f"  ✓ Saved checkpoint at epoch {epoch}")
        
        # Visualize reconstructions
        if epoch % args.checkpoint_freq == 0 or epoch == 1:
            visualize_reconstructions(
                model, val_dataset, num_samples=8, device=device,
                save_path=output_dir / f'reconstructions_epoch_{epoch}.png'
            )
    
    # Training complete
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete! Total time: {elapsed/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}")
    
    # Save final model
    torch.save(
        model.state_dict(),
        output_dir / 'final_model.pt'
    )
    print(f"Saved final model to {output_dir / 'final_model.pt'}")
    
    # Plot training curves
    plot_training_curves(history, output_dir / 'training_curves.png')
    
    # Visualize final reconstructions
    visualize_reconstructions(
        model, val_dataset, num_samples=8, device=device,
        save_path=output_dir / 'final_reconstructions.png'
    )
    
    print(f"\nAll outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
