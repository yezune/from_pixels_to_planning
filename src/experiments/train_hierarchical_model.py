#!/usr/bin/env python3
"""
Train 3-Level Hierarchical RGM on Atari.

This script implements full hierarchical training with:
- Level 0: Pixel-level representation (VAE already trained)
- Level 1: Feature-level abstraction (temporal resolution τ=4)
- Level 2: Path-level abstraction (temporal resolution τ=16)

Training Strategy:
1. Load pre-trained Level 0 VAE
2. Train Level 1 VAE on Level 0 latents
3. Train Level 1 Transition Model
4. Train Level 2 VAE on Level 1 latents
5. Train Level 2 Transition Model
6. Fine-tune entire hierarchy end-to-end

Usage:
    python src/experiments/train_hierarchical_model.py \
        --level0_vae_path outputs/vae_full_training/best_model.pt \
        --env_name Breakout \
        --num_episodes 100 \
        --output_dir outputs/hierarchical_training
"""

import argparse
import os
import sys
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.vae import VAE
from src.models.base_vae import BaseVAE
from src.models.transition import TransitionModel
from src.models.multi_level_rgm import MultiLevelRGM, LevelConfig
from src.envs.atari_env import AtariPixelEnv


class SimpleVAE(BaseVAE):
    """Simple VAE for higher levels (operates on latent vectors, not images)."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x: torch.Tensor):
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z: torch.Tensor):
        """Decode latent to reconstruction."""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor):
        """Full forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, z, (mu, logvar)


def collect_hierarchical_data(
    env_name: str,
    level0_vae: VAE,
    num_episodes: int,
    max_steps: int,
    device: str
):
    """
    Collect hierarchical data:
    - Level 0: pixel observations
    - Level 0 latents for training Level 1
    - Actions and transitions
    """
    env = AtariPixelEnv(env_id=env_name, device=device)
    level0_vae.eval()
    
    # Storage
    observations = []
    level0_latents = []
    actions = []
    
    print(f"Collecting {num_episodes} episodes...")
    for ep in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        
        for step in range(max_steps):
            # Encode observation to Level 0 latent
            with torch.no_grad():
                obs_batch = obs.unsqueeze(0).to(device)
                mu, logvar = level0_vae.encode(obs_batch)
                z0 = level0_vae.reparameterize(mu, logvar)
            
            observations.append(obs.cpu())
            level0_latents.append(z0.squeeze(0).cpu())
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            
            if terminated or truncated:
                break
    
    print(f"Collected {len(observations)} frames")
    
    return {
        'observations': torch.stack(observations),
        'level0_latents': torch.stack(level0_latents),
        'actions': torch.tensor(actions, dtype=torch.long)
    }


def train_vae_on_latents(
    vae: SimpleVAE,
    latents: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    output_dir: Path,
    level_name: str
):
    """Train VAE on latent representations from lower level."""
    print(f"\nTraining {level_name} VAE...")
    
    # Create dataset
    dataset = TensorDataset(latents)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        # Train
        vae.train()
        train_loss = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            
            recon, z, (mu, logvar) = vae(batch)
            
            # VAE loss
            recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (recon_loss + kl_loss) / batch.size(0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, z, (mu, logvar) = vae(batch)
                recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (recon_loss + kl_loss) / batch.size(0)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(vae.state_dict(), output_dir / f'{level_name}_vae_best.pt')
    
    # Save final model
    torch.save(vae.state_dict(), output_dir / f'{level_name}_vae_final.pt')
    
    # Plot training curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{level_name} VAE Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'{level_name}_vae_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"{level_name} VAE training complete! Best val loss: {best_val_loss:.4f}")
    return best_val_loss


def train_transition_on_latents(
    transition: TransitionModel,
    latents: torch.Tensor,
    actions: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    output_dir: Path,
    level_name: str,
    temporal_resolution: int = 1
):
    """Train Transition Model on latent transitions."""
    print(f"\nTraining {level_name} Transition Model (τ={temporal_resolution})...")
    
    # Create transition pairs with temporal resolution
    transitions = []
    for i in range(0, len(latents) - temporal_resolution, temporal_resolution):
        z_t = latents[i]
        z_next = latents[i + temporal_resolution]
        # For simplicity, use the first action in the temporal window
        action = actions[i]
        transitions.append((z_t, action, z_next))
    
    print(f"Created {len(transitions)} transition pairs (τ={temporal_resolution})")
    
    # Split train/val
    train_size = int(0.9 * len(transitions))
    train_transitions = transitions[:train_size]
    val_transitions = transitions[train_size:]
    
    # Optimizer
    optimizer = optim.Adam(transition.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        # Train
        transition.train()
        train_loss = 0
        np.random.shuffle(train_transitions)
        
        for i in range(0, len(train_transitions), batch_size):
            batch = train_transitions[i:i+batch_size]
            z_batch = torch.stack([t[0] for t in batch]).to(device)
            action_batch = torch.tensor([t[1] for t in batch], dtype=torch.long).to(device)
            z_next_batch = torch.stack([t[2] for t in batch]).to(device)
            
            (z_next_pred, _), _ = transition(z_batch, action_batch)
            loss = nn.functional.mse_loss(z_next_pred, z_next_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= (len(train_transitions) / batch_size)
        train_losses.append(train_loss)
        
        # Validate
        transition.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_transitions), batch_size):
                batch = val_transitions[i:i+batch_size]
                if not batch:
                    continue
                z_batch = torch.stack([t[0] for t in batch]).to(device)
                action_batch = torch.tensor([t[1] for t in batch], dtype=torch.long).to(device)
                z_next_batch = torch.stack([t[2] for t in batch]).to(device)
                
                (z_next_pred, _), _ = transition(z_batch, action_batch)
                loss = nn.functional.mse_loss(z_next_pred, z_next_batch)
                val_loss += loss.item()
        
        if len(val_transitions) > 0:
            val_loss /= (len(val_transitions) / batch_size)
            val_losses.append(val_loss)
        else:
            val_loss = train_loss
            val_losses.append(val_loss)
        
        print(f"Epoch {epoch}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(transition.state_dict(), output_dir / f'{level_name}_transition_best.pt')
    
    # Save final model
    torch.save(transition.state_dict(), output_dir / f'{level_name}_transition_final.pt')
    
    # Plot training curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{level_name} Transition Training (τ={temporal_resolution})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'{level_name}_transition_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"{level_name} Transition training complete! Best val loss: {best_val_loss:.6f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical RGM')
    parser.add_argument('--level0_vae_path', type=str, required=True,
                       help='Path to pre-trained Level 0 VAE')
    parser.add_argument('--env_name', type=str, default='Breakout',
                       help='Atari environment name')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of episodes for data collection')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--epochs_vae', type=int, default=50,
                       help='Epochs for VAE training')
    parser.add_argument('--epochs_transition', type=int, default=50,
                       help='Epochs for Transition training')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/mps/auto)')
    parser.add_argument('--output_dir', type=str, default='outputs/hierarchical_training',
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
    
    # Configuration
    level0_latent_dim = 32
    level1_latent_dim = 16
    level2_latent_dim = 8
    level1_temporal_resolution = 4
    level2_temporal_resolution = 16
    
    print("\n" + "="*60)
    print("HIERARCHICAL MODEL TRAINING")
    print("="*60)
    print(f"Level 0: Pixel → {level0_latent_dim}D (τ=1, pre-trained)")
    print(f"Level 1: {level0_latent_dim}D → {level1_latent_dim}D (τ={level1_temporal_resolution})")
    print(f"Level 2: {level1_latent_dim}D → {level2_latent_dim}D (τ={level2_temporal_resolution})")
    print("="*60)
    
    # Load Level 0 VAE
    print("\n[1/7] Loading Level 0 VAE...")
    level0_vae = VAE(input_shape=(3, 64, 64), latent_dim=level0_latent_dim).to(device)
    level0_vae.load_state_dict(torch.load(args.level0_vae_path, weights_only=True, map_location=device))
    level0_vae.eval()
    print("Level 0 VAE loaded!")
    
    # Collect data
    print("\n[2/7] Collecting hierarchical data...")
    start_time = time.time()
    data = collect_hierarchical_data(
        args.env_name,
        level0_vae,
        args.num_episodes,
        args.max_steps,
        device
    )
    print(f"Data collection complete! Time: {(time.time() - start_time)/60:.1f} min")
    
    # Train Level 1 VAE
    print("\n[3/7] Training Level 1 VAE...")
    env = AtariPixelEnv(env_id=args.env_name, device=device)
    action_dim = env.action_space.n
    
    level1_vae = SimpleVAE(
        input_dim=level0_latent_dim,
        latent_dim=level1_latent_dim,
        hidden_dim=128
    ).to(device)
    
    train_vae_on_latents(
        level1_vae,
        data['level0_latents'],
        args.epochs_vae,
        args.batch_size,
        args.lr,
        device,
        output_dir,
        'level1'
    )
    
    # Encode Level 0 latents to Level 1
    print("\n[4/7] Encoding Level 1 latents...")
    level1_vae.eval()
    with torch.no_grad():
        level1_latents = []
        for i in range(0, len(data['level0_latents']), args.batch_size):
            batch = data['level0_latents'][i:i+args.batch_size].to(device)
            mu, logvar = level1_vae.encode(batch)
            z1 = level1_vae.reparameterize(mu, logvar)
            level1_latents.append(z1.cpu())
        level1_latents = torch.cat(level1_latents, dim=0)
    print(f"Encoded {len(level1_latents)} Level 1 latents")
    
    # Train Level 1 Transition
    print("\n[5/7] Training Level 1 Transition...")
    level1_transition = TransitionModel(
        latent_dim=level1_latent_dim,
        action_dim=action_dim,
        hidden_dim=64
    ).to(device)
    
    train_transition_on_latents(
        level1_transition,
        level1_latents,
        data['actions'],
        args.epochs_transition,
        args.batch_size,
        args.lr,
        device,
        output_dir,
        'level1',
        temporal_resolution=level1_temporal_resolution
    )
    
    # Train Level 2 VAE
    print("\n[6/7] Training Level 2 VAE...")
    level2_vae = SimpleVAE(
        input_dim=level1_latent_dim,
        latent_dim=level2_latent_dim,
        hidden_dim=64
    ).to(device)
    
    train_vae_on_latents(
        level2_vae,
        level1_latents,
        args.epochs_vae,
        args.batch_size,
        args.lr,
        device,
        output_dir,
        'level2'
    )
    
    # Train Level 2 Transition
    print("\n[7/7] Training Level 2 Transition...")
    level2_vae.eval()
    with torch.no_grad():
        level2_latents = []
        for i in range(0, len(level1_latents), args.batch_size):
            batch = level1_latents[i:i+args.batch_size].to(device)
            mu, logvar = level2_vae.encode(batch)
            z2 = level2_vae.reparameterize(mu, logvar)
            level2_latents.append(z2.cpu())
        level2_latents = torch.cat(level2_latents, dim=0)
    print(f"Encoded {len(level2_latents)} Level 2 latents")
    
    level2_transition = TransitionModel(
        latent_dim=level2_latent_dim,
        action_dim=action_dim,
        hidden_dim=32
    ).to(device)
    
    train_transition_on_latents(
        level2_transition,
        level2_latents,
        data['actions'],
        args.epochs_transition,
        args.batch_size,
        args.lr,
        device,
        output_dir,
        'level2',
        temporal_resolution=level2_temporal_resolution
    )
    
    # Save configuration
    print("\nSaving hierarchical model configuration...")
    config = {
        'level0_vae_path': args.level0_vae_path,
        'level0_latent_dim': level0_latent_dim,
        'level1_latent_dim': level1_latent_dim,
        'level2_latent_dim': level2_latent_dim,
        'level1_temporal_resolution': level1_temporal_resolution,
        'level2_temporal_resolution': level2_temporal_resolution,
        'action_dim': action_dim,
        'device': device
    }
    torch.save(config, output_dir / 'hierarchical_config.pt')
    
    print("\n" + "="*60)
    print("HIERARCHICAL TRAINING COMPLETE!")
    print("="*60)
    print(f"All models saved to: {output_dir}")
    print("\nModel files:")
    print("  - level1_vae_best.pt")
    print("  - level1_transition_best.pt")
    print("  - level2_vae_best.pt")
    print("  - level2_transition_best.pt")
    print("  - hierarchical_config.pt")


if __name__ == '__main__':
    main()
