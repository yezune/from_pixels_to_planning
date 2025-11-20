import torch
import gymnasium as gym
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.envs.atari_env import AtariPixelEnv
from src.models.hierarchical_agent import HierarchicalAgent
from src.models.vae import VAE
from src.models.mlp_vae import MlpVAE
from src.models.transition import TransitionModel
from src.hierarchical_trainer import HierarchicalTrainer

def run_atari_experiment(env_id="BreakoutNoFrameskip-v4", num_epochs=100, steps_per_epoch=1000, batch_size=32):
    print(f"Starting Atari Experiment on {env_id}")
    
    # Environment
    env = AtariPixelEnv(env_id, image_size=64)
    
    # Model parameters
    input_dim = (3, 64, 64)
    action_dim = env.action_space.n
    latent_dim = 64 # Using same latent dim for both levels for now
    hidden_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Level 1 Models
    vae1 = VAE(input_shape=input_dim, latent_dim=latent_dim).to(device)
    trans1 = TransitionModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    
    # Level 2 Models
    vae2 = MlpVAE(input_dim=latent_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    trans2 = TransitionModel(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    
    agent = HierarchicalAgent(
        level1_models=(vae1, trans1),
        level2_models=(vae2, trans2),
        action_dim=action_dim,
        device=device
    )
    
    trainer = HierarchicalTrainer(
        env=env,
        agent=agent,
        buffer_size=10000,
        batch_size=batch_size,
        lr=1e-4,
        device=device
    )
    
    # Training Loop
    losses_history = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Collect Data
        trainer.collect_data(num_steps=steps_per_epoch)
        
        # Train Step (multiple updates per epoch?)
        # Usually we do multiple updates, but let's do proportional to steps
        num_updates = steps_per_epoch // batch_size
        epoch_losses = {}
        
        pbar = tqdm(range(num_updates), desc="Training")
        for _ in pbar:
            losses = trainer.train_step()
            if not losses:
                continue
                
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v)
            
            pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
            
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        losses_history.append(avg_losses)
        print(f"Avg Losses: {avg_losses}")
        
        # Save checkpoint occasionally
        if (epoch + 1) % 10 == 0:
            save_path = f"checkpoints/atari_{env_id}_epoch_{epoch+1}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'vae1': vae1.state_dict(),
                'trans1': trans1.state_dict(),
                'vae2': vae2.state_dict(),
                'trans2': trans2.state_dict(),
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

    return losses_history

if __name__ == "__main__":
    run_atari_experiment(num_epochs=2, steps_per_epoch=100) # Short run for testing
