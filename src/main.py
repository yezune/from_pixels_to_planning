import os
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from src.envs.env_wrapper import ActiveInferenceEnv
from src.envs.synthetic_env import BouncingBallEnv
from src.models.vae import VAE
from src.models.transition import TransitionModel
from src.models.agent import ActiveInferenceAgent
from src.trainer import ActiveInferenceTrainer
from src.utils.visualization import plot_reconstruction, plot_free_energy_history

def main():
    # Register Custom Env
    gym.register(id='BouncingBall-v0', entry_point='src.envs.synthetic_env:BouncingBallEnv')

    # Hyperparameters
    ENV_ID = 'BouncingBall-v0'
    TARGET_SIZE = (64, 64)
    LATENT_DIM = 32
    HIDDEN_DIM = 64
    BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    LR = 1e-3
    ITERATIONS = 5 # Reduced for quick demo
    STEPS_PER_ITER = 50
    TRAIN_EPOCHS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Setup
    # Note: BouncingBallEnv natively returns 64x64 images, so target_size matches
    env = ActiveInferenceEnv(env_id=ENV_ID, target_size=TARGET_SIZE, grayscale=True, device=DEVICE)
    action_dim = env.action_space.n
    
    vae = VAE(input_shape=(1, *TARGET_SIZE), latent_dim=LATENT_DIM).to(DEVICE)
    transition = TransitionModel(latent_dim=LATENT_DIM, action_dim=action_dim, hidden_dim=HIDDEN_DIM).to(DEVICE)
    agent = ActiveInferenceAgent(vae, transition, action_dim=action_dim, device=DEVICE)
    
    trainer = ActiveInferenceTrainer(env, agent, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, lr=LR, device=DEVICE)
    
    # Logging
    vae_losses = []
    trans_losses = []
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting Training Loop...")
    
    for i in range(ITERATIONS):
        print(f"\nIteration {i+1}/{ITERATIONS}")
        
        # 1. Collect Data
        trainer.collect_data(num_steps=STEPS_PER_ITER)
        print(f"  - Collected {STEPS_PER_ITER} steps. Buffer size: {len(trainer.buffer)}")
        
        # 2. Train Models
        v_loss = trainer.train_vae(epochs=TRAIN_EPOCHS)
        t_loss = trainer.train_transition(epochs=TRAIN_EPOCHS)
        
        vae_losses.append(v_loss)
        trans_losses.append(t_loss)
        
        print(f"  - VAE Loss: {v_loss:.4f}")
        print(f"  - Transition Loss: {t_loss:.4f}")
        
        # 3. Visualization (every iteration for demo)
        if True:
            # Plot Loss History
            plot_free_energy_history(vae_losses, save_path=os.path.join(output_dir, 'vae_loss.png'))
            
            # Plot Reconstruction
            batch = trainer.buffer.sample(4)
            obs = batch['obs']
            with torch.no_grad():
                recon, _, _ = vae(obs)
            plot_reconstruction(obs, recon, save_path=os.path.join(output_dir, f'recon_iter_{i+1}.png'))
            print("  - Saved visualizations.")

    print("Training Complete.")
    env.close()

if __name__ == '__main__':
    main()
