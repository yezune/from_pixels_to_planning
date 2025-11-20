import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from src.utils.buffer import ReplayBuffer

class ActiveInferenceTrainer:
    def __init__(self, env, agent, buffer_size=10000, batch_size=64, lr=1e-3, device='cpu'):
        self.env = env
        self.agent = agent
        self.device = device
        self.batch_size = batch_size
        
        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            obs_shape=env.observation_space.shape if hasattr(env.observation_space, 'shape') else (1, 64, 64), # Fallback or use env wrapper target size
            action_dim=agent.action_dim,
            device=device
        )
        # Fix buffer obs shape if env wrapper is used
        if hasattr(env, 'target_size'):
             # (C, H, W)
             c = 1 if env.grayscale else 3
             self.buffer.obs_shape = (c, *env.target_size)
             self.buffer.obs_buf = np.zeros((buffer_size, c, *env.target_size), dtype=np.float32)
             self.buffer.next_obs_buf = np.zeros((buffer_size, c, *env.target_size), dtype=np.float32)

        self.vae_optimizer = optim.Adam(self.agent.vae.parameters(), lr=lr)
        self.transition_optimizer = optim.Adam(self.agent.transition_model.parameters(), lr=lr)

    def collect_data(self, num_steps):
        """
        Interacts with the environment and stores data in the buffer.
        """
        obs, _ = self.env.reset()
        self.agent.reset()
        
        for _ in range(num_steps):
            # Select action (using current policy/planning)
            # Initially this might be random if models are untrained
            action = self.agent.select_action(obs)
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            
            # Store in buffer
            # obs and next_obs are tensors on device, buffer handles conversion
            self.buffer.add(obs, action, reward, next_obs, float(terminated))
            
            obs = next_obs
            
            if terminated or truncated:
                obs, _ = self.env.reset()
                self.agent.reset()

    def train_vae(self, epochs=1):
        """
        Trains the VAE component.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0
            
        total_loss = 0
        num_batches = 0
        
        self.agent.vae.train()
        
        for _ in range(epochs):
            # Simple epoch: just sample a few batches? 
            # Or iterate through buffer? 
            # For RL, we usually sample N batches per 'epoch' call.
            # Let's sample 10 batches per call for now.
            for _ in range(10):
                batch = self.buffer.sample(self.batch_size)
                obs = batch['obs']
                
                # Permute if needed (B, H, W, C) -> (B, C, H, W)
                if obs.dim() == 4 and obs.shape[-1] in [1, 3] and obs.shape[1] not in [1, 3]:
                    obs = obs.permute(0, 3, 1, 2)
                
                self.vae_optimizer.zero_grad()
                
                recon, mu, logvar = self.agent.vae(obs)
                loss_dict = self.agent.vae.loss_function(recon, obs, mu, logvar)
                
                loss = loss_dict['loss']
                loss.backward()
                self.vae_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / max(1, num_batches)

    def train_transition(self, epochs=1):
        """
        Trains the Transition Model.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0
            
        total_loss = 0
        num_batches = 0
        
        self.agent.transition_model.train()
        self.agent.vae.eval() # Freeze VAE for this part
        
        for _ in range(epochs):
            for _ in range(10):
                batch = self.buffer.sample(self.batch_size)
                obs = batch['obs']
                next_obs = batch['next_obs']
                act = batch['act'].squeeze() # (Batch,)
                
                # Permute if needed (B, H, W, C) -> (B, C, H, W)
                if obs.dim() == 4 and obs.shape[-1] in [1, 3] and obs.shape[1] not in [1, 3]:
                    obs = obs.permute(0, 3, 1, 2)
                if next_obs.dim() == 4 and next_obs.shape[-1] in [1, 3] and next_obs.shape[1] not in [1, 3]:
                    next_obs = next_obs.permute(0, 3, 1, 2)
                
                # Encode observations to get latent states (Targets)
                with torch.no_grad():
                    mu_t, logvar_t = self.agent.vae.encode(obs)
                    z_t = self.agent.vae.reparameterize(mu_t, logvar_t)
                    
                    mu_tp1, logvar_tp1 = self.agent.vae.encode(next_obs)
                    z_tp1 = self.agent.vae.reparameterize(mu_tp1, logvar_tp1)
                
                self.transition_optimizer.zero_grad()
                
                # Predict next state
                # Hidden state is not used in training here (stateless training for simplicity)
                # Or we should use sequences. For now, let's assume 1-step transition training.
                # If using RNN, we usually need sequences. 
                # But our TransitionModel takes (z_t, action, hidden).
                # If we pass hidden=None, it initializes zero hidden state.
                # This is suboptimal for RNN but okay for testing the loop.
                # Ideally we sample sequences from buffer.
                
                (pred_mu, pred_logvar), _ = self.agent.transition_model(z_t, act, hidden=None)
                
                # Loss: Negative Log Likelihood of z_tp1 under predicted distribution
                # NLL = 0.5 * (log(var) + (target - mu)^2 / var)
                # pred_logvar is log(sigma^2)
                var = torch.exp(pred_logvar)
                nll = 0.5 * (pred_logvar + (z_tp1 - pred_mu).pow(2) / var)
                loss = torch.mean(nll)
                
                loss.backward()
                self.transition_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / max(1, num_batches)
