import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from src.utils.buffer import ReplayBuffer
from src.base_trainer import BaseTrainer

class HierarchicalTrainer(BaseTrainer):
    def __init__(self, env, agent, buffer_size=10000, batch_size=64, lr=1e-3, device='cpu'):
        super().__init__(env, agent, buffer_size, batch_size, lr, device)
        
        # Optimizers for Level 1
        self.vae1_optimizer = optim.Adam(self.agent.vae1.parameters(), lr=lr)
        self.trans1_optimizer = optim.Adam(self.agent.trans1.parameters(), lr=lr)
        
        # Optimizers for Level 2
        self.vae2_optimizer = optim.Adam(self.agent.vae2.parameters(), lr=lr)
        self.trans2_optimizer = optim.Adam(self.agent.trans2.parameters(), lr=lr)

    # collect_data is inherited from BaseTrainer

    def train_step(self):
        if self.buffer.size < self.batch_size:
            return {}
            
        # Sample batch
        batch = self.buffer.sample(self.batch_size)
        obs = batch['obs']
        actions = batch['act']
        next_obs = batch['next_obs']
        
        # --- Level 1 Training ---
        
        # 1. VAE1 Update
        recon_batch, mu, logvar = self.agent.vae1(obs)
        # Use BaseVAE loss function
        loss_dict = self.agent.vae1.loss_function(recon_batch, obs, mu, logvar)
        vae1_loss = loss_dict['loss'] / self.batch_size
        
        self.vae1_optimizer.zero_grad()
        vae1_loss.backward()
        self.vae1_optimizer.step()
        
        # 2. Transition1 Update
        # Get z_t and z_{t+1} using the (now updated or frozen) VAE1
        with torch.no_grad():
            mu_t, logvar_t = self.agent.vae1.encode(obs)
            z_t = self.agent.vae1.reparameterize(mu_t, logvar_t)
            
            mu_next, logvar_next = self.agent.vae1.encode(next_obs)
            z_next_target = self.agent.vae1.reparameterize(mu_next, logvar_next)
            
        # Predict next state
        # Note: Transition model usually takes hidden state. 
        # For batch training from replay buffer, we often ignore hidden state (stateless training)
        # or use a sequence buffer. Here we do stateless for simplicity as in Phase 3.
        # Squeeze actions to (Batch,) for embedding
        actions_squeezed = actions.squeeze(1) if actions.dim() > 1 else actions
        (pred_mu, pred_logvar), _ = self.agent.trans1(z_t, actions_squeezed, None)
        
        # Transition Loss: Negative Log Likelihood of z_next_target under predicted dist
        # Or simply MSE if we treat it deterministically, but let's use KLD-like or Gaussian NLL
        # NLL = 0.5 * (logvar + (target - mu)^2 / exp(logvar))
        trans1_loss = F.mse_loss(pred_mu, z_next_target) # Simplified for now
        
        self.trans1_optimizer.zero_grad()
        trans1_loss.backward()
        self.trans1_optimizer.step()
        
        # --- Level 2 Training ---
        # Level 2 treats z_t (Level 1 latent) as its observation.
        
        # 3. VAE2 Update
        # Input: z_t (detached from Level 1 graph)
        z_t_input = z_t.detach()
        
        recon_z, mu2, logvar2 = self.agent.vae2(z_t_input)
        
        loss_dict2 = self.agent.vae2.loss_function(recon_z, z_t_input, mu2, logvar2)
        vae2_loss = loss_dict2['loss'] / self.batch_size
        
        self.vae2_optimizer.zero_grad()
        vae2_loss.backward()
        self.vae2_optimizer.step()
        
        # 4. Transition2 Update
        # Predict next z2 from current z2 and action (or abstract action)
        # For now, we use the same primitive action
        with torch.no_grad():
            mu2_t, logvar2_t = self.agent.vae2.encode(z_t_input)
            z2_t = self.agent.vae2.reparameterize(mu2_t, logvar2_t)
            
            z_next_input = z_next_target.detach()
            mu2_next, logvar2_next = self.agent.vae2.encode(z_next_input)
            z2_next_target = self.agent.vae2.reparameterize(mu2_next, logvar2_next)
            
        (pred_mu2, pred_logvar2), _ = self.agent.trans2(z2_t, actions_squeezed, None)
        
        trans2_loss = F.mse_loss(pred_mu2, z2_next_target)
        
        self.trans2_optimizer.zero_grad()
        trans2_loss.backward()
        self.trans2_optimizer.step()
        
        return {
            'vae1_loss': vae1_loss.item(),
            'trans1_loss': trans1_loss.item(),
            'vae2_loss': vae2_loss.item(),
            'trans2_loss': trans2_loss.item()
        }
    
    def train_on_expert_trajectories(self, trajectories, epochs=1):
        """
        Trains Level 2 models on expert trajectories.
        Assumes Level 1 is frozen or pre-trained.
        """
        losses = {'level2_loss': 0.0}
        
        for epoch in range(epochs):
            epoch_loss = 0
            count = 0
            
            for traj in trajectories:
                obs_list = traj['obs']
                act_list = traj['actions']
                
                # Convert to tensors
                obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
                # Ensure (T, C, H, W)
                if obs_tensor.dim() == 3: # (T, H, W) -> (T, 1, H, W)
                     obs_tensor = obs_tensor.unsqueeze(1)
                elif obs_tensor.dim() == 5: # (T, 1, C, H, W) -> (T, C, H, W)
                     obs_tensor = obs_tensor.squeeze(1)
                     
                act_tensor = torch.tensor(act_list, dtype=torch.long, device=self.device)
                
                # 1. Encode with Level 1 VAE (Frozen)
                with torch.no_grad():
                    mu1, logvar1 = self.agent.vae1.encode(obs_tensor)
                    z1_seq = self.agent.vae1.reparameterize(mu1, logvar1)
                    
                # 2. Train Level 2 VAE to reconstruct z1 sequence
                # Input: z1_seq (T, Latent1)
                recon_z1, mu2, logvar2 = self.agent.vae2(z1_seq)
                
                recon_loss = F.mse_loss(recon_z1, z1_seq, reduction='mean')
                kld_loss = -0.5 * torch.mean(1 + logvar2 - mu2.pow(2) - logvar2.exp())
                vae2_loss = recon_loss + kld_loss
                
                self.vae2_optimizer.zero_grad()
                vae2_loss.backward()
                self.vae2_optimizer.step()
                
                # 3. Train Level 2 Transition
                # Predict z2_{t+1} from z2_t and action
                # Get z2 sequence
                with torch.no_grad():
                    mu2, logvar2 = self.agent.vae2.encode(z1_seq)
                    z2_seq = self.agent.vae2.reparameterize(mu2, logvar2)
                
                z2_current = z2_seq[:-1]
                z2_next_target = z2_seq[1:]
                actions_current = act_tensor[:-1]
                
                (pred_mu2, pred_logvar2), _ = self.agent.trans2(z2_current, actions_current, None)
                
                trans2_loss = F.mse_loss(pred_mu2, z2_next_target)
                
                self.trans2_optimizer.zero_grad()
                trans2_loss.backward()
                self.trans2_optimizer.step()
                
                epoch_loss += (vae2_loss.item() + trans2_loss.item())
                count += 1
                
            losses['level2_loss'] = epoch_loss / count if count > 0 else 0
            
        return losses
