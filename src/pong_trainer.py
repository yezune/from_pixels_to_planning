import torch
import torch.optim as optim
import numpy as np
from collections import deque
import os
import sys
from src.models.logical_pong_agent import LogicalPongAgent
from src.l_fep.utils import calculate_distinction_bonus
from src.envs.atari_env import AtariPixelEnv

class FrameStackWrapper(AtariPixelEnv):
    def __init__(self, env_id, image_size=64, k=4, device='cpu'):
        super().__init__(env_id, image_size, device)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.device = device
        
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options) # obs is (3, 64, 64) tensor on device
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
        
    def _get_obs(self):
        assert len(self.frames) == self.k
        return torch.cat(list(self.frames), dim=0) # (C*k, H, W)

class PongTrainer:
    def __init__(self, env_id='PongNoFrameskip-v4', lr=1e-4, gamma=0.99, intrinsic_weight=0.01, device='cpu'):
        # Allow device override, default to cpu for stability in current phase
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        self.env = FrameStackWrapper(env_id, k=4, device=self.device)
        self.action_dim = self.env.action_space.n
        self.input_channels = 3 * 4 # 3 channels * 4 frames
        
        self.model = LogicalPongAgent(self.input_channels, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.gamma = gamma
        self.intrinsic_weight = intrinsic_weight
        
        # Storage for one episode
        self.saved_log_probs = []
        self.rewards = []
        self.distinctions = []

    def select_action(self, state):
        # state is (12, 64, 64) tensor on device
        state = state.unsqueeze(0) # (1, 12, 64, 64)
        
        psi = self.model(state) # (1, action_dim) amplitude
        probs = psi ** 2 # Born rule
        
        # Calculate distinction (intrinsic reward)
        distinction = calculate_distinction_bonus(probs)
        self.distinctions.append(distinction)
        
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        
        return action.item()

    def update(self):
        R = 0
        policy_loss = []
        returns = []
        
        # Combine extrinsic and intrinsic rewards
        total_rewards = []
        for r, d in zip(self.rewards, self.distinctions):
            # Intrinsic reward is the distinction bonus
            r_total = r + self.intrinsic_weight * d.item()
            total_rewards.append(r_total)
            
        for r in reversed(total_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(self.device)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # Clear memory
        del self.saved_log_probs[:]
        del self.rewards[:]
        del self.distinctions[:]
        
        return loss.item()

    def save_checkpoint(self, episode, filename="pong_checkpoint.pth"):
        path = os.path.join("checkpoints", filename)
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def train(self, num_episodes=100, checkpoint_interval=20):
        print(f"Starting L-AGI Pong Training for {num_episodes} episodes...")
        
        recent_rewards = deque(maxlen=checkpoint_interval)
        best_avg_reward = -22.0 # Lower than min possible reward (-21)
        
        for i_episode in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            ep_reward = 0
            
            for t in range(10000): # Max steps per episode
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                
                self.rewards.append(reward)
                ep_reward += reward
                
                if terminated or truncated:
                    break
            
            loss = self.update()
            recent_rewards.append(ep_reward)
            
            print(f"Episode {i_episode}\tReward: {ep_reward:.2f}\tLoss: {loss:.4f}")
            
            # Checkpoint and Improvement Check
            if i_episode % checkpoint_interval == 0:
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                print(f"--- Checkpoint {i_episode} --- Avg Reward: {avg_reward:.2f}")
                self.save_checkpoint(i_episode, f"pong_ep{i_episode}.pth")
                
                # Check for improvement
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    print(f"Improvement detected! New best avg reward: {best_avg_reward:.2f}")
                else:
                    if avg_reward <= -21.0 and i_episode >= 20:
                         print("No improvement detected (stuck at -21.0). Stopping experiment.")
                         return False
                    elif avg_reward < best_avg_reward - 1.0: # Significant degradation
                         print(f"Performance degraded ({best_avg_reward:.2f} -> {avg_reward:.2f}). Stopping experiment.")
                         return False
                         
        print("Training Completed.")
        return True
