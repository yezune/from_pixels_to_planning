import torch
import numpy as np

class ExpertCollector:
    def __init__(self, env, expert_agent, device='cpu'):
        self.env = env
        self.expert_agent = expert_agent
        self.device = device

    def collect_episodes(self, n_episodes, max_steps=100):
        trajectories = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_obs = []
            episode_actions = []
            
            for _ in range(max_steps):
                # Store observation
                if isinstance(obs, torch.Tensor):
                    episode_obs.append(obs.cpu().numpy())
                else:
                    episode_obs.append(obs)
                
                # Get expert action
                action = self.expert_agent.select_action(obs)
                episode_actions.append(action)
                
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            trajectories.append({
                'obs': episode_obs,
                'actions': episode_actions
            })
            
        return trajectories
