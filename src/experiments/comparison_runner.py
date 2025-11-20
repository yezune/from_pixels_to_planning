import torch
import numpy as np
from tqdm import tqdm

class ComparisonRunner:
    def __init__(self, env, agents):
        """
        Args:
            env: Gymnasium environment (wrapped for pixel input)
            agents: Dictionary of agents {'name': agent_instance}
        """
        self.env = env
        self.agents = agents
        
    def run_episode(self, agent, max_steps=1000):
        """
        Runs a single episode for the given agent.
        Returns:
            total_reward: float
            steps: int
        """
        obs, _ = self.env.reset()
        # Ensure obs is tensor
        if not isinstance(obs, torch.Tensor):
            # Assuming env wrapper handles this, but for safety in tests
            pass 
            
        agent.reset()
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            obs = next_obs
            
        return total_reward, steps

    def evaluate(self, num_episodes=10):
        """
        Evaluates all agents for a number of episodes.
        Returns:
            results: Dictionary {'agent_name': {'rewards': [], 'lengths': [], 'mean_reward': float}}
        """
        results = {}
        
        for name, agent in self.agents.items():
            print(f"Evaluating {name}...")
            agent_rewards = []
            agent_lengths = []
            
            for _ in tqdm(range(num_episodes), desc=f"{name} Episodes"):
                reward, steps = self.run_episode(agent)
                agent_rewards.append(reward)
                agent_lengths.append(steps)
                
            results[name] = {
                'rewards': agent_rewards,
                'lengths': agent_lengths,
                'mean_reward': np.mean(agent_rewards),
                'std_reward': np.std(agent_rewards),
                'mean_length': np.mean(agent_lengths)
            }
            
        return results
