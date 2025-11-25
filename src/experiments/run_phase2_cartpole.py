import gymnasium as gym
import torch
import numpy as np
from src.models.simple_logical_agent import SimpleLogicalAgent

def run_cartpole_experiment():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = SimpleLogicalAgent(state_dim, action_dim, intrinsic_weight=0.01)
    
    print("Starting Phase 2: Logical Active Inference on CartPole-v1")
    
    running_reward = 10
    for i_episode in range(1, 501): # 500 episodes
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1000):
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            ep_reward += reward
            if terminated or truncated:
                break
        
        agent.update()
        
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        
        if i_episode % 10 == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')
            
        if running_reward > env.spec.reward_threshold:
            print(f"Solved! Running reward is now {running_reward:.2f} and the last episode runs to {ep_reward:.2f} time steps!")
            break

if __name__ == '__main__':
    run_cartpole_experiment()
