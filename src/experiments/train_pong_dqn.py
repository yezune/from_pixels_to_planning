#!/usr/bin/env python3
"""
Train DQN on Pong with Frame Stacking.

This script implements a standard DQN agent with Frame Stacking (k=4) to solve Pong.
It serves as a strong baseline and a "Reactive" component for the hierarchical model.

Key Features:
- Frame Stacking (4 frames) for motion detection
- CNN-based Q-Network
- Experience Replay
- Target Network
- Epsilon-Greedy Exploration

Usage:
    python src/experiments/train_pong_dqn.py --num_episodes 500 --output_dir outputs/pong_dqn
"""

import argparse
import os
import sys
import random
import math
from collections import deque
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.envs.atari_env import AtariPixelEnv

# --- Architecture ---

class CNNDQN(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(CNNDQN, self).__init__()
        # Input: (B, C*k, H, W) -> (B, 12, 64, 64)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate FC input size
        # 64x64 -> conv1(s4) -> 15x15 -> conv2(s2) -> 6x6 -> conv3(s1) -> 4x4
        self.fc_input_dim = 64 * 4 * 4
        
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state: (C, H, W) tensor
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.stack(state), 
                torch.tensor(action), 
                torch.tensor(reward, dtype=torch.float32), 
                torch.stack(next_state), 
                torch.tensor(done, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

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
        # Stack along channel dimension: (3, H, W) * 4 -> (12, H, W)
        return torch.cat(list(self.frames), dim=0)

# --- Training ---

def train(args):
    # Setup
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_available() and args.device == 'cpu':
        device = torch.device('mps')
        print("Using MPS device")
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment
    env = FrameStackWrapper(
        env_id=f'{args.env_name}NoFrameskip-v4',
        image_size=64,
        k=4,
        device=device
    )
    
    n_actions = env.action_space.n
    input_channels = 3 * 4 # RGB * 4 frames
    
    print(f"Env: {args.env_name}, Actions: {n_actions}, Input Channels: {input_channels}")
    
    # Networks
    policy_net = CNNDQN(input_channels, n_actions).to(device)
    target_net = CNNDQN(input_channels, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    memory = ReplayBuffer(args.buffer_size)
    
    # Metrics
    episode_rewards = []
    losses = []
    best_reward = -float('inf')
    
    steps_done = 0
    
    print("Starting training...")
    start_time = time.time()
    
    for i_episode in range(args.num_episodes):
        obs, _ = env.reset()
        state = obs # (12, 64, 64) on device
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon Greedy
            epsilon = args.eps_end + (args.eps_start - args.eps_end) * \
                      math.exp(-1. * steps_done / args.eps_decay)
            steps_done += 1
            
            if random.random() > epsilon:
                with torch.no_grad():
                    # state.unsqueeze(0) -> (1, 12, 64, 64)
                    action = policy_net(state.unsqueeze(0)).max(1)[1].item()
            else:
                action = env.action_space.sample()
                
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = next_obs
            total_reward += reward
            
            # Store in CPU memory to save GPU/MPS memory
            memory.push(state.cpu(), action, reward, next_state.cpu(), done)
            state = next_state
            
            # Optimize
            if len(memory) > args.batch_size:
                states, actions, rewards, next_states, dones = memory.sample(args.batch_size)
                
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                
                # Q(s, a)
                q_values = policy_net(states).gather(1, actions.unsqueeze(1))
                
                # V(s') = max_a Q(s', a)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                
                # Target = r + gamma * V(s')
                expected_q_values = rewards + (args.gamma * next_q_values * (1 - dones))
                
                loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
                
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                
                losses.append(loss.item())
                
            if steps_done % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards[-10:])
        
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy_net.state_dict(), output_dir / 'best_model.pt')
            
        if i_episode % 10 == 0:
            print(f"Episode {i_episode}/{args.num_episodes} | "
                  f"Reward: {total_reward:.1f} | Avg(10): {avg_reward:.1f} | "
                  f"Eps: {epsilon:.2f} | Steps: {steps_done}")
            
            # Save checkpoint
            torch.save({
                'episode': i_episode,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': total_reward,
            }, output_dir / 'checkpoint.pt')
            
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best Reward: {best_reward}")
    
    # Save final model
    torch.save(policy_net.state_dict(), output_dir / 'final_model.pt')
    
    # Save logs
    import json
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump({
            'rewards': episode_rewards,
            'losses': losses
        }, f)

def main():
    parser = argparse.ArgumentParser(description='Train DQN on Pong')
    parser.add_argument('--env_name', type=str, default='Pong', help='Atari environment name')
    parser.add_argument('--num_episodes', type=int, default=500, help='Number of episodes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps_start', type=float, default=1.0, help='Starting epsilon')
    parser.add_argument('--eps_end', type=float, default=0.05, help='Ending epsilon')
    parser.add_argument('--eps_decay', type=int, default=10000, help='Epsilon decay steps')
    parser.add_argument('--target_update', type=int, default=1000, help='Target network update frequency')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=50000, help='Replay buffer size')
    parser.add_argument('--output_dir', type=str, default='outputs/pong_dqn', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
