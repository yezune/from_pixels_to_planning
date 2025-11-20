import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_dim, device='cpu'):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        
        self.ptr = 0
        self.size = 0
        
        # Buffers
        # obs_shape is (C, H, W)
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.act_buf = np.zeros((capacity, 1), dtype=np.int64) # Discrete action index
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        # obs is tensor or numpy. Convert to numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(next_obs, torch.Tensor):
            next_obs = next_obs.cpu().numpy()
            
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return dict(
            obs=torch.FloatTensor(self.obs_buf[idxs]).to(self.device),
            next_obs=torch.FloatTensor(self.next_obs_buf[idxs]).to(self.device),
            act=torch.LongTensor(self.act_buf[idxs]).to(self.device),
            rew=torch.FloatTensor(self.rew_buf[idxs]).to(self.device),
            done=torch.FloatTensor(self.done_buf[idxs]).to(self.device)
        )

    def __len__(self):
        return self.size
