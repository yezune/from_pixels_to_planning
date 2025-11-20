import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

class BouncingBallEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=64, ball_radius=5):
        self.size = size
        self.ball_radius = ball_radius
        self.render_mode = render_mode
        
        # Action: 0: No-op, 1: Up, 2: Down, 3: Left, 4: Right
        self.action_space = spaces.Discrete(5)
        
        # Observation: RGB Image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size, 3), dtype=np.uint8
        )
        
        self.pos = np.array([size // 2, size // 2], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.max_speed = 2.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([self.size // 2, self.size // 2], dtype=np.float32)
        self.vel = np.random.uniform(-1, 1, size=2)
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Apply action force
        force = 0.5
        if action == 1: self.vel[1] -= force
        elif action == 2: self.vel[1] += force
        elif action == 3: self.vel[0] -= force
        elif action == 4: self.vel[0] += force
        
        # Update position
        self.pos += self.vel
        
        # Bounce off walls
        if self.pos[0] < self.ball_radius:
            self.pos[0] = self.ball_radius
            self.vel[0] *= -1
        elif self.pos[0] > self.size - self.ball_radius:
            self.pos[0] = self.size - self.ball_radius
            self.vel[0] *= -1
            
        if self.pos[1] < self.ball_radius:
            self.pos[1] = self.ball_radius
            self.vel[1] *= -1
        elif self.pos[1] > self.size - self.ball_radius:
            self.pos[1] = self.size - self.ball_radius
            self.vel[1] *= -1
            
        # Friction
        self.vel *= 0.95
        
        obs = self._get_obs()
        
        # Reward: Keep ball in center? Or just survival?
        # Let's say reward is distance to center (minimize distance -> maximize negative distance)
        center = np.array([self.size / 2, self.size / 2])
        dist = np.linalg.norm(self.pos - center)
        reward = -dist / self.size
        
        terminated = False
        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        # Draw ball
        cv2.circle(
            img, 
            (int(self.pos[0]), int(self.pos[1])), 
            self.ball_radius, 
            (255, 255, 255), 
            -1
        )
        return img

    def render(self):
        return self._get_obs()
