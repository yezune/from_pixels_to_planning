import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

class PixelMiniGridEnv(gym.Wrapper):
    def __init__(self, env_id, render_mode='rgb_array', image_size=64):
        # Create the base environment
        env = gym.make(env_id, render_mode=render_mode)
        # We wrap it with RGBImgObsWrapper if we wanted egocentric RGB, 
        # but here we want the global view from render(), or we can use the wrapper.
        # Let's use the base env and override step/reset to return render() output.
        
        super().__init__(env)
        self.image_size = image_size
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, image_size, image_size), dtype=np.float32
        )
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(), # Converts to [0, 1] and (C, H, W)
        ])

    def _get_pixel_obs(self):
        # Get the full rendered image
        frame = self.env.render()
        if frame is None:
            # Fallback if render fails or returns None (e.g. no display)
            # Create a black image
            frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            
        # Frame is (H, W, 3) numpy array
        obs = self.transform(frame)
        return obs

    def reset(self, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        obs = self._get_pixel_obs()
        return obs, info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        obs = self._get_pixel_obs()
        return obs, reward, terminated, truncated, info
