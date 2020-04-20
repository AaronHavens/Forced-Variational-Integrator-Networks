import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np


class ModelEnv(gym.Env):

    def __init__(self, model, reward_fn, env, H):
        self.H = H
        self.model = model
        self.reward = reward_fn
        self.env = env
        self.action_space  = env.action_space
        self.observation_space = env.observation_space
        self.seed()
    
    def seed(self, seed=None):
        self.env.seed(seed)

    def step(self, a):
        done = False
        x_next = self.model.predict(self.state, a)
        r = self.reward(self.state, a)
        if self.t > self.H:
            done = True
        self.state = x_next
        self.t += 1

        return np.copy(self.state), r, done, {}

    def reset(self):
        self.t = 0
        self.state = self.env.reset()
        return np.copy(self.state)
