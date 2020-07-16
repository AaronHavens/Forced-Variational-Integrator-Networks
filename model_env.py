import gym
from gym import spaces
from gym.utils import seeding
import torch
import numpy as np

class ModelEnv():
    #metadata = {
    #    'render.modes' : ['human', 'rgb_array'],
    #    'video.frames_per_second' : 30
    #}

    def __init__(self, env, model):
        self.model = model
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.T = 100
        self.seed()

    def seed(self, seed=None):
        self.env.seed(seed)
        return [seed]

    def step(self,u_):
        y = self.state
        u = torch.from_numpy(np.expand_dims(u_, axis=0)).float()
        reward = self.model.reward(y,u)[0]
        y_next, x_next = self.model.predict_from_q(y, u_)
        self.state = y_next
        self.t += 1
        if self.t > self.T:
            done = True
        else:
            done = False
        return x_next, reward, done, {}

    def reset(self):
        x = self.env.reset()
        x = torch.from_numpy(np.expand_dims(x,axis=0)).float()
        self.state = self.model.encoder(x)
        self.t = 0
        return self.model.decoder(self.state).detach().numpy()[0]

    def render(self, mode='human'):
        return 0
   
