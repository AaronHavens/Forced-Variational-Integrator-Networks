import gym
import numpy as np
import torch
import gym_softreacher
import os
from model_env import ModelEnv




def soft_reward(s, a):
    #print(s[:3], a)
    return -(np.inner(s[:3],s[:3]) + 0.001*np.inner(a,a))

model = torch.load('models/theta_test_u.pt')
env_ = gym.make('SoftReacher-v0') 
env = ModelEnv(model, soft_reward, env_, H=20)


#print(env.reset())
env.reset()
env_.reset()
a = env.action_space.sample()
print(env_.step(env_.action_space.sample()))
print(env.step(a))
