import gym
import gym_custom
import time
import numpy as np

env = gym.make('QPendulum-v0')
x = env.reset()
for i in range(100):
    x, r, done, _ = env.step(np.zeros(env.action_space.sample().shape))
    print(x)
    env.render()
