import gym
import gym_softreacher
import numpy as np

env = gym.make("SoftReacher-v0")
x = env.reset()
print(x)

a = np.array([0,0,0,1e-2/3e-2])
print(3e-2*a[3])
for i in range(10):
    x, r, done, _ = env.step(a)
    print(x)
