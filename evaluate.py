import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from cem import CEM
import gym_softreacher
from matplotlib import rc
rc('text', usetex=True)

model = torch.load('models/sin.pt')
env = gym.make('SoftReacher-v0')
x_dim = len(env.observation_space.low)
u_dim = len(env.action_space.low)
pi = CEM(model, 1000, 10, env.action_space.low, env.action_space.high)
H = 100
env.seed(0)
x = env.reset()
x_hat = np.copy(x)

xs = np.zeros((H,x_dim))
xs_hat = np.zeros((H,x_dim))

for i in range(H):
    
    u, _  = pi.predict(x)#np.zeros(env.action_space.sample().shape)
    x_hat_next = modle.predict(x_hat, u)
    x_next, r, done, _ = env.step(u)
    #env.render()
    xs[i,:] = x_next
    xs_hat[i,:] = x_hat_next
    x = x_next
    print(x)
    x_hat = x_hat_next
    if done:
        print("pressure constraint")

normed_true = np.linalg.norm(xs[:i,:3], axis=1)
normed_predict = np.linalg.norm(xs_hat[:i,:3], axis=1)

plt.style.use('ggplot')
plt.plot(normed_true, label=r'true')
plt.plot(normed_predict, label=r'predicted')
plt.xlabel(r'action step')
plt.ylabel(r'$||x||_2$ distance')
plt.legend()
plt.show()
