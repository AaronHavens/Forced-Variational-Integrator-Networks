import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from cem import CEM
import gym_softreacher
from matplotlib import rc
from multi_env import MultiEnv
from stable_baselines import SAC
rc('text', usetex=True)

#pi = torch.load('models/softarm_policy.pt')
#model = torch.load('models/theta_test_u.pt')
model = torch.load('models/soft_big_steps_pf.pt')
#env = gym.make('Pendulum-v0')
env = MultiEnv('SoftReacher-v0')
#sac_model = SAC.load('sac_cheetah_wexpert_iclr_1_3')
x_dim = len(env.observation_space.low)
u_dim = len(env.action_space.low)
pi = CEM(model, 1000, 10, env.action_space.low, env.action_space.high)
H = 100
env.seed(30)
x = env.reset()
x_hat = np.copy(x)

xs = np.zeros((H,x_dim))
xs_hat = np.zeros((H,x_dim))
r_tot = 0
for i in range(H):
    #u = env.action_space.sample()
    #u, _ = sac_model.predict(x)
    #u += np.random.normal(0, 0.1)
    u, _  = pi.predict(x)
    #u += np.random.uniform(0., 0.1)
    print(u, np.linalg.norm(x[:3]))
    #u = np.zeros(env.action_space.sample().shape)
    x_hat_next = model.predict(x_hat, u)
    x_next, r, done, _ = env.step(u)
    r_tot += r
    #env.render()
    xs[i,:] = x_next
    xs_hat[i,:] = x_hat_next
    x = x_next
    x_hat = x_hat_next
    #if done:
    #    break
print('total cost', r_tot)
normed_true = np.linalg.norm(xs[:i,:3], axis=1)
normed_predict = np.linalg.norm(xs_hat[:i,:3], axis=1)
plt.plot(normed_true)
plt.show()

plt.style.use('ggplot')
ax1 = plt.subplot(311)
plt.plot(xs[:i,0], label=r'true')
plt.plot(xs_hat[:i,0], label=r'predicted')
plt.ylabel(r'$\bar x_1$ distance [cm]')

ax2 = plt.subplot(312)
plt.plot(xs[:i,1], label=r'true')
plt.plot(xs_hat[:i,1], label=r'predicted')
plt.ylabel(r'$\bar x_2$ distance [cm]')

ax1 = plt.subplot(313)
plt.plot(xs[:i,2], label=r'true')
plt.plot(xs_hat[:i,2], label=r'predicted')
plt.xlabel(r'action step')
plt.ylabel(r'$\bar x_3$ distance [cm]')
plt.legend()
plt.show()
