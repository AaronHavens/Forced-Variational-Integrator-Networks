import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym
from cem import CEM
import gym_softreacher
from matplotlib import rc
from multi_env import MultiEnv
import gym_custom

rc('text', usetex=True)

#pi = torch.load('models/softarm_policy.pt')
#model = torch.load('models/theta_test_u.pt')
model = torch.load('models/real_mpc_test.pt')
#env = gym.make('Pendulum-v0')
env = gym.make('SoftReacher-v0')
#sac_model = SAC.load('sac_cheetah_wexpert_iclr_1_3')
x_dim = len(env.observation_space.low)
u_dim = len(env.action_space.low)
pi = CEM(model, 500, 10, env)
H = 100
#env.seed(5)
x = env.reset()
x_hat = np.copy(x)

xs = np.zeros((H,x_dim))
xs_hat = np.zeros((H,x_dim))
r_tot = 0

def plot_frame(xs, i, target):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xyz = xs[:i+1,:3]
    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2])
    ax.scatter(target[0], target[1], target[2], s=200, c='orange',label='goal')
    fig.legend()
    fig.savefig('extend/'+str(i) + '.png')
    plt.close()

for i in range(H):
    #u = env.action_space.sample()
    #u, _ = sac_model.predict(x)
    #u += np.random.normal(0, 0.1)
    u, _  = pi.predict(x, env.target)
    #u += np.random.uniform(0., 0.1)
    #print('state:', x)
    print('control and error: ', u, np.linalg.norm(x[:3]-env.target))
    #u = np.zeros(env.action_space.sample().shape)
    x_hat_next = model.predict(x_hat, u)
    x_next, r, done, _ = env.step(u)
    r_tot += r
    env.render()
    xs[i,:] = x_next
    xs[i,:3] -= env.target
    #plot_frame(xs, i, env.target)
    #xs[i,:3] -= env.target
    xs_hat[i,:] = x_hat_next
    xs_hat[i,:3] -= env.target
    x = x_next
    x_hat = x_hat_next
    #if done:
    #    break
print('total cost', r_tot)


plt.style.use('ggplot')
ax1 = plt.subplot(311)
plt.plot(xs[:i,0], label=r'true')
#plt.plot(xs_hat[:i,0], label=r'predicted')
plt.ylabel(r'$\bar x_1$ distance [cm]')

ax2 = plt.subplot(312)
plt.plot(xs[:i,1], label=r'true')
#plt.plot(xs_hat[:i,1], label=r'predicted')
plt.ylabel(r'$\bar x_2$ distance [cm]')

ax3 = plt.subplot(313)
plt.plot(xs[:i,2], label=r'true')
#plt.plot(xs_hat[:i,2], label=r'predicted')
plt.xlabel(r'action step')
plt.ylabel(r'$\bar x_3$ distance [cm]')
plt.legend()
plt.show()
