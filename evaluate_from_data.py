import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym
from cem import CEM
#import gym_softreacher
from matplotlib import rc
from multi_env import MultiEnv
import gym_custom
import pickle
rc('text', usetex=True)

model = torch.load('models/real_mpc_test_test.pt')
x_dim = 6#len(env.observation_space.low)
u_dim = 2#len(env.action_space.low)
n = 10
start = n*30
H = 29

xs_hat = np.zeros((H,x_dim))
r_tot = 0

with open('data_train_test.pkl', 'rb') as f:
    traj_dict = pickle.load(f)
states = traj_dict['states']
controls = traj_dict['controls']
news = traj_dict['news']
print(len(states))
xs = states[start:start+H]
u = controls[start:start+H]
news = news[start:start+H]
#xs_hat[0,:] = xs[0,:]
def plot_frame(xs, i, target):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xyz = xs[:i+1,:3]
    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2])
    ax.scatter(target[0], target[1], target[2], s=200, c='orange',label='goal')
    fig.legend()
    fig.savefig('extend/'+str(i) + '.png')
    plt.close()

for i in range(1,H):
    if news[i-1]:
        xs_hat[i-1,:] = xs[i-1,:] 
    print(u[i-1,:])
    x_hat_next = model.predict(xs_hat[i-1,:], u[i-1,:])
    #plot_frame(xs, i, env.target)
    xs_hat[i,:] = x_hat_next


plt.style.use('ggplot')
ax1 = plt.subplot(511)
plt.plot(xs[:,0], label=r'true')
plt.plot(xs_hat[:,0], label=r'predicted')
plt.ylabel(r'$\bar x_1$ [cm]')

ax2 = plt.subplot(512)
plt.plot(xs[:,1], label=r'true')
plt.plot(xs_hat[:,1], label=r'predicted')
plt.ylabel(r'$\bar x_2$ [cm]')

ax3 = plt.subplot(513)
plt.plot(xs[:,2], label=r'true')
plt.plot(xs_hat[:,2], label=r'predicted')
plt.xlabel(r'action step')
plt.ylabel(r'$\bar x_3$ [cm]')

ax1 = plt.subplot(514)
plt.plot(xs[:,3], label=r'true')
plt.plot(xs_hat[:,3], label=r'predicted')
plt.ylabel(r'$P_b$')

ax2 = plt.subplot(515)
plt.plot(xs[:,4], label=r'true')
plt.plot(xs_hat[:,4], label=r'predicted')
plt.ylabel(r'$P_r$')

plt.legend()
plt.show()
