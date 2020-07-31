import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from VI_model import VI_VV_model, VI_SV_model, Res_model
from matplotlib import rc
from multi_env import MultiEnv
from mpl_toolkits.mplot3d import Axes3D
import gym_custom
from VISV_cem import CEM
from evaluate_utils import evaluate

rc('text', usetex=True)

fname1 = 'models/res_spring.pt'
fname2 = 'models/vv_spring.pt'
envname = 'SpringMass-v0'
seed = 1
H = 50
pi_H = 15

xs1, xs_hat1, us1, cost1 = evaluate(fname1, envname, seed=seed, pi='mpc',
                                    pi_H = pi_H, H=H)
xs2, xs_hat2, us2, cost2 = evaluate(fname2, envname, seed=seed, pi='mpc',
                                    pi_H= pi_H,H=H)
ucum1 = np.cumsum(np.linalg.norm(us1,axis=1))
ucum2 = np.cumsum(np.linalg.norm(us2,axis=1))
print('ResNN: ', cost1)
print('F-VIN: ', cost2)
plt.style.use('ggplot')
fig = plt.figure(constrained_layout=False)
gs = fig.add_gridspec(2,2)
#ax0 = fig.add_subplot(gs[:,1])
#plt.plot(E,label=r'true energy')
#plt.plot(E_hat, label=r'predicted energy')
#plt.ylim([-10,15])
#plt.xlabel(r'time step')
#plt.ylabel(r'energy')
#plt.legend()
#plt.show()
#plt.close()

ax1 = fig.add_subplot(gs[0,0])
#plt.plot(xs[:,0], c='black',label=r'true')
plt.plot(xs1[:,0], c='r',label=r'ResNN')
plt.plot(xs2[:,0], c='c',label=r'F-VIN')

#plt.xlabel(r'time step')
plt.ylabel(r'$x$')
plt.legend()
ax2 = fig.add_subplot(gs[1,0])
#plt.plot(xs1[:,1], c='black',label=r'true')
plt.plot(xs1[:,1], c='r',label=r'ResNN')
plt.plot(xs2[:,1], c='c',label=r'F-VIN')

plt.xlabel(r'time step')
plt.ylabel(r'$y$')

#ax3 = fig.add_subplot(gs[0,1])
#plt.plot(xs1[:,2], c='black', label=r'true')
#plt.plot(xs1[:,2], c='r', label=r'ResNN')
#plt.plot(xs2[:,2], c='c', label=r'F-VIN')
#plt.xlabel(r'time step')
#plt.ylabel(r'$\dot \theta$')

ax4 = fig.add_subplot(gs[1,1])
plt.plot(us1, c='r', label=r'ResNN')
#plt.plot(ucum1, c='r')
plt.plot(us2, c='c', label=r'F-VIN')
#plt.plot(ucum2, c='c')
#plt.legend()
#plt.plot(xs_hat[:i,3], label=r'predicted')
plt.xlabel(r'time step')
plt.ylabel(r'$u$')

plt.show()

