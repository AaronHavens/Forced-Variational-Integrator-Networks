import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from multi_env import MultiEnv
from mpl_toolkits.mplot3d import Axes3D
from evaluate_utils import evaluate
rc('text', usetex=True)

fname1 = 'models/res_enc_10_new_pend.pt'
fname2 = 'models/vv_enc_10_new_pend.pt'
envname = 'Pendulum-v0'
seed = 5
H = 400
pi_H = 15
pi = 'random'#'mpc'


xs1, xs_hat1, us1, cost1 = evaluate(fname1, envname, seed=seed, 
                                    pi=pi, pi_H=pi_H, H=H)
xs2, xs_hat2, us2, cost2 = evaluate(fname2, envname, seed=seed, 
                                    pi=pi, pi_H=pi_H, H=H)

pi = None

xs3, xs_hat3, us3, cost3 = evaluate(fname1, envname, seed=seed, 
                                    pi=pi, pi_H=pi_H, H=H)
xs4, xs_hat4, us4, cost4 = evaluate(fname2, envname, seed=seed, 
                                    pi=pi, pi_H=pi_H, H=H)






plt.style.use('ggplot')
fig = plt.figure(constrained_layout=False)
gs = fig.add_gridspec(3,4)

ax1 = fig.add_subplot(gs[:,1])
#plt.plot(xs[:,0], c='black',label=r'true')
e1 = xs_hat1 - xs1
e2 = xs_hat2 - xs2
plt.plot(np.linalg.norm(e1,axis=1), c='r',label=r'ResNN')
plt.plot(np.linalg.norm(e2,axis=1), c='c',label=r'F-VIN')
plt.xlabel(r'time step')
plt.ylabel(r'$l_2$ error')

ax3 = fig.add_subplot(gs[0,0])
plt.plot(xs2[:,0], c='black',label=r'Ground Truth')
plt.plot(xs_hat1[:,0], c='r',label=r'ResNN')
plt.plot(xs_hat2[:,0], c='c',label=r'F-VIN')
plt.ylabel(r'$\cos{\theta}$')

ax4 = fig.add_subplot(gs[1,0])
plt.plot(xs2[:,1], c='black')
plt.plot(xs_hat1[:,1], c='r')
plt.plot(xs_hat2[:,1], c='c')
plt.ylabel(r'$\sin{\theta}$')

ax5 = fig.add_subplot(gs[2,0])
plt.plot(xs2[:,2], c='black')
plt.plot(xs_hat1[:,2], c='r')
plt.plot(xs_hat2[:,2], c='c')
plt.ylabel(r'$\dot \theta$')

plt.xlabel(r'time step')

ax6 = fig.add_subplot(gs[:,3])
#plt.plot(xs[:,0], c='black',label=r'true')
e3 = xs_hat3 - xs3
e4 = xs_hat4 - xs4
plt.plot(np.linalg.norm(e3,axis=1), c='r',label=r'ResNN')
plt.plot(np.linalg.norm(e4,axis=1), c='c',label=r'F-VIN')
plt.xlabel(r'time step')
plt.ylabel(r'$l_2$ error')

ax7 = fig.add_subplot(gs[0,2])
plt.plot(xs3[:,0], c='black',label=r'Ground Truth')
plt.plot(xs_hat3[:,0], c='r',label=r'ResNN')
plt.plot(xs_hat4[:,0], c='c',label=r'F-VIN')
plt.ylabel(r'$\cos{\theta}$')
plt.legend()
ax7.legend(loc='lower center', bbox_to_anchor=(0.0, 1.05),
          ncol=3, fancybox=True, shadow=True)


ax8 = fig.add_subplot(gs[1,2])
plt.plot(xs3[:,1], c='black')
plt.plot(xs_hat3[:,1], c='r')
plt.plot(xs_hat4[:,1], c='c')
plt.ylabel(r'$\sin{\theta}$')

ax9 = fig.add_subplot(gs[2,2])
plt.plot(xs3[:,2], c='black')
plt.plot(xs_hat3[:,2], c='r')
plt.plot(xs_hat4[:,2], c='c')
plt.ylabel(r'$\dot \theta$')

plt.xlabel(r'time step')


plt.show()

