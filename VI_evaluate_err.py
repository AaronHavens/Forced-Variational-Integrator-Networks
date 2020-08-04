import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from evaluate_utils import evaluate
rc('text', usetex=True)

fname1 = 'models/res_predict_cart_25.pt'
fname2 = 'models/vv_predict_cart_25.pt'
envname = 'CartpoleMod-v0'
seed = 5
H = 100
pi_H = 20
pi = None
#pi ='random'
#pi = 'mpc'
xs1, xs_hat1, us1, cost1 = evaluate(fname1, envname, seed=seed, 
                                    pi=pi, pi_H=pi_H, H=H)
xs2, xs_hat2, us2, cost2 = evaluate(fname2, envname, seed=seed, 
                                    pi=pi, pi_H=pi_H, H=H)

plt.style.use('ggplot')
fig = plt.figure(constrained_layout=False)
gs = fig.add_gridspec(5,2)

ax1 = fig.add_subplot(gs[:,1])
#plt.plot(xs[:,0], c='black',label=r'true')
e1 = xs_hat1 - xs1
e2 = xs_hat2 - xs2
plt.plot([], c='black', label='Ground Truth')
plt.plot(np.linalg.norm(e1,axis=1), c='r',label=r'ResNN')
plt.plot(np.linalg.norm(e2,axis=1), c='c',label=r'F-VIN')

plt.xlabel(r'time step')
plt.ylabel(r'$l_2$ error')
plt.legend()



ax3 = fig.add_subplot(gs[0,0])
plt.plot(xs2[:,0], c='black',label=r'Ground Truth')
plt.plot(xs_hat1[:,0], c='r',label=r'ResNN')
plt.plot(xs_hat2[:,0], c='c',label=r'F-VIN')
plt.ylabel(r'$x$')

ax4 = fig.add_subplot(gs[1,0])
plt.plot(xs2[:,1], c='black')
plt.plot(xs_hat1[:,1], c='r')
plt.plot(xs_hat2[:,1], c='c')
plt.ylabel(r'$y$')

ax5 = fig.add_subplot(gs[2,0])
plt.plot(xs1[:,2], c='black')
plt.plot(xs_hat1[:,2], c='r')
plt.plot(xs_hat2[:,2], c='c')
plt.ylabel(r'$x$')

ax6 = fig.add_subplot(gs[3,0])
plt.plot(xs1[:,3], c='black')
plt.plot(xs_hat1[:,3], c='r')
plt.plot(xs_hat2[:,3], c='c')
plt.ylabel(r'$x \theta$')

ax7 = fig.add_subplot(gs[4,0])
plt.plot(xs1[:,4], c='black')
plt.plot(xs_hat1[:,4], c='r')
plt.plot(xs_hat2[:,4], c='c')
plt.ylabel(r'$\dot \theta$')

#ax8 = fig.add_subplot(gs[5,0])
#plt.plot(xs1[:,5], c='black')
#plt.plot(xs_hat1[:,5], c='r')
#plt.plot(xs_hat2[:,5], c='c')
#plt.ylabel(r'$\dot \theta$')


plt.xlabel(r'time step')


plt.show()

