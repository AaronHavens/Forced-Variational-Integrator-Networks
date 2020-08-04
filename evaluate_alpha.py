import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from multi_env import MultiEnv
from mpl_toolkits.mplot3d import Axes3D
from evaluate_utils import evaluate
rc('text', usetex=True)

fname1 = 'models/vv_predict_cart.pt'
fname2 = 'models/vv_alpha_quan.pt'
envname = 'CartpoleMod-v0'
seed = 5
H = 50
pi_H = 10
pi = None#'mpc'


xs1, xs_hat1, us1, cost1 = evaluate(fname1, envname, seed=seed, 
                                    pi=pi, pi_H=pi_H, H=H)
xs2, xs_hat2, us2, cost2 = evaluate(fname1, envname, seed=seed, 
                                    pi=pi, pi_H=pi_H, H=H, alpha=0.0)
xs3, xs_hat3, us3, cost3 = evaluate(fname1, envname, seed=seed, 
                                    pi=pi, pi_H=pi_H, H=H, alpha=0.5)
xs4, xs_hat4, us4, cost4 = evaluate(fname1, envname, seed=seed, 
                                    pi=pi, pi_H=pi_H, H=H, alpha=1.5)



#ucum1 = np.cumsum(np.linalg.norm(us1,axis=1))
#ucum2 = np.cumsum(np.linalg.norm(us2,axis=1))

plt.style.use('ggplot')
fig = plt.figure(constrained_layout=False)
gs = fig.add_gridspec(3,1)
#ax1 = fig.add_subplot(gs[:,1])
##plt.plot(xs[:,0], c='black',label=r'true')
#e1 = xs_hat1 - xs1
#e2 = xs_hat2 - xs2
#plt.plot(np.linalg.norm(e1,axis=1), c='r',label=r'ResNN')
#plt.plot(np.linalg.norm(e2,axis=1), c='c',label=r'F-VIN')
#plt.xlabel(r'time step')
#plt.ylabel(r'$l_2$ error')
#
ax3 = fig.add_subplot(gs[0,0])
plt.plot(xs1[:,0], c='b', linestyle='--')#,label=r'$\alpha=1.0$')
plt.plot(xs2[:,0], c='r', linestyle='--')#,label=r'$\alpha=0.0$')
plt.plot(xs3[:,0], c='c', linestyle='--')#,label=r'$\alpha=-0.3$')
plt.plot(xs4[:,0], c='g', linestyle='--')#,label=r'$\alpha=1.5$')
plt.plot(xs_hat1[:,0], c='b',label=r'$\alpha=1.0$')
plt.plot(xs_hat2[:,0], c='r',label=r'$\alpha=0.0$')
plt.plot(xs_hat3[:,0], c='c',label=r'$\alpha=-0.3$')
plt.plot(xs_hat4[:,0], c='g',label=r'$\alpha=2.0$')
ax3.set_xticklabels([])
plt.ylabel(r'$\cos{\theta}$')
#plt.legend()
ax3.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=2, fancybox=True, shadow=True)
ax4 = fig.add_subplot(gs[1,0])
plt.plot(xs1[:,1], c='b', linestyle='--')#,label=r'$\alpha=1.0$')
plt.plot(xs2[:,1], c='r', linestyle='--')#,label=r'$\alpha=0.0$')
plt.plot(xs3[:,1], c='c', linestyle='--')#,label=r'$\alpha=0.5$')
plt.plot(xs4[:,1], c='g', linestyle='--')#,label=r'$\alpha=2.0$')
plt.plot(xs_hat1[:,1], c='b')
plt.plot(xs_hat2[:,1], c='r')
plt.plot(xs_hat3[:,1], c='c')
plt.plot(xs_hat4[:,1], c='g')
ax4.set_xticklabels([])
plt.ylabel(r'$\sin{\theta}$')

ax5 = fig.add_subplot(gs[2,0])

plt.plot(xs1[:,2], c='b', linestyle='--')
plt.plot(xs2[:,2], c='r', linestyle='--')
plt.plot(xs3[:,2], c='c', linestyle='--')
plt.plot(xs4[:,2], c='g', linestyle='--')
plt.plot(xs_hat1[:,2], c='b')
plt.plot(xs_hat2[:,2], c='r')
plt.plot(xs_hat3[:,2], c='c')
plt.plot(xs_hat4[:,2], c='g')

plt.ylabel(r'$\dot \theta$')

plt.xlabel(r'time step')


plt.show()

