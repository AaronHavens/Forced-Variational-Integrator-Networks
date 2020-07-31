import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from multi_env import MultiEnv
from mpl_toolkits.mplot3d import Axes3D
from evaluate_utils import evaluate_data
rc('text', usetex=True)
#fname1 = 'models/res_uhlen_short_alt_quan.pt'
#fname2 = 'models/vv_uhlen_short_alt_quan.pt'
fname1 = 'models/vv_10_policy_quan.pt'
fname2 = 'models/vv_10_policy_quan.pt'
dname = 'quansar_policy_10_short_train.pkl'

n = 0
start = n*199
stop = n*199 + 500
xs1, xs_hat1 = evaluate_data(fname1, dname, start, stop)
xs2, xs_hat2 = evaluate_data(fname2, dname, start, stop) 

plt.style.use('ggplot')
fig = plt.figure(constrained_layout=False)
gs = fig.add_gridspec(5,2)

ax1 = fig.add_subplot(gs[:,1])
#plt.plot(xs[:,0], c='black',label=r'true')
e1 = xs_hat1 - xs1
e2 = xs_hat2 - xs2
plt.plot(np.linalg.norm(e1,axis=1), c='r',label=r'ResNN')
plt.plot(np.linalg.norm(e2,axis=1), c='c',label=r'F-VIN')
plt.plot([], c='black', label=r'Ground Truth')
plt.xlabel(r'time step')
plt.ylabel(r'$l_2$ error')
plt.legend()
ax3 = fig.add_subplot(gs[0,0])
plt.plot(xs2[:,0], c='black',label=r'Ground Truth')
plt.plot(xs_hat1[:,0], c='r',label=r'ResNN')
plt.plot(xs_hat2[:,0], c='c',label=r'F-VIN')
plt.ylabel(r'$\cos{\theta_1}$')
#plt.legend()

ax4 = fig.add_subplot(gs[1,0])
plt.plot(xs1[:,1], c='black')
plt.plot(xs_hat1[:,1], c='r')
plt.plot(xs_hat2[:,1], c='c')
plt.ylabel(r'$\sin{\theta_1}$')

ax5 = fig.add_subplot(gs[2,0])
plt.plot(xs1[:,2], c='black')
plt.plot(xs_hat1[:,2], c='r')
plt.plot(xs_hat2[:,2], c='c')
plt.ylabel(r'$\dot \theta_2$')

ax5 = fig.add_subplot(gs[3,0])
plt.plot(xs1[:,4], c='black')
plt.plot(xs_hat1[:,4], c='r')
plt.plot(xs_hat2[:,4], c='c')
plt.ylabel(r'$\dot \theta_1$')

ax5 = fig.add_subplot(gs[4,0])
plt.plot(xs1[:,3], c='black')
plt.plot(xs_hat1[:,3], c='r')
plt.plot(xs_hat2[:,3], c='c')
plt.ylabel(r'$\dot \theta_2$')


plt.xlabel(r'time step')


plt.show()

