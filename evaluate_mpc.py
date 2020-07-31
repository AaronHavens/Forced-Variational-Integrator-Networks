import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from multi_env import MultiEnv
from mpl_toolkits.mplot3d import Axes3D
from evaluate_utils import evaluate
rc('text', usetex=True)

fname1 = 'models/res_5_50_up_cart.pt'
fname2 = 'models/vv_5_30_up_cart.pt'
envname = 'CartpoleMod-v0'
seed = None
H = 70
pi = 'mpc'
pi_H = 20
N = 5

theta = np.linspace(-np.pi, np.pi, N)
theta_dot = np.linspace(-2, 2, N)
cost_gridres = np.zeros((N,N))
cost_gridvv = np.zeros((N,N))
ucost_gridres = np.zeros((N,N))
ucost_gridvv = np.zeros((N,N))
succ_count1 = 0
succ_count2 = 0

mean1 = 0
mean2 = 0
for i in range(N):
    for j in range(N):
        #x = np.array([theta[i], theta_dot[j]])
        #x = np.random.uniform(-1,1,size=(4))
        seed = i*N + j
        xs1, xs_hat1, us1, cost1,succ1 = evaluate(fname1, envname, pi=pi,
                                            pi_H=pi_H, H=H,seed=seed)
        xs2, xs_hat2, us2, cost2,succ2 = evaluate(fname2, envname, pi=pi, 
                                            pi_H=pi_H, H=H,seed=seed)
        #theta_final1 = np.arctan2(xs1[-1,2], xs1[-1,1])
        #theta_final2 = np.arctan2(xs2[-1,2], xs2[-1,1])
        succ_count1 += succ1#np.abs(theta_final1) < 0.1
        succ_count2 += succ2#np.abs(theta_final2) < 0.1
        print(succ_count1, succ_count2)
        cost_gridres[i,j] = cost1[0]
        cost_gridvv[i,j] = cost2[0]
        ucost_gridres[i,j] = cost1[1]
        ucost_gridvv[i,j] = cost2[1]
        #cost_grid[i,j] = cost2[0] - cost1[0]
        #ucost_grid[i,j] = cost1[1] - cost2[1]
        #mean1 += cost1[0]/(N*N)
        #mean2 += cost2[0]/(N*N)

print('mean, std cost res:', np.mean(cost_gridres), np.std(cost_gridres))
print('mean, std ucost res:', np.mean(ucost_gridres), np.std(ucost_gridres))
print('mean, std cost vv:', np.mean(cost_gridvv), np.std(cost_gridvv))
print('mean, std ucost vv:', np.mean(ucost_gridvv), np.std(ucost_gridvv))

#np.save('cost_5_csm_res_cost.npy', cost_gridres)
#np.save('ucost_5_csm_vv_cost.npy', ucost_gridvv)
