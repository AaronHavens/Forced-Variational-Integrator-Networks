import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from VI_model import VI_VV_model, VI_SV_model, Res_model
#import gym_softreacher
from matplotlib import rc
from multi_env import MultiEnv
from mpl_toolkits.mplot3d import Axes3D
import gym_custom
from VISV_cem import CEM
#from model_env import ModelEnv
import pickle
from core.agent import Agent 
from evaluate_utils import evaluate_data
rc('text', usetex=True)

#env = gym.make('QPendulum-v0')
#env = gym.make('AcrobotPos-v0')
#env = gym.make('SpringMassPos-v0')
#env_model = gym.make('Pendulum-v0')
#env = gym.make('PendulumPos-v0')
#env = gym.make('Acrobot-v1')
#env = gym.make('SoftReacher-v0')
#fname = 'models/vi_acrobot_enc_d2.pt'
fname = 'models/res_10_quan_test.pt'
x_dim = 5#len(env.observation_space.low) 
u_dim = 1#len(env.action_space.low) 

n = 8
xhat1,x1 = evaluate_data(fname, 'quansar_vv_uhlen_long_train.pkl', n*199, n*199+199)

xs = x1
xs_hat = xhat1
plt.style.use('ggplot')
ax1 = plt.subplot(611)
plt.plot(xs[:,0], label=r'true')
plt.plot(xs_hat[:,0], label=r'predicted')

ax2 = plt.subplot(612)
plt.plot(xs[:,1], label=r'true')
plt.plot(xs_hat[:,1], label=r'predicted')

ax3 = plt.subplot(613)
plt.plot(xs[:,2], label=r'true')
plt.plot(xs_hat[:,2], label=r'predicted')

ax4 = plt.subplot(614)
plt.plot(xs[:,3], label=r'true')
plt.plot(xs_hat[:,3], label=r'predicted')
#
#ax5 = plt.subplot(615)
#plt.plot(xs[:i,4], label=r'true')
#plt.plot(xs_hat[:i,4], label=r'predicted')
#
ax6 = plt.subplot(616)
e = xs-xs_hat
plt.plot(np.linalg.norm(e,axis=1), label=r'true')
#plt.plot(xs_hat[:i,5], label=r'predicted')


#plt.legend()
plt.show()
