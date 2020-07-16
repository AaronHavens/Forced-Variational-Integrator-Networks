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
from model_env import ModelEnv
import pickle
from core.agent import Agent 
rc('text', usetex=True)

#env = gym.make('QPendulum-v0')
#env = gym.make('AcrobotPos-v0')
#env = gym.make('SpringMassPos-v0')
#env_model = gym.make('Pendulum-v0')
#env = gym.make('PendulumPos-v0')
#env = gym.make('Acrobot-v1')
#env = gym.make('SoftReacher-v0')
#fname = 'models/vi_acrobot_enc_d2.pt'
fname = 'models/vvi_quas_wat.pt'
x_dim = 2#len(env.observation_space.low) 
u_dim = 1#len(env.action_space.low) 

with open('quansar_nof_long_data_train.pkl', 'rb') as f:
    traj_dict = pickle.load(f)
states = traj_dict['states']
controls = traj_dict['controls']

model = torch.load(fname) 
start = 0
H = 400
T = 1000
#x_tt = states[1,:]#,_,_,_ = env.step(np.zeros(np.shape(env.action_space.sample())))
#x_hatt = np.copy(x_tt)
xs_hat = np.zeros((T,x_dim))
xs = states[start:start+H]
u = controls[start:start+H]
x_t_ = torch.from_numpy(np.expand_dims(xs[0,:], axis=0)).float()
q_t = model.encoder(x_t_)
#x_tt_ = torch.from_numpy(np.expand_dims(x_hatt, axis=0)).float()
#q_tt = model.encoder(x_tt_)
xs_hat[0,:] = xs[0,:]#model.decoder(q_t1.float()).detach().numpy()

for i in range(1, T):
    u_t = np.zeros(1)#u[i-1]
    q_t, x_hat_next = model.predict_from_q(q_t, u_t)
    xs_hat[i,:] = x_hat_next

plt.style.use('ggplot')
plt.plot(states)
plt.show()
ax1 = plt.subplot(211)
plt.plot(xs[:i,0], label=r'true')
plt.plot(xs_hat[:i,0], label=r'predicted')

ax2 = plt.subplot(212)
plt.plot(xs[:i,1], label=r'true')
plt.plot(xs_hat[:i,1], label=r'predicted')

#ax3 = plt.subplot(413)
#plt.plot(xs[:i,2], label=r'true')
#plt.plot(xs_hat[:i,2], label=r'predicted')

#ax4 = plt.subplot(414)
#plt.plot(xs[:i,3], label=r'true')
#plt.plot(xs_hat[:i,3], label=r'predicted')

#ax5 = plt.subplot(515)
#plt.plot(xs[:i,4], label=r'true')
#plt.plot(xs_hat[:i,4], label=r'predicted')

#ax6 = plt.subplot(616)
#plt.plot(xs[:i,5], label=r'true')
#plt.plot(xs_hat[:i,5], label=r'predicted')


plt.legend()
plt.show()
