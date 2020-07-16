import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from VI_model import VI_VV_model, VI_SV_model, Res_model
import gym_softreacher
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
env = gym.make('AcrobotPos-v0')
#env = gym.make('SpringMassPos-v0')
#env_model = gym.make('Pendulum-v0')
#env = gym.make('PendulumPos-v0')
#env = gym.make('Acrobot-v1')
#env = gym.make('SoftReacher-v0')
#fname = 'models/vi_acrobot_enc_d2.pt'
fname = 'models/vi_acr_encode_iterative.pt'
x_dim = len(env.observation_space.low) 
u_dim = len(env.action_space.low) 

#model = VI_VV_model(x_dim, q_dim, u_dim=u_dim, h=h, encoder=True) 
#model = Res_model(x_dim, q_dim, u_dim=u_dim, h=h, encoder=False)
model = torch.load(fname) 
#print(checkpoint)
#model.load_state_dict(checkpoint)

H = 100
env.seed(15)
pi = CEM(model, 1000, 10, env)

x_t = env.reset()

#policy_net, value_net, running_state = pickle.load(open("learned_models/trpo/-Pendulum-v0.p", "rb"))
#device = torch.device('cpu')
#agent = Agent(env, policy_net, device, running_state=running_state)

x_hat = np.copy(x_t)
x_tt,_,_,_ = env.step(np.zeros(np.shape(env.action_space.sample())))
x_hatt = np.copy(x_tt)
xs = np.zeros((H,x_dim))
xs_hat = np.zeros((H,x_dim))
xs[0,:] = x_tt
x_t_ = torch.from_numpy(np.expand_dims(x_hat, axis=0)).float()
q_t = model.encoder(x_t_)
x_tt_ = torch.from_numpy(np.expand_dims(x_hatt, axis=0)).float()
q_tt = model.encoder(x_tt_)
xs_hat[0,:] = x_tt#model.decoder(q_t1.float()).detach().numpy()
E = np.zeros(H-1)
E_hat = np.zeros(H-1)
R = np.zeros(H-1)
R_hat = np.zeros(H-1)

for i in range(1, H):
    #q_next, x_hat_next = model.predict_from_q(q_t, q_tt)
    #u = np.zeros(env.action_space.low.shape)
    #u, _  = pi.predict(x_t, x_tt)
    #u = agent.act(x)
    u = env.action_space.sample()
    #u_ = torch.from_numpy(np.expand_dims(u,axis=0))

    q_tt, q_t, x_hat_next = model.predict_from_q(q_t, q_tt, u)
    #R_hat[i-1] = model.pendulum_reward(torch.from_numpy(np.expand_dims(x_hat,axis=0)), u_)[0]
    x_next, r, done, _ = env.step(u)
    R[i-1] = r
    #print(np.arctan2(x[1],x[0]))
    xs[i,:] = x_next
    xs_hat[i,:] = x_hat_next
    #x_hat = x_hat_next
    #q_t = q_tt
    #q_tt = q_next
    env.render()
    #q_e = q_next.detach().numpy()[0]
    #E[i-1] = 1/2*(env.state[1]**2) -10*np.cos(env.state[0]-np.pi)
    #E[i-1] = 1/2*x_next[1]**2 - 10*np.cos(x_next[0])
    #theta_ = np.arctan2(x_hat_next[1], x_hat_next[0])
    #thetadot_ = env.state[1]
    #E_hat[i-1] = 1/2*thetadot_**2 - 10*np.cos(theta_-np.pi)
    #E_hat[i-1] = 1/2*x_hat_next[1]**2 - 10*np.cos(x_hat_next[0])
    #q_t = q_next
    x_t = x_tt
    x_tt = x_next
    #x_hat = x_hat_next
    #print(np.sum(R[:i-1]))
    #if done:
    #    break


#plt.plot(E,label='true energy')
#plt.plot(E_hat, label='predicted energy')
#plt.ylim([-10,15])
#plt.legend()
#plt.show()
#plt.close()

plt.style.use('ggplot')
ax1 = plt.subplot(211)
plt.plot(xs[:i,0], label=r'true')
plt.plot(xs_hat[:i,0], label=r'predicted')

ax2 = plt.subplot(212)
plt.plot(xs[:i,1], label=r'true')
plt.plot(xs_hat[:i,1], label=r'predicted')

#ax3 = plt.subplot(613)
#plt.plot(xs[:i,2], label=r'true')
#plt.plot(xs_hat[:i,2], label=r'predicted')

#ax4 = plt.subplot(614)
#plt.plot(xs[:i,3], label=r'true')
#plt.plot(xs_hat[:i,3], label=r'predicted')

#ax5 = plt.subplot(615)
#plt.plot(xs[:i,4], label=r'true')
#plt.plot(xs_hat[:i,4], label=r'predicted')

#ax6 = plt.subplot(616)
#plt.plot(xs[:i,5], label=r'true')
#plt.plot(xs_hat[:i,5], label=r'predicted')

#ax7 = plt.subplot(717)
#plt.plot(xs[:i,6], label=r'true')
#plt.plot(xs_hat[:i,6], label=r'predicted')


plt.legend()
plt.show()
