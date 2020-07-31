import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from VI_model import VI_SV_model, VI_VV_model, Res_model
#import gym_softreacher
from matplotlib import rc
from multi_env import MultiEnv
from mpl_toolkits.mplot3d import Axes3D
import gym_custom
from VI_cem import CEM
from model_env import ModelEnv
import pickle
from core.agent import Agent 
import time
rc('text', usetex=True)

#env = gym.make('QPendulum-v0')
env = gym.make('Pendulum-v0')
#env_model = gym.make('Pendulum-v0')
#env = gym.make('Acrobot-v1')
#env = gym.make('SoftReacher-v0')
#env = gym.make('SpringMass-v0')
#env = gym.make('QAcrobot-v0')
#env = gym.make('CartpoleMod-v0')
#fname = 'models/vi_acrobot_enc_d2.pt'

fname = 'models/res_enc5_pend.pt'
x_dim = len(env.observation_space.low) 
u_dim = len(env.action_space.low) 

model = torch.load(fname) 
#print(model.frout.weight.data)
#print(model.frout.bias.data)
H = 100
env.seed(15)
pi = CEM(model, 1000, 10, env)

x = env.reset()
x, _,_,_ = env.step(np.array([0]))
#policy_net, value_net, running_state = pickle.load(open("learned_models/trpo/-Pendulum-v0.p", "rb"))
#device = torch.device('cpu')
#agent = Agent(env, policy_net, device, running_state=running_state)

x_hat = np.copy(x)
#x,_,_,_ = env.step(np.zeros(np.shape(env.action_space.sample())))
#x_hatt = np.copy(x)
xs = np.zeros((H,x_dim))
xs_hat = np.zeros((H,x_dim))
xs[0,:] = x
x_t = torch.from_numpy(np.expand_dims(x_hat, axis=0)).float()
q_t = model.encoder(x_t)
xs_hat[0,:] = x#model.decoder(q_t1.float()).detach().numpy()
E = np.zeros(H-1)
E_hat = np.zeros(H-1)
R = np.zeros(H-1)
R_hat = np.zeros(H-1)

for i in range(1, H):
    #q_next, x_hat_next = model.predict_from_q(q_t, q_tt)
    u = np.zeros(env.action_space.low.shape)
    #u, _  = pi.predict(x)
    #u = agent.act(x)
    #u = env.action_space.sample()
    u_ = torch.from_numpy(np.expand_dims(u,axis=0))

    q_next, x_hat_next = model.predict_from_q(q_t, u)
    R_hat[i-1] = model.reward(torch.from_numpy(np.expand_dims(x,axis=0)), u_)[0]
    x_next, r, done, _ = env.step(u)
    R[i-1] = r
    #print(np.arctan2(x[1],x[0]))
    xs[i,:] = x_next
    xs_hat[i,:] = x_hat_next
    #x_hat = x_hat_next
    #q_t = q_tt
    #q_tt = q_next
    env.render()
    #time.sleep(0.05)
    #q_e = q_next.detach().numpy()[0]
    #E[i-1] = 1/2*(env.env.state[1]**2) -10*np.cos(env.env.state[0]-np.pi)
    #E[i-1] = 1/2*x_next[1]**2 - 10*np.cos(x_next[0])
    #theta_ = np.arctan2(x_hat_next[1], x_hat_next[0])
    #thetadot_ = x_hat_next[2]
    #E_hat[i-1] = 1/2*thetadot_**2 - 10*np.cos(theta_-np.pi)
    #E_hat[i-1] = 1/2*x_hat_next[1]**2 - 10*np.cos(x_hat_next[0])
    q_t = q_next
    x = x_next
    x_hat = x_hat_next
    print(np.sum(R_hat[:i-1]))
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

#ax3 = plt.subplot(413)
#plt.plot(xs[:i,2], label=r'true')
#plt.plot(xs_hat[:i,2], label=r'predicted')

#ax4 = plt.subplot(414)
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
