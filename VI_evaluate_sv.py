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

def uhlenbeck_sample(env, u_last, theta=0.05, sigma=0.6):
    dim = len(u_last)
    u = (u_last + -theta*u_last + sigma*env.env.np_random.normal(0,1,dim)).astype(np.float32)
    return np.clip(u, a_min=env.action_space.low, a_max=env.action_space.high)


def action_sample(env):
    return env.np_random.uniform(low=env.action_space.low,
                                high=env.action_space.high)

def evaluate(fname, envname, seed=None, pi=None, H=100):
    env = gym.make(envname)
    model = torch.load(fname)


    x_dim = len(env.observation_space.low) 
    u_dim = len(env.action_space.low) 
    if seed is not None:
        env.seed(seed)
    mpc = CEM(model, 1000, 15, env)

    x_t = env.reset()

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

    xs_hat[0,:] = x_tt
    E = np.zeros(H-1)
    E_hat = np.zeros(H-1)
    R = np.zeros(H-1)
    R_hat = np.zeros(H-1)
    us = np.zeros((H-1, u_dim))
    u_cost = 0
    u_last = np.zeros(u_dim)
    for i in range(1, H):
        if pi == 'random':
            u = action_sample(env)
        elif pi == 'uhlenbeck':
            u = uhlenbeck_sample(env, u_last)
        elif pi == 'mpc':
            u, _  = mpc.predict(x_t, x_tt)
        else:
            u = np.zeros(env.action_space.low.shape)
        
        u_last = u
        u_cost += u[0]**2
        print('control cost: ', u_cost)
    
        #R[i-1] = env.env.reward_test(x_t, x_tt, u)

        q_tt, q_t, x_hat_next = model.predict_from_q(q_t, q_tt, u)

        
        x_next, r, done, _ = env.step(u)
        us[i-1,:] = u
        xs[i,:] = x_next
        xs_hat[i,:] = x_hat_next
        env.render()
        #E[i-1] = 1/2*(env.env.state[1]**2) -10*np.cos(env.env.state[0]-np.pi)
        #E[i-1] = 1/2*x_next[1]**2 - 10*np.cos(x_next[0])
        #theta_ = np.arctan2(x_hat_next[1], x_hat_next[0])
        #thetadot_ = x_hat_next[2]
        #E_hat[i-1] = 1/2*thetadot_**2 - 10*np.cos(theta_-np.pi)
        #E_hat[i-1] = 1/2*x_hat_next[1]**2 - 10*np.cos(x_hat_next[0])
        #q_t = q_next
        x_t = x_tt
        x_tt = x_next
        #x_hat = x_hat_next
        print('total cost: ',np.sum(R[:i-1]))
    print(np.sum(R))

    env.close()
    return xs, xs_hat, us

fname1 = 'models/vv_spring.pt'
fname2 = 'models/vv_enc5_pend.pt'
envname = 'SpringMass-v0'
seed = 5
H = 400
pi = 'random'#'mpc'

xs1, xs_hat1, us1 = evaluate(fname1, envname, seed=seed, pi=pi, H=H)
#xs2, xs_hat2, us2 = evaluate(fname2, envname, seed=seed, pi=pi, H=H)


plt.style.use('ggplot')
fig = plt.figure(constrained_layout=False)
gs = fig.add_gridspec(3,2)

ax1 = fig.add_subplot(gs[:,1])
#plt.plot(xs[:,0], c='black',label=r'true')
e1 = xs_hat1 - xs1
#e2 = xs_hat2 - xs2
plt.plot(np.linalg.norm(e1,axis=1), c='r',label=r'ResNN')
#plt.plot(np.linalg.norm(e2,axis=1), c='c',label=r'F-VIN')
plt.xlabel(r'time step')
plt.ylabel(r'$l_2$ error')

ax3 = fig.add_subplot(gs[0,0])
plt.plot(xs1[:,0], c='black',label=r'Ground Truth')
plt.plot(xs_hat1[:,0], c='r',label=r'ResNN')
#plt.plot(xs_hat2[:,0], c='c',label=r'F-VIN')
plt.ylabel(r'$x$')
plt.legend()

ax4 = fig.add_subplot(gs[1,0])
plt.plot(xs1[:,1], c='black')
plt.plot(xs_hat1[:,1], c='r')
#plt.plot(xs_hat2[:,1], c='c')
plt.ylabel(r'$y$')

#ax5 = fig.add_subplot(gs[2,0])
#plt.plot(xs1[:,2], c='black')
#plt.plot(xs_hat1[:,2], c='r')
#plt.plot(xs_hat2[:,2], c='c')
#plt.ylabel(r'$\dot \theta$')

#ax6 = fig.add_subplot(gs[3,0])
#plt.plot(xs1[:,3], c='black')
#plt.plot(xs_hat1[:,3], c='r')
#plt.plot(xs_hat2[:,2], c='c')
#plt.ylabel(r'$\dot \theta$')

#ax7 = fig.add_subplot(gs[4,0])
#plt.plot(xs1[:,4], c='black')
#plt.plot(xs_hat1[:,4], c='r')
#plt.plot(xs_hat2[:,2], c='c')
#plt.ylabel(r'$\dot \theta$')

#ax8 = fig.add_subplot(gs[5,0])
#plt.plot(xs1[:,5], c='black')
#plt.plot(xs_hat1[:,5], c='r')
#plt.plot(xs_hat2[:,2], c='c')


plt.xlabel(r'time step')


plt.show()

