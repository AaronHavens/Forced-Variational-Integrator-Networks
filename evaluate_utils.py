import torch
import numpy as np
import gym
import gym_custom
from VISV_cem import CEM
import pickle
from quad_dynamics import pid

def uhlenbeck_sample(env, u_last, theta=0.1, sigma=0.6):
    dim = len(u_last)
    u = (u_last + -theta*u_last + sigma*env.env.np_random.normal(0,1,dim)).astype(np.float32)
    return np.clip(u, a_min=env.action_space.low, a_max=env.action_space.high)


def action_sample(env):
    return env.env.np_random.uniform(low=env.action_space.low,
                                high=env.action_space.high)

def evaluate(fname, envname, seed=None, init_state=None, pi=None,
                        pi_H=20,H=100, alpha=None):
    env = gym.make(envname)
    model = torch.load(fname)
    if alpha is not None:
        env.env.alpha = alpha
        model.alpha = alpha
    


    x_dim = len(env.observation_space.low) 
    u_dim = len(env.action_space.low) 
    if seed is not None:
        env.seed(seed)
    #mpc = pid(env)#CEM(model, 1000, pi_H, env)
    mpc = CEM(model, 1000, pi_H, env)
    x_t = env.reset()
    if init_state is not None:
        env.env.state = init_state
        x_t = init_state

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
    succ = False
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
    
        R[i-1] = env.env.reward_test(x_t, x_tt, u)

        q_tt, q_t, x_hat_next = model.predict_from_q(q_t, q_tt, u)

        
        x_next, r, done, _ = env.step(u)
        us[i-1,:] = u
        xs[i,:] = x_next
        xs_hat[i,:] = x_hat_next
        env.render()
        x_t = x_tt
        x_tt = x_next
        theta = np.arctan2(x_tt[2], x_tt[1])
        if not succ and np.abs(theta) < 0.1 and np.abs(x_tt[4]) < 0.2:
            succ = True
        #x_hat = x_hat_next
        #print(np.arctan2(x_tt[2], x_tt[1]))
        #print(np.linalg.norm(x_tt - np.array([1,1,0,0])))
        print(x_tt)
        print('total cost: ',np.sum(R[:i-1]))
    
    env.close()
    return xs, xs_hat, us, [np.sum(R), u_cost], succ

def evaluate_data(fname, dataname, start, stop):
    model = torch.load(fname)
    with open(dataname, 'rb') as f:
        traj_dict = pickle.load(f)

    states = traj_dict['states'][start:stop]
    controls = traj_dict['controls'][start:stop]
    H = stop-start-1
    x_dim = states.shape[1]
    u_dim = controls.shape[1]
    
    x_t = states[0,:]
    x_hat = np.copy(x_t)

    x_tt= states[1,:]
    x_hatt = np.copy(x_tt)
    
    xs_hat = np.zeros((H,x_dim))
    xs = states[1:]
    us = controls[1:]
    x_t_ = torch.from_numpy(np.expand_dims(x_hat, axis=0)).float()
    q_t = model.encoder(x_t_)
    x_tt_ = torch.from_numpy(np.expand_dims(x_hatt, axis=0)).float()
    q_tt = model.encoder(x_tt_)

    xs_hat[0,:] = x_tt
    for i in range(1, H):
        u = us[i-1]
        q_tt, q_t, x_hat_next = model.predict_from_q(q_t, q_tt, u)
        xs_hat[i,:] = x_hat_next
    
    return xs, xs_hat

