from torch.utils.data.dataset import Dataset
import numpy as np
from collections import deque
from itertools import islice
import matplotlib.pyplot as plt
import time
from gym_brt.envs import QubeSwingupEnv

class TrajDataset(Dataset):
    def __init__(self, traj_dict, T):
    
        self.initial_states, self.input_controls, self.out_states = self.parse_traj(traj_dict, T)

    def __getitem__(self, index):
        return self.initial_states[index], self.input_controls[index], self.out_states[index]

    def __len__(self):
        return len(self.initial_states)
    

    def parse_traj(self, traj_dict, T):
        states  = traj_dict['states']
        news    = traj_dict['news']
        controls = traj_dict['controls']

        states_deq = deque(maxlen=T)
        controls_deq = deque(maxlen=T)
        initial_states = []
        input_controls = []
        out_states = []
        for i in range(len(states)):
            if news[i]:
                states_deq.clear()
                controls_deq.clear()

            states_deq.append(states[i])
            controls_deq.append(controls[i])

            if len(states_deq) == T:
                initial_states.append(states_deq[0])
                input_controls.append(list(islice(controls_deq,0,T-1)))
                out_states.append(list(islice(states_deq,1, T)))
        
        return np.array(initial_states), np.array(input_controls), np.array(out_states)

def uhlenbeck_sample(env, u_last, theta=0.05, sigma=0.6):
    dim = len(u_last)
    u = (u_last + -theta*u_last + sigma*env.np_random.normal(0,1,dim)).astype(np.float32)
    return np.clip(u, a_min=env.action_space.low, a_max=env.action_space.high)

def gym_gen(env, T, pi=None, stochastic=False, seed=None, render=False):
    x_dim = len(env.observation_space.low)
    u_dim = len(env.action_space.low)
    states = np.zeros((T, x_dim))
    news = np.zeros(T, 'int32')
    controls = np.zeros((T,u_dim))
    done = False
    u_last = np.zeros(u_dim)
    if seed is not None:
        env.seed(seed)
    for i in range(T):
        if i==0 or done:
            xt = env.reset()
            xtt,_,_,_ = env.step(np.zeros(env.action_space.sample().shape))
            news[i] = 1
        if pi=='random':
            u = env.action_space.sample()
        elif pi=='uhlenbeck':
            u = uhlenbeck_sample(env, u_last)
            u_last = u
        elif pi is not None:
            u,_ = pi.predict(xt, xtt)
            if stochastic:
                u += env.env.np_random.normal(0,0.2,size=(u_dim))
        else:
            u = np.zeros(u_dim)
        if i%500==0:
            print('samples collected: ', i)
        if render:
            env.render()
        states[i,:] = xtt
        controls[i,:] = np.array([u])
        x,r,done,_ = env.step(u)
        xt = xtt
        xtt = x
    env.close()
    return {'states':states, 'controls':controls, 'news':news}

def gym_gen_quan(env, T, pi=None, stochastic=False, seed=None, render=False):
    
    x_dim = 5#len(env.observation_space.low)
    u_dim = 1#len(env.action_space.low)
    states = np.zeros((T, x_dim))
    news = np.zeros(T, 'int32')
    controls = np.zeros((T,u_dim))
    done = False
    u_last = np.zeros(u_dim)
    frequency = 250
    skip = 25
    print(T)
    T_Q = T*skip
    with QubeSwingupEnv(frequency=frequency) as env:
        if seed is not None:
            env.seed(seed)
        for i in range(T_Q):
            if i==0:
                xt = env.reset()
                xtt,_,_,_ = env.step(np.zeros(env.action_space.sample().shape))
                news[i] = 1
            if i%skip==0:
                print(i)
                if pi=='random':
                    u = env.action_space.sample()
                elif pi=='uhlenbeck':
                    u = uhlenbeck_sample(env, u_last)
                    u_last = u
                elif pi is not None:
                    u,_ = pi.predict(xt, xtt)
                    if stochastic:
                        u += np.random.normal(0,0.2,size=(u_dim))
                else:
                    u = np.zeros(u_dim)
                if i%500==0:
                    print('samples collected: ', i)

                states[i//skip,:] = xtt
                controls[i//skip,:] = np.array([u])

            x,r,done,_ = env.step(u)
            xt = xtt
            xtt = x
    
    return {'states':states, 'controls':controls, 'news':news}
