import gym
#import roboschool
from gym import spaces
from gym.utils import seeding
#import dm_control2gym
import numpy as np
from os import path
#from donutWorld import DonutWorld
#from stable_baselines import SAC

class MultiEnv(gym.Env):

    def __init__(self, env_name, n=1):
        self.n = n
        self.R = 1
        self.env_per_row = 4
        self.envs = []
        #self.pi = SAC.load('sac_cheetah_iclr_base') 
        for i in range(n):
            self.envs.append(gym.make(env_name))
            #self.envs.append(dm_control2gym.make(env_name,task_name='run'))
        high = np.repeat(self.envs[0].observation_space.high,n,axis=0)
        
        #high = np.concatenate((high, np.ones(50)),axis=0)
        self.action_space = self.envs[0].action_space
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.last_ob = np.zeros((self.n,len(self.envs[0].observation_space.low)))
        self.seed()

    def seed(self, seed=None):
        for i in range(self.n):
            self.envs[i].seed(seed)
        return [seed]

    def step(self,u):
        sum_r = 0.
        for k in range(self.R):
            s,r,done,_ = self.envs[0].step(u)
            sum_r += r

        for i in range(1, self.n):
            action, _states = self.pi.predict(self.last_ob[i-1])
            for k in range(self.R):
                si,ri,donei,_ = self.envs[i].step(action)
            self.last_ob[i-1] = si
            s = np.concatenate((s,si),axis=0)
    
        return s, sum_r, done, {}

    def reset(self):
        s = self.envs[0].reset()
        for i in range(1,self.n):
            self.last_ob[i-1] = self.envs[i].reset()
            s = np.concatenate((s,self.last_ob[i-1]),axis=0)
        return s

    def render(self, mode='human'):
        if mode=='rgb_array':
            #self.envs[0].close()
            dims = (500,500)
            rows = int(np.ceil(self.n/self.env_per_row))
            rp = dims[0]
            cp = dims[1]
            remain = self.n%self.env_per_row
            cols = min(self.n, self.env_per_row)
            frame = np.zeros((rp*rows, cp*cols,3))
            for i in range(rows):
                #if i == rows-1:
                #    cols = remain
                for j in range(cols):
                    framei = self.envs[i*rows + j].render(mode=mode)
                    #self.envs[i*rows +j].close()
                    frame[i*rp:i*rp+rp, j*cp:j*cp+cp,:] = framei[:,:,:]
            return frame
        #else:
            #for i in range(1,self.n):
            #    resulti = self.envs[i].render(mode=mode)
            #    result = result or resulti
        else:
            return self.envs[0].render(mode=mode)   

    def close(self):
        for i in range(self.n):
            self.envs[i].close()

