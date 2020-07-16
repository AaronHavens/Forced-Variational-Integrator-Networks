import gym
import numpy as np
import copy
#import dm_control2gym
#from multi_env import MultiEnv
#from multiprocessing import Pool
import os
import torch

class CEM():

    def __init__(self, model, K, T, env, u_init = 0, base=False):
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.u_lower_bound = env.action_space.low
        self.u_upper_bound = env.action_space.high
        self.s_lower_bound = torch.from_numpy(env.observation_space.low).double()
        self.s_upper_bound = torch.from_numpy(env.observation_space.high).double()
        self.u_dim = len(self.u_upper_bound)
        self.U_mean = np.zeros((self.T-1, self.u_dim))
        self.U_sigma = np.ones((self.T-1, self.u_dim))
        self.u_init = u_init
        self.cost_total = np.zeros(shape=(self.K))
        self.topk = int(K//10)
        self.model = model
        self.is_baseline = base
        self.S = np.zeros(self.K)        
   
    def evaluate_rollouts(self, u_):
        #if True:
        with torch.no_grad():
            S_k = 0
            zt = self.z0_init
            ztt = self.z1_init
            u = (torch.from_numpy(u_)).double()
            for i in range(self.T-1):
                u_i = u[:,i]
                #if i==0:
                #    x_hat = self.x_init
                #else:
                #    x_hat = self.model.decoder(z)
                r = self.model.reward(zt, ztt, u_i)
                ztt_ = self.model.forward(zt.float(), ztt.float(), u_i.float())
                self.S[:] += -r
                zt = ztt
                ztt = ztt_
                #if i==0:
                    #print(self.S, u)



    def control(self, x_0, x_1):
        self.U_mean = np.zeros((self.T-1, self.u_dim))
        self.U_sigma = np.ones((self.T-1, self.u_dim))
        #z0 = self.model.encoder(torch.from_numpy(np.expand_dims(x_0, axis=0)))
        x0 = torch.from_numpy(np.repeat(np.expand_dims(x_0, axis=0), self.K, axis=0)).float()
        x1 = torch.from_numpy(np.repeat(np.expand_dims(x_1, axis=0), self.K, axis=0)).float()

        z0 = self.model.encoder(x0)
        z1 = self.model.encoder(x1)
        self.x_init = x0
        self.z0_init = z0
        self.z1_init = z1
        #self.z_init = z0.repeat(self.K, 1)
        for opt_iters in range(5):
            actions = np.clip(np.random.normal(loc=self.U_mean, scale=np.square(self.U_sigma), 
                                        size=(self.K, self.T-1, self.u_dim)),self.u_lower_bound,
                                        self.u_upper_bound)
           
            self.evaluate_rollouts(actions.astype(np.float32))

            ind = np.argsort(self.S)
            top_K = actions[ind[:self.topk]]
            self.U_mean = np.mean(top_K,axis=0)
            #self.U_sigma = np.std(top_K, axis=0)
            self.S[:] = 0

        u_0 = self.U_mean[0,:]
        #self.U_mean = np.roll(self.U_mean, -1)
        #self.U_mean[-1,:] = self.u_init
        return u_0

    def predict(self, x0, x1):
        #self.target = target
        return np.array(self.control(x0, x1),dtype=np.float32), None
