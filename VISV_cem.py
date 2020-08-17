import gym
import numpy as np
import copy
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
        self.env = env
    def evaluate_rollouts(self, u_):
        with torch.no_grad():
            S_k = 0
            zt = self.z0_init
            ztt = self.z1_init
            u = (torch.from_numpy(u_)).double()
            for i in range(self.T-1):
                u_i = u[:,i]
                r = self.env.env.reward(self.model, zt, ztt, u_i)
                ztt_ = self.model.forward(zt.float(),
                ztt.float(),u_i.float())
                self.S[:] += -r
                zt = ztt
                ztt = ztt_



    def control(self, x_0, x_1):
        self.U_sigma = np.ones((self.T-1, self.u_dim))
        x0 = torch.from_numpy(np.repeat(np.expand_dims(x_0, axis=0), self.K, axis=0)).float()
        x1 = torch.from_numpy(np.repeat(np.expand_dims(x_1, axis=0), self.K, axis=0)).float()

        z0 = self.model.encoder(x0)
        z1 = self.model.encoder(x1)
        self.x_init = x0
        self.z0_init = z0
        self.z1_init = z1
        for opt_iters in range(10):
            actions = np.clip(np.random.normal(loc=self.U_mean, scale=np.square(self.U_sigma), 
                                        size=(self.K, self.T-1, self.u_dim)),self.u_lower_bound,
                                        self.u_upper_bound)
           
            self.evaluate_rollouts(actions.astype(np.float32))

            ind = np.argsort(self.S)
            top_K = actions[ind[:self.topk]]
            self.U_mean = np.mean(top_K,axis=0)
            self.U_sigma = np.std(top_K, axis=0)
            self.S[:] = 0

        u_0 = self.U_mean[0,:]
        self.U_mean = np.roll(self.U_mean, -1)
        self.U_mean[-1,:] = self.u_init
        return u_0

    def predict(self, x0, x1):
        return np.array(self.control(x0, x1),dtype=np.float32), None
