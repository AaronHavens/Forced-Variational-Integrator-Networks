import gym
import numpy as np
import copy
#import dm_control2gym
#from multi_env import MultiEnv
#from multiprocessing import Pool
import os
import torch

class CEM():

    def __init__(self, model, K, T, u_lower_bound, u_upper_bound, u_init = 0, base=False):
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.u_dim = len(u_upper_bound)
        self.U_mean = np.zeros((self.T, self.u_dim))
        self.U_sigma = np.ones((self.T, self.u_dim))
        self.u_init = u_init
        self.cost_total = np.zeros(shape=(self.K))
        self.topk = int(K//10)
        self.model = model
        self.u_upper_bound = u_upper_bound
        self.u_lower_bound = u_lower_bound
        self.is_baseline = base
        self.S = np.zeros(self.K)        
#    def evaluate_rollouts(self, u):
#        S_k = 0
#        n = u.shape[0]
#        z = np.repeat(self.z_init, n, axis=0)
#        for i in range(self.T-1):
#            u_i = u[:,i]
#            if self.is_baseline:
#                x_hat = self.model.decoder(z)
#                r = self.model.reward_predictor(x_hat, u_i).array[:,0]
#            else:
#                r = self.model.reward_predictor(z, u_i).array[:,0]
#            #print(u_i)
#            #print('+++++++++++++')
#            z_next = self.model.dyn_predictor(z, u_i)
#            self.S[:] += -r
#            z = z_next
#

    def evaluate_rollouts(self, u_):
        S_k = 0
        x = self.x_init
        u = (torch.from_numpy(u_)).double()
        for i in range(self.T-1):
            u_i = u[:,i]
            #x_hat = self.model.decoder(z)
            r = self.model.reward_predict(x, u_i)[:]
            #print(u_i)
            #print('+++++++++++++')
            x_next = self.model(x, u_i)
            self.S[:] += -r
            x = x_next


    def control(self, x_0):
        #self.U_mean = np.zeros((self.T, self.u_dim))
        self.U_sigma = np.ones((self.T, self.u_dim))
        self.x_init = torch.from_numpy(np.repeat(np.expand_dims(x_0,axis=0),self.K,axis=0)).double()
        for opt_iters in range(5):
            actions = np.clip(np.random.normal(loc=self.U_mean, scale=np.square(self.U_sigma), 
                                        size=(self.K, self.T, self.u_dim)),self.u_lower_bound,
                                        self.u_upper_bound)
           
            self.evaluate_rollouts(actions.astype(np.float32))

            ind = np.argsort(self.S)
            top_K = actions[ind[:self.topk]]
            self.U_mean = np.mean(top_K,axis=0)
            #self.U_sigma = np.std(top_K, axis=0)
            self.S[:] = 0

        u_0 = self.U_mean[0,:]
        self.U_mean = np.roll(self.U_mean, -1)
        self.U_mean[-1,:] = self.u_init
        return u_0

    def predict(self, x):
        return np.array(self.control(x),dtype=np.float32), None
