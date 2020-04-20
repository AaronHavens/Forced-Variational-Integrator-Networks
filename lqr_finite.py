import numpy as np
import torch

class MPC():
    def __init__(self, model, A, B, Q, R, x_dim, u_dim, C=None, T=10, open_loop=False):
        self.A = np.asmatrix(A)
        self.B = np.asmatrix(B)
        self.Q = np.asmatrix(Q)
        self.R = np.asmatrix(R)
        self.T = T
        self.m = u_dim
        self.C = C
        self.model = model
        self.n = x_dim
        self.open_loop = open_loop
        if self.C is not None:
            self.C = np.asmatrix(self.C)
            self.n += 1
            on = np.asmatrix(np.zeros((1,self.n-1)))
            self.A = np.block([[self.A, self.C],[on,1]])
            om = np.asmatrix(np.zeros((1, self.m)))
            self.B = np.block([[self.B],[om]])
            self.Q = np.block([[self.Q,on.T],[on,0]])
        self.precompute_K()

    def precompute_K(self):
        A = self.A
        B = self.B
        Q = self.Q
        R = self.R

        P = np.zeros((self.n, self.n, self.T+1))
        K = np.zeros((self.m, self.n, self.T))
        P[:,:,self.T] = Q
        
        for i in range(self.T-1, -1, -1):
            print(i)
            Pi = P[:,:,i+1]
            K[:,:,i] = -np.linalg.inv(R + B.T*Pi*B)*B.T*Pi*A
            Ki = K[:,:,i]
            P[:,:,i] = Q + A.T*Pi*A + A.T*Pi*B*Ki
        
        self.K = K

        
    def predict(self, x0, i):
        z0 = self.model.encoder(torch.from_numpy(np.expand_dims(x0, axis=0))).detach().numpy()
        z0 = np.asmatrix(z0).reshape(self.n,1)
        if self.C is not None:
            z0 = np.asmatrix(z0).reshape(self.n-1,1)
            z0 = np.block([[z0],[1]])
        if self.open_loop:
            u = self.K[:,:,i]*z0
        else:
            u = self.K[:,:,0]*z0
        return np.asarray(u).reshape(2,)

