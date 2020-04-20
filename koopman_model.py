import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def multistep_predict_loss(model, x0, u, x):
    loss = 0
    H = x.shape[1]
    x_hat = x0
    z_hat = model.encoder(x0)
    for i in range(H):
        z_next_hat = model(z_hat, u[:,i])
        z_next = model.encoder(x[:,i])
        loss += 1/H*torch.mean(torch.pow(z_next_hat[:,:model.x_dim] - x[:,i], 2))
        loss += 0.0001/H*torch.mean(torch.pow(z_next_hat[:,model.x_dim:] - z_next[:,model.x_dim:], 2))
        z_hat = z_next_hat
    
    return loss


class KoopmanPredict(nn.Module):

    def __init__(self, x_dim, u_dim, z_dim, hid_units):
        super(KoopmanPredict, self).__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.encoder = Encoder(x_dim, z_dim, hid_units)
        self.A = nn.Linear(x_dim+z_dim, x_dim+z_dim, bias=False) #constant C
        self.B = nn.Linear(u_dim, x_dim+z_dim, bias=False)    
    def predict_from_state(self, x_, u_):

        x = np.expand_dims(x_,axis=0)
        u = np.expand_dims(u_,axis=0)
        x = torch.from_numpy(x).float()
        u = torch.from_numpy(u).float()
        with torch.no_grad():
            z = self.encoder(x)
            z_next = self.forward(z, u)
        return z_next.numpy()[0]
    
    def predict_from_latent(self, z_, u_):

        u = np.expand_dims(u_,axis=0)
        u = torch.from_numpy(u).float()
        z = np.expand_dims(z_,axis=0)
        z = torch.from_numpy(z).float()

        with torch.no_grad():
            z_next = self.forward(z, u)
        return z_next.numpy()[0]
    
    def decode(self, z):
        return z[:self.x_dim]

    def reward_predict(self, x_, u_):
        cost = 0
        x = x_.double()
        u = u_.double()
        #with torch.no_grad():
        #    cost += torch.norm(x[:, :3], p=2, dim=1)
        #    cost += 0.001*torch.norm(u, p=2, dim=1)
        with torch.no_grad():
            #cost += torch.pow(x[:,1]-1,2)
            theta = torch.atan2(x[:,1],x[:,0])
            cost += torch.pow(theta,2)
            cost += 0.1*torch.pow(x[:,2], 2)
            cost += 0.001*torch.pow(u[:,0], 2)
        return -cost.data.numpy()


    def forward(self, x, u):
        x = x.float()
        u = u.float()
        out = self.A(x) + self.B(u)

        return out.double()

class Encoder(nn.Module):
    
    def __init__(self, x_dim, z_dim, hid_units):
        super(Encoder, self).__init__()
        
        self.z1 = nn.Linear(x_dim, hid_units, bias=False)
        self.z2 = nn.Linear(hid_units, hid_units, bias=False)
        self.zout = nn.Linear(hid_units, z_dim, bias=False)
        
    def forward(self, x_):
        x = x_.float()
        h1 = F.relu(self.z1(x))
        h2 = F.relu(self.z2(h1))
        return torch.cat((x,self.zout(h2)), 1).double()



