import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def multistep_predict_loss(model, x0, u, x):
    loss = 0
    H = x.shape[1]
    x_hat = x0
    for i in range(H):
        x_next_hat = model(x_hat, u[:,i])
        loss += 1/H*torch.mean(torch.pow(x_next_hat - x[:,i], 2))
        x_hat = x_next_hat
    
    return loss


def imitation_loss(model,x, u):
    return torch.mean(torch.pow(model(x)-u, 2))


class ImitatePolicy(nn.Module):

    def __init__(self, x_dim, u_dim, hid_units):
        super(ImitatePolicy, self).__init__()
        
        self.f1  = nn.Linear(x_dim, hid_units)
        self.f2  = nn.Linear(hid_units, hid_units)
        self.out  = nn.Linear(hid_units, u_dim, bias=True)
    
    def predict(self, x_):

        x = np.expand_dims(x_,axis=0)
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            u = self.forward(x)
        return u.numpy()[0]
    
    def forward(self, x):
        h1 = F.relu(self.f1(x.float()))
        h2 = F.relu(self.f2(h1))
        out = self.out(h2)

        return out.double()


class ResPredict(nn.Module):

    def __init__(self, x_dim, u_dim, hid_units):
        super(ResPredict, self).__init__()
        
        #self.A = nn.Linear(x_dim, x_dim)
        #self.B = nn.Linear(u_dim, x_dim)
        self.f1  = nn.Linear(x_dim+u_dim, hid_units)
        self.f2  = nn.Linear(hid_units, hid_units)
        self.out  = nn.Linear(hid_units, x_dim)
    
    def predict(self, x_, u_):

        x = np.expand_dims(x_,axis=0)
        u = np.expand_dims(u_,axis=0)
        x = torch.from_numpy(x).float()
        u = torch.from_numpy(u).float()
        with torch.no_grad():
            x_next = self.forward(x, u)
        return x_next.numpy()[0]
    
    def reward_predict(self, x_, u_, target):
        cost = 0
        x = x_.double()
        u = u_.double()
        with torch.no_grad():
            e = x[:, :3]-torch.from_numpy(target)
            cost += torch.norm(e, p=2, dim=1)
            cost += torch.norm(u, p=2, dim=1)
        #with torch.no_grad():
        #    theta = torch.atan2(x[:,1],x[:,0])
        #    cost += torch.pow(theta,2)
        #    cost += 0.1*torch.pow(x[:,2], 2)
        #    cost += 0.001*torch.pow(u[:,0], 2)
        return -cost.data.numpy()
    
    def rollout(self, x, U, H, target):
        with torch.no_grad():
            for i in range(H):
                u = U[:,i]
                if i == 0:
                    R = self.reward_predict(x, u, target)
                else:
                    R += self.reward_predict(x, u, target)
                x = self.forward(x, u)
        return -R
    
    def forward(self, x, u):
        x_u = torch.cat((x,u), dim=1).float()
        #print(x_u)
        #out = torch.add(self.A(x.float()), self.B(u.float()))
        #out = self.f1(x_u)
        h1 = F.relu(self.f1(x_u))
        h2 = F.relu(self.f2(h1))
        out = torch.add(x.float(), self.out(h2))

        return out.double()



