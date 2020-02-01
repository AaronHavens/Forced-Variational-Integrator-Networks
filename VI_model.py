import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def VI_loss(model, x0, x):
    loss = 0
    H = x.shape[1]
    q_hat = model.encoder(x0)
        #loss = torch.mean(torch.pow(x_next_hat - x[:,0], 2))
    for i in range(H):
        q_next_hat = model(q_hat.float()).double()
        q_next = model.encoder(x[:,i]).double()
        x_next_hat = model.decoder(q_next_hat.float())
        #loss += 1/H*torch.mean(torch.pow(q_next_hat - q_next, 2))
        loss += 1/H*torch.mean(torch.pow(x_next_hat - x[:,i], 2))
        q_hat = q_next_hat
    return loss


class VI_model(nn.Module):

    def __init__(self, encoder, decoder, q_dim, h=0.1):
        super(VI_model, self).__init__()

        self.h = np.float32(h)
        self.q_dim = q_dim
        self.f1 = nn.Linear(q_dim, q_dim)
        self.fout = nn.Linear(q_dim, q_dim)
        self.encoder = encoder
        self.decoder = decoder

    def predict_from_x(self, x_):
        x = np.expand_dims(x_, axis=0)
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            q = self.encoder(x)
            q_next = self.forward(q)
            x_next_hat = self.decoder(q_next)
        return q_next, x_next_hat.numpy()[0]

    def predict_from_q(self, q):
        with torch.no_grad():
            q_next = self.forward(q)
            x_next_hat = self.decoder(q_next)
        return q_next, x_next_hat.numpy()[0]


    def forward(self, q):
        q_t = q[:,:self.q_dim]
        q_dot_t = q[:,-self.q_dim:]
        q_dot_tt = torch.add(q_dot_t, torch.add(q_t, self.h*self.fout(F.relu(self.f1(q_t)))))
        q_tt = torch.add(q_t, self.h*q_dot_tt)

        return torch.cat((q_tt, q_dot_tt), axis=1)

class Encoder(nn.Module):
    
    def __init__(self, x_dim, q_dim, hid_units):
        super(Encoder, self).__init__()
        
        self.z1 = nn.Linear(x_dim, hid_units)
        self.z2 = nn.Linear(hid_units, hid_units)
        self.zout = nn.Linear(hid_units, 2*q_dim)
        
    def forward(self, x_):
        x = x_.float()
        h1 = F.relu(self.z1(x))
        h2 = F.relu(self.z2(h1))
        return self.zout(h2)

class Decoder(nn.Module):

    def __init__(self, x_dim, q_dim, hid_units):
        super(Decoder, self).__init__()
        
        self.z_inv1 = nn.Linear(2*q_dim, hid_units)
        self.z_inv2 = nn.Linear(hid_units, hid_units)
        self.z_invout = nn.Linear(hid_units, x_dim)

    def forward(self, q):

        h1 = F.relu(self.z_inv1(q))
        h2 = F.relu(self.z_inv2(h1))
        return self.z_invout(h2).double()
        
