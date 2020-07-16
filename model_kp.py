import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def VI_loss(model, x0, x):
    loss = 0
    H = x.shape[1]
    q_hat = model.encoder(x0)
    q_hatt = model.encoder(x[:,0])
        #loss = torch.mean(torch.pow(x_next_hat - x[:,0], 2))
   # for i in range(H):
   #     q_next_hat = model(q_hat.float()).double()
   #     q_next = model.encoder(x[:,i]).double()
   #     x_next_hat = model.decoder(q_next_hat.float())
   #     #loss += 1/H*torch.mean(torch.pow(q_next_hat - q_next, 2))
   #     loss += 1/H*torch.mean(torch.pow(x_next_hat - x[:,i], 2))
   #     q_hat = q_next_hat
    x0_hat = model.decoder(q_hat.float())
    x1_hat = model.decoder(q_hatt.float())
    loss += torch.mean(torch.pow(x0_hat - x0, 2))
    loss += torch.mean(torch.pow(x1_hat - x[:,0], 2))
    #lam = np.float32(0.95)
    for i in range(1,H):
        q_next_hat = model(q_hat.float(), q_hatt.float()).double()
        #q_next = model.encoder(x[:,i]).double()
        x_next_hat = model.decoder(q_next_hat.float())
        #loss += torch.mean(torch.pow(q_next_hat - q_next, 2))
        #loss += 1/H*lam**(i-1)*
        loss += torch.mean(torch.pow(x_next_hat - x[:,i], 2))
        q_hat = q_hatt
        q_hatt = q_next_hat

    return loss


class KoopmanModel(nn.Module):

    def __init__(self, encoder, q_dim, h=0.1):
        super(VI_model, self).__init__()

        self.q_dim = q_dim
        self.f1 = nn.Linear(q_dim, q_dim)
        self.fout = nn.Linear(q_dim, q_dim)
        self.encoder = encoder
        self.decoder = decoder

    def predict_from_x(self, x_t_, x_tt_):
        x_t = np.expand_dims(x_t_, axis=0)
        x_t = torch.from_numpy(x_t).float()
        x_tt = np.expand_dims(x_tt_, axis=0)
        x_tt = torch.from_numpy(x_tt).float()
        with torch.no_grad():
            q_t = self.encoder(x_t)
            q_tt = self.encoder(x_tt)
            q_next = self.forward(q_t, q_tt)
            x_next_hat = self.decoder(q_next)
        return q_tt, q_next, x_next_hat.numpy()[0]

    def predict_from_q(self, q_t, q_tt):
        with torch.no_grad():
            q_next = self.forward(q_t, q_tt)
            x_next_hat = self.decoder(q_next)
        return q_next, x_next_hat.numpy()[0]


    def forward(self, q_t, q_tt):
        #q_ttt = np.float(2.0)*q_tt - q_t - self.h*(q_tt+ self.fout(torch.tanh(self.f1(q_tt))))
        q_ttt = np.float(2.0)*q_tt - q_t - self.fout(torch.relu(self.f1(q_t)))

        return q_ttt
        #q_t = q[:,:self.q_dim]
        #q_dot_t = q[:,-self.q_dim:]
        #q_dot_tt = torch.add(q_dot_t, torch.add(q_t, self.h*self.fout(F.relu(self.f1(q_t)))))
        #q_tt = torch.add(q_t, self.h*q_dot_tt)

        #return torch.cat((q_tt, q_dot_tt), axis=1)

class Encoder(nn.Module):
    
    def __init__(self, x_dim, q_dim, hid_units):
        super(Encoder, self).__init__()
        
        self.z1 = nn.Linear(x_dim, hid_units)
        self.z2 = nn.Linear(hid_units, hid_units)
        self.zout = nn.Linear(hid_units, q_dim)
        
    def forward(self, x_):
        x = x_.float()
        h1 = F.relu(self.z1(x))
        h2 = F.relu(self.z2(h1))
        return self.zout(h2)

        
