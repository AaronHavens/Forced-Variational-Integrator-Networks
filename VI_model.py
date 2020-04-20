import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def VI_SV_loss(model, x0, x):
    loss = 0
    H = x.shape[1]
    q_hat = x0
    q_hatt = x[:,0]
   #     q_hat = q_next_hat
    #lam = np.float32(0.95)
    for i in range(1,H):
        q_next_hat = model(q_hat.float(), q_hatt.float()).double()
        loss += torch.mean(torch.pow(q_next_hat - x[:,i,0], 2))
        q_hat = q_hatt
        q_hatt = q_next_hat

    return loss

def VI_VV_loss(model, x0, x, u=None):
    loss = 0
    H = x.shape[1]
    q_hat = x0
   #     q_hat = q_next_hat
    #lam = np.float32(0.95)
    for i in range(H):
        if u is not None:
            q_next_hat = model(q_hat.float(), u[:,i].float()).double()
        else:
            q_next_hat = model(q_hat.float()).double()
        loss += torch.mean(torch.pow(q_next_hat - x[:,i], 2))
        q_hat = q_next_hat

    return loss

class Res_model(nn.Module):

    def __init__(self, q_dim, u_dim=1, h=0.1):
        super(Res_model, self).__init__()

        self.h = np.float32(h)
        self.h2 = np.float(h**2)
        self.q_dim = q_dim
        self.u_dim = u_dim
        self.f1 = nn.Linear(2*q_dim + u_dim, 64)
        self.fout = nn.Linear(64, 2*q_dim, bias=False)
        

    def predict_from_q(self, q_t, u=None):
        #with torch.no_grad():
        if u is not None:
            u_t = np.expand_dims(u, axis=0)
            u_t = torch.from_numpy(u_t).float()

        q_next = self.forward(q_t, u_t)
        return q_next, q_next.detach().numpy()[0]


    def forward(self, q_t, u_t=None):
        if u_t is not None:
            x_t = torch.cat((q_t, u_t),axis=1)
        else:
            x_t = torch.cat((q_t, torch.zeros(q_t.shape[0], self.u_dim)), axis=1)
        delta = self.fout(torch.relu(self.f1(x_t)))
        return q_t + delta
 


class VI_VV_model(nn.Module):

    def __init__(self, q_dim, u_dim=1, h=0.1):
        super(VI_VV_model, self).__init__()

        self.h = np.float32(h)
        self.h2 = np.float(h**2)
        self.q_dim = q_dim
        #self.f1 = nn.Linear(q_dim, 64)
        #self.fout = nn.Linear(64, q_dim, bias=False)
        
        self.u1 = nn.Linear(q_dim, 100)
        self.u2 = nn.Linear(100, 100)
        self.uout = nn.Linear(100, 1, bias=False)
        self.F1 = nn.Linear(u_dim, 100)
        self.Fout = nn.Linear(100, q_dim)
        #self.du = nn.Linear(q_dim, 64)
        #self.dout = nn.Linear(64, q_dim)

    def predict_from_q(self, q_t, u=None):
        #with torch.no_grad():
        if u is not None:
            u_t = np.expand_dims(u, axis=0)
            u_t = torch.from_numpy(u_t).float()

        q_next = self.forward(q_t, u_t)
        return q_next, q_next.detach().numpy()[0]

    def dUdq(self, qt):
        U = self.uout(torch.tanh(self.u2(torch.tanh(self.u1(qt)))))
        dUdq = torch.autograd.grad([a for a in U], [qt], create_graph=True, only_inputs=True)[0]
        return dUdq
    
    def forward(self, q_t, u=None):
        q = q_t[:,:self.q_dim]
        qdot = q_t[:, self.q_dim:]
        q.requires_grad_(True)
        
        #U = self.uout(torch.relu(self.u1(q)))
        #dUdq = torch.autograd.grad([ai for a in U], [q], create_graph=True, only_inputs=True)[0]
        #dUdq = self.dout(torch.relu(self.du(q)))
        dUdq = self.dUdq(q)
        q_next = q + self.h * qdot - 0.5 * self.h2 * dUdq
        if u is not None:
            q_next = q_next + self.h2 * self.Fout(torch.tanh(self.F1(u)))
        U = self.uout(torch.relu(self.u1(q_next)))
        dUdq_next = self.dUdq(q_next)#torch.autograd.grad([a for a in U], [q_next], create_graph=True, only_inputs=True)[0]
        #dUdq_next = self.dout(torch.relu(self.du(q_next)))
        dUdq_mid = dUdq + dUdq_next
        qdot_next = qdot - 0.5 * self.h * dUdq_mid
        if u is not None:
            qdot_next = qdot_next +  self.h * self.Fout(torch.tanh(self.F1(u)))
        
        
        return torch.cat([q_next, qdot_next], 1)
 
class VI_SV_model(nn.Module):

    def __init__(self, q_dim, h=0.1):
        super(VI_SV_model, self).__init__()

        self.h = np.float32(h**2)
        self.q_dim = q_dim
        #self.f1 = nn.Linear(q_dim, 64)
        #self.fout = nn.Linear(64, q_dim, bias=False)
        self.u1 = nn.Linear(q_dim, 64)
        self.uout = nn.Linear(64, 1, bias=False)

    #def predict_from_x(self, x_t_, x_tt_):
    #    x_t = np.expand_dims(x_t_, axis=0)
    #    x_t = torch.from_numpy(x_t).float()
    #    x_tt = np.expand_dims(x_tt_, axis=0)
    #    x_tt = torch.from_numpy(x_tt).float()
    #    with torch.no_grad():
    #        q_t = self.encoder(x_t)
    #        q_tt = self.encoder(x_tt)
    #        q_next = self.forward(q_t, q_tt)
    #        x_next_hat = self.decoder(q_next)
    #    return q_tt, q_next, x_next_hat.numpy()[0]

    def predict_from_q(self, q_t, q_tt, red=False):
        #with torch.no_grad():

        q_next = self.forward(q_t, q_tt)
        return q_next, q_next.detach().numpy()[0]


    def forward(self, q_t, q_tt):
        qt = q_t[:,:self.q_dim]
        qtt = q_tt[:, :self.q_dim]
        qtt.requires_grad_(True)
        U = self.uout(torch.relu(self.u1(qtt)))
        dUdq = torch.autograd.grad([a for a in U], [qtt], create_graph=True, only_inputs=True)[0]
        #q_ttt = np.float(2.0)*q_tt - q_t - self.h*(q_tt+ self.fout(torch.tanh(self.f1(q_tt))))
        q_ttt = np.float(2.0)*qtt - qt - self.h * dUdq

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
        self.zout = nn.Linear(hid_units, q_dim, bias=False)
        
    def forward(self, x_):
        x = x_.float()
        h1 = F.relu(self.z1(x))
        h2 = F.relu(self.z2(h1))
        return self.zout(h2)

class Decoder(nn.Module):

    def __init__(self, x_dim, q_dim, hid_units):
        super(Decoder, self).__init__()
        
        self.z_inv1 = nn.Linear(q_dim, hid_units)
        self.z_inv2 = nn.Linear(hid_units, hid_units)
        self.z_invout = nn.Linear(hid_units, x_dim, bias=False)

    def forward(self, q):

        h1 = F.relu(self.z_inv1(q))
        h2 = F.relu(self.z_inv2(h1))
        return self.z_invout(h2).double()
        
