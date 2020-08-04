import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def VI_SV_loss(model, x0, x, u=None):
    loss = 0
    H = x.shape[1]
    q_hat = model.encoder(x0)
    q_hatt = model.encoder(x[:,0])
    x_hat = model.decoder(q_hat)
    x_hatt = model.decoder(q_hatt)
    #loss += torch.mean(torch.pow(x_hat - x0, 2))
    loss += torch.mean(torch.pow(x_hatt - x[:,0], 2))
    for i in range(1,H):
        if u is not None:
            q_next_hat = model(q_hat.float(), q_hatt.float(), u[:,i].float()).double()
        else:
            q_next_hat = model(q_hat.float(), q_hatt.float()).double()
        
        x_next_hat = model.decoder(q_next_hat)
        loss += torch.mean(torch.pow(x_next_hat - x[:,i], 2))
        q_hat = q_hatt
        q_hatt = q_next_hat

    return loss

def VI_VV_loss(model, x0, x, u=None):
    loss = 0
    H = x.shape[1]
    q_hat = model.encoder(x0)
    x_hat = model.decoder(q_hat)
    loss += torch.mean(torch.pow(x_hat - x0, 2))
    for i in range(H):
        if u is not None:
            q_next_hat = model(q_hat.float(), u[:,i].float()).double()
        else:
            q_next_hat = model(q_hat.float()).double()
        x_next_hat = model.decoder(q_next_hat)
        loss += torch.mean(torch.pow(x_next_hat - x[:,i], 2))
        q_hat = q_next_hat

    return loss

class Res_model(nn.Module):

    def __init__(self, x_dim, q_dim, n_hid=100, u_dim=0, h=0.1, encoder=False):
        super(Res_model, self).__init__()
        if encoder:
            self.encoder = Encoder(x_dim, q_dim, 100)
            self.decoder = Decoder(x_dim, q_dim, 100)
            input_dim = 2*q_dim
        else:
            self.encoder = torch.nn.Identity()
            self.decoder = torch.nn.Identity()
            input_dim = x_dim
            
        self.h = np.float32(h)
        self.h2 = np.float(h**2)
        self.q_dim = q_dim
        self.u_dim = u_dim
        self.f1 = nn.Linear(2*q_dim, n_hid)
        self.f2 = nn.Linear(n_hid, n_hid)
        self.fout = nn.Linear(n_hid, input_dim, bias=False)
        
        self.force1 = nn.Linear(u_dim+2*q_dim, 100,bias=True)
        self.force2 = nn.Linear(n_hid, n_hid)
        self.forceout = nn.Linear(n_hid, input_dim, bias=False)

    def predict_from_q(self, q_t_, q_t, u):
        u_t = np.expand_dims(u, axis=0)
        u_t = torch.from_numpy(u_t).float()
        q_t = q_t.float()
        q_t_ = q_t_.float()
        q_next = self.forward(q_t_, q_t, u_t)
        return q_next, q_t, self.decoder(q_next).detach().numpy()[0]
    
    def forward(self, q_t_, q_t, u_t):
        qu = torch.cat([q_t, u_t], 1)
        fu = self.forceout(torch.relu(self.force2(torch.relu(self.force1(qu)))))
        delta = self.fout(torch.relu(self.f2(torch.relu(self.f1(q_t)))))
        #if u_t is not None:
            #delta += self.forceout(torch.relu(self.force1(u_t)))
            #delta = self.fout(torch.relu(self.f1(qu)))
            #delta = self.fout(torch.relu(self.f1(qu)))
            #x_t = torch.cat((q_t, u_t),axis=1)
        #else:
            #x_t = q_t
        #delta = self.fout(torch.relu(self.f1(x_t)))
        return q_t + delta + fu
 
class ResPos_model(nn.Module):

    def __init__(self, x_dim, q_dim, n_hid=100, u_dim=0, h=0.1, encoder=False):
        super(ResPos_model, self).__init__()
        if encoder:
            self.encoder = EncoderSV(x_dim, q_dim, 100)
            self.decoder = DecoderSV(x_dim, q_dim, 100)
            input_dim = 2*q_dim
        else:
            self.encoder = torch.nn.Identity()
            self.decoder = torch.nn.Identity()
            input_dim = x_dim
            
        self.h = np.float32(h)
        self.h2 = np.float(h**2)
        self.q_dim = q_dim
        self.u_dim = u_dim
        self.f1 = nn.Linear(2*q_dim, n_hid)
        self.f2 = nn.Linear(n_hid, n_hid)
        self.fout = nn.Linear(n_hid, q_dim, bias=False)
        
        self.force1 = nn.Linear(q_dim+u_dim, n_hid,bias=True)
        self.force2 = nn.Linear(n_hid, n_hid)
        self.forceout = nn.Linear(n_hid, q_dim, bias=False)

    def predict_from_q(self, q_t_, q_t, u):
        u_t = np.expand_dims(u, axis=0)
        u_t = torch.from_numpy(u_t).float()
        q_t = q_t.float()
        q_t_ = q_t_.float()
        q_next = self.forward(q_t_, q_t, u_t)
        return q_next, q_t, self.decoder(q_next).detach().numpy()[0]
    
    def forward(self, q_t_, q_t, u_t):
        qq = torch.cat([q_t_, q_t], 1)
        qu = torch.cat([q_t, u_t], 1)
        fu = self.forceout(torch.relu(self.force2(torch.relu(self.force1(qu)))))
        dq = self.fout(torch.relu(self.f2(torch.relu(self.f1(qq)))))
        #if u_t is not None:
            #delta += self.forceout(torch.relu(self.force1(u_t)))
            #delta = self.fout(torch.relu(self.f1(qu)))
            #delta = self.fout(torch.relu(self.f1(qu)))
            #x_t = torch.cat((q_t, u_t),axis=1)
        #else:
            #x_t = q_t
        #delta = self.fout(torch.relu(self.f1(x_t)))
        return q_t + fu + dq
 




class VI_VV_model(nn.Module):

    def __init__(self, x_dim, q_dim, n_hid=100, u_dim=0, h=0.1,encoder=False):
        super(VI_VV_model, self).__init__()
        if encoder:
            self.encoder = Encoder(x_dim, q_dim, 100)
            self.decoder = Decoder(x_dim, q_dim, 100)
            #self.encoder = EncoderImage(q_dim)
            #self.decoder = DecoderImage(q_dim)
        else:
            self.encoder = torch.nn.Identity()
            self.decoder = torch.nn.Identity()

        self.h = np.float32(h)
        self.h2 = np.float(h**2)
        self.q_dim = q_dim
        self.u1 = nn.Linear(q_dim, n_hid)
        self.u2 = nn.Linear(n_hid, n_hid)
        self.alpha = np.float(1.0)
        self.friction_ = True
        self.uout = nn.Linear(n_hid, q_dim, bias=False)
        if u_dim != 0:
            self.force1 = nn.Linear(u_dim+q_dim, n_hid, bias=True)
            self.force2 = nn.Linear(n_hid, n_hid)
            self.forceout = nn.Linear(n_hid, q_dim, bias=False)
        self.fr1 = nn.Linear(2*q_dim, n_hid, bias=True)
        self.frout = nn.Linear(n_hid, q_dim, bias=False)
    
    def predict_from_q(self, q_t_, q_t, u):
        u_t = np.expand_dims(u, axis=0)
        u_t = torch.from_numpy(u_t).float()
        q_t =q_t.float()
        q_t_ = q_t_.float()
        q_next = self.forward(q_t_, q_t, u_t)
        return q_next, q_t, self.decoder(q_next).detach().numpy()[0]

    def dUdq(self, qt):
        #return self.uout(torch.tanh(self.u2(torch.tanh(self.u1(qt)))))
        return self.uout(torch.relu(self.u2(torch.relu(self.u1(qt)))))

    def forward(self, q_t_, q_t, u):
        q = q_t[:,:self.q_dim]
        q_ = q_t_[:,:self.q_dim]
        qdot = q_t[:, self.q_dim:]
        
        dUdq = self.dUdq(q)
        friction = self.frout(torch.relu(self.fr1(q_t)))
        q_next = q + self.h * qdot - np.float32(0.5) * self.h2 * (dUdq)
        
        if self.friction_:
            q_next += self.alpha * np.float32(0.5) * self.h2 * friction
        
        qu = torch.cat([q, u],1)
        fu = self.forceout(torch.relu(self.force2(torch.relu(self.force1(qu)))))
        
        q_next = q_next + np.float32(0.5) * self.h2 * fu        
        dUdq_next = self.dUdq(q_next)#
        
        dUdq_mid = dUdq + dUdq_next
        qdot_next = qdot - np.float(0.5) * self.h * dUdq_mid
        if u is not None:
            qdot_next = qdot_next +  self.h * fu
        if self.friction_:
            qdot_next = qdot_next + self.alpha * self.h * friction
        
        
        return torch.cat([q_next, qdot_next], 1)
 

class VI_SV_model(nn.Module):
    def __init__(self, x_dim, q_dim, n_hid=100, u_dim=0, h=0.1, encoder=True):
        super(VI_SV_model, self).__init__()
        if encoder:
            self.encoder = EncoderSV(x_dim, q_dim, 100)
            self.decoder = DecoderSV(x_dim, q_dim, 100)
        else:
            self.encoder = torch.nn.Identity()
            self.decoder = torch.nn.Identity()

        self.h = np.float32(h)
        self.h2 = np.float(h**2)
        self.q_dim = q_dim
        self.u1 = nn.Linear(q_dim, n_hid)
        self.u2 = nn.Linear(n_hid, n_hid)
        self.uout = nn.Linear(n_hid, q_dim, bias=False)
        self.alpha = np.float(1.0)
        self.friction = True

        self.force1 = nn.Linear(u_dim, n_hid, bias=False)
        self.force2 = nn.Linear(n_hid, n_hid)
        self.forceout = nn.Linear(n_hid, q_dim, bias=False)
        
        self.fr1 = nn.Linear(2*q_dim, 100, bias=True)
        self.frout = nn.Linear(100, q_dim, bias=False)

    def predict_from_q(self, q_t, q_tt, u=None):
        u_t = np.expand_dims(u, axis=0)
        u_t = torch.from_numpy(u_t).float()
        
        q_t =q_t.float()
        q_tt = q_tt.float()
        q_next = self.forward(q_t, q_tt, u_t)
        
        return q_next, q_tt, self.decoder(q_next).detach().numpy()[0]
    

    def forward(self, q_t, q_tt, u=None):
        qt = q_t
        qtt = q_tt
        dV = self.uout(torch.relu(self.u2(torch.relu(self.u1(qtt)))))
        qttt = np.float(2.0)*qtt - qt - self.h2*dV
        
        qu = u#torch.cat([qtt, u],1)
        fu = self.forceout(torch.relu(self.force2(torch.relu(self.force1(qu)))))
        qttt += self.h2*fu
       
        qq = torch.cat([qt,qtt],1)
        qttt += self.h2*self.alpha*self.frout(torch.relu(self.fr1(qq)))

        return qttt
    
class Encoder(nn.Module):
    
    def __init__(self, x_dim, q_dim, hid_units):
        super(Encoder, self).__init__()
        self.q_dim = q_dim
        self.x_dim = x_dim
            
        self.z1 = nn.Linear(x_dim-self.q_dim, hid_units)
        self.z2 = nn.Linear(hid_units, hid_units)
        self.zout = nn.Linear(hid_units, q_dim, bias=False)
    
    def forward(self, x_):
        x = x_.float()
        y = x[:,:self.x_dim - self.q_dim]
        qdot = x[:,self.x_dim- self.q_dim:]
        
        h1 = torch.relu(self.z1(y))
        h2 = torch.relu(self.z2(h1))
        return torch.cat((self.zout(h2), qdot),1).double()

class Decoder(nn.Module):

    def __init__(self, x_dim, q_dim, hid_units):
        super(Decoder, self).__init__()
        self.q_dim = q_dim
        
        self.z_inv1 = nn.Linear(self.q_dim, hid_units)
        self.z_inv2 = nn.Linear(hid_units, hid_units)
        self.z_invout = nn.Linear(hid_units, x_dim-self.q_dim, bias=False)

    def forward(self, q_):
        q_ = q_.float()
        q = q_[:,:self.q_dim]
        qdot = q_[:,self.q_dim:]
        
        h1 = torch.relu(self.z_inv1(q))
        h2 = torch.relu(self.z_inv2(h1))
        return torch.cat((self.z_invout(h2), qdot),1).double()

class EncoderSV(nn.Module):
    
    def __init__(self, x_dim, q_dim, hid_units):
        super(EncoderSV, self).__init__()
        self.q_dim = q_dim
        self.x_dim = x_dim
            
        self.z1 = nn.Linear(x_dim, hid_units)
        self.z2 = nn.Linear(hid_units, hid_units)
        self.zout = nn.Linear(hid_units, q_dim, bias=False)
    
    def forward(self, x_):
        x = x_.float()
        
        h1 = torch.relu(self.z1(x))
        h2 = torch.relu(self.z2(h1))
        return self.zout(h2).double()

class DecoderSV(nn.Module):

    def __init__(self, x_dim, q_dim, hid_units):
        super(DecoderSV, self).__init__()
        self.q_dim = q_dim
        
        self.z_inv1 = nn.Linear(self.q_dim, hid_units)
        self.z_inv2 = nn.Linear(hid_units, hid_units)
        self.z_invout = nn.Linear(hid_units, x_dim, bias=False)

    def forward(self, q_):
        q_ = q_.float()
        
        h1 = torch.relu(self.z_inv1(q_))
        h2 = torch.relu(self.z_inv2(h1))
        return self.z_invout(h2).double()


class EncoderImage(nn.Module):
    def __init__(self, q_dim):
        super(EncoderImage, self).__init__()
        self.q_dim = q_dim
        self.encode_q = nn.Sequential(
            nn.Linear(28* 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, self.q_dim))

    def forward(self, x):
        x = x.float()
        y = x[:,:-self.q_dim]
        q_dot = x[:,-self.q_dim:]
        q = self.encode_q(y)
        return torch.cat([q,q_dot],1)

class DecoderImage(nn.Module):
    def __init__(self, q_dim):
        super(DecoderImage, self).__init__()
        self.q_dim = q_dim
        self.decode_q = nn.Sequential(
            nn.Linear(self.q_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh())
    
    def forward(self, q):
        q = q.float()
        q_ = q[:,:self.q_dim]
        q_dot = q[:, self.q_dim:]
        image = self.decode_q(q_)
        return torch.cat([image, q_dot],1)
    
