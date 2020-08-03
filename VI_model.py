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
    #q_hat = model.encoder(x[:,0])
    x_hat = model.decoder(q_hat)
    loss += torch.mean(torch.pow(x_hat - x0, 2))
    #loss += torch.mean(torch.pow(x_hat -x[:,0],2))
    #q_hat = q_next_hat
    #lam = np.float32(0.95)
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

    def __init__(self, x_dim, q_dim, u_dim=0, h=0.1, encoder=False):
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
        self.f1 = nn.Linear(input_dim+u_dim, 100)
        self.f2 = nn.Linear(100, 100)
        self.fout = nn.Linear(100, input_dim, bias=False)
        #self.force1 = nn.Linear(u_dim, 100,bias=False)
        #self.forceout = nn.Linear(100, input_dim, bias=False)

    def predict_from_q(self, q_t_, q_t, u=None):
        #with torch.no_grad():
        if u is not None:
            u_t = np.expand_dims(u, axis=0)
            u_t = torch.from_numpy(u_t).float()
        else:
            u_t = u
        q_t = q_t.float()
        q_t_ = q_t_.float()
        q_next = self.forward(q_t_, q_t, u_t)
        return q_next, q_t, self.decoder(q_next).detach().numpy()[0]
    
    def reward(self, y__, y_, u_):
        cost = 0
        x = self.decoder(y_).double()#x_.double()
        u = u_.double()
        #with torch.no_grad():
        #    e = x[:, :3]-torch.from_numpy(target)
        #    cost += torch.norm(e, p=2, dim=1)
        #    cost += torch.norm(u, p=2, dim=1)
        with torch.no_grad():
            theta = torch.atan2(x[:,1],x[:,0])
            cost += torch.pow(theta,2)
            cost += 0.1*torch.pow(x[:,2], 2)
            #cost += 0.001*torch.pow(u[:,0], 2)
            #cost += torch.pow(x[:,0], 2)
            #cost += 0.01*torch.pow(x[:,1], 2)
            cost += 0.001*torch.pow(u[:,0], 2)
        

        return -cost.data.numpy()
    
    def forward(self, q_t_, q_t, u_t=None):
        qu = torch.cat([q_t, u_t], 1)
        delta = self.fout(torch.relu(self.f2(torch.relu(self.f1(qu)))))
        #if u_t is not None:
            #delta += self.forceout(torch.relu(self.force1(u_t)))
            #delta = self.fout(torch.relu(self.f1(qu)))
            #delta = self.fout(torch.relu(self.f1(qu)))
            #x_t = torch.cat((q_t, u_t),axis=1)
        #else:
            #x_t = q_t
        #delta = self.fout(torch.relu(self.f1(x_t)))
        return q_t + delta
 


class VI_VV_model(nn.Module):

    def __init__(self, x_dim, q_dim, u_dim=0, h=0.1,encoder=False):
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
        #self.f1 = nn.Linear(q_dim, 64)
        #self.fout = nn.Linear(64, q_dim, bias=False)
        self.u1 = nn.Linear(q_dim, 100)
        self.u2 = nn.Linear(100, 100)
        self.alpha = np.float(1.0)
        self.friction_ = True
        #self.uout = nn.Linear(100, 1, bias=False)
        self.uout = nn.Linear(100, q_dim, bias=False)
        if u_dim != 0:
            self.force1 = nn.Linear(u_dim+q_dim, 100, bias=True)
            self.force2 = nn.Linear(100, 100)
            self.forceout = nn.Linear(100, q_dim, bias=False)
        self.fr1 = nn.Linear(2*q_dim, 100, bias=True)
        self.frout = nn.Linear(100, q_dim, bias=False)
        #self.du = nn.Linear(q_dim, 64)
        #self.dout = nn.Linear(64, q_dim)
    
    def predict_from_q(self, q_t_, q_t, u=None):
        #with torch.no_grad():
        if u is not None:
            u_t = np.expand_dims(u, axis=0)
            u_t = torch.from_numpy(u_t).float()
        else:
            u_t = u 
        q_t =q_t.float()
        q_t_ = q_t_.float()
        q_next = self.forward(q_t_, q_t, u_t)
        return q_next, q_t, self.decoder(q_next).detach().numpy()[0]

    def dUdq(self, qt):
        #U = self.uout(torch.tanh(self.u2(torch.tanh(self.u1(qt)))))
        #U = self.uout(torch.relu(self.u1(qt)))

        #dUdq = torch.autograd.grad([a for a in U], [qt], create_graph=True, only_inputs=True)[0]
        #return dUdq
        #return self.uout(torch.tanh(self.u2(torch.tanh(self.u1(qt)))))
        return self.uout(torch.relu(self.u2(torch.relu(self.u1(qt)))))

        #return self.uout(torch.tanh(self.u1(qt)))
    def forward(self, q_t_, q_t, u=None):
        q = q_t[:,:self.q_dim]
        q_ = q_t_[:,:self.q_dim]
        qdot = q_t[:, self.q_dim:]
        #q.requires_grad_(True)
        
        dUdq = self.dUdq(q)
        #dUdq = self.uout(torch.relu(self.u1(q)))
        #qu = torch.cat([q_,q],1)
        friction = self.frout(torch.relu(self.fr1(q_t)))
        q_next = q + self.h * qdot - np.float32(0.5) * self.h2 * (dUdq)
        if self.friction_:
            q_next += self.alpha * np.float32(0.5) * self.h2 * friction
        if u is not None:
            qu = torch.cat([q, u],1)
            fu = self.forceout(torch.relu(self.force2(torch.relu(self.force1(qu)))))
            q_next = q_next + np.float32(0.5) * self.h2 * fu        
        #U = self.uout(torch.relu(self.u1(q_next)))
        dUdq_next = self.dUdq(q_next)#torch.autograd.grad([a for a in U], [q_next], create_graph=True, only_inputs=True)[0]
        #dUdq_next = self.uout(torch.relu(self.u1(q_next)))
        
        dUdq_mid = dUdq + dUdq_next
        qdot_next = qdot - np.float(0.5) * self.h * dUdq_mid
        if u is not None:
            qdot_next = qdot_next +  self.h * fu
        if self.friction_:
            qdot_next = qdot_next + self.alpha * self.h * friction
        
        
        return torch.cat([q_next, qdot_next], 1)
 

    def reward(self, y__, y_, u_):
        cost = 0
        x = self.decoder(y_).double()#x_.double()
        u = u_.double()
        #with torch.no_grad():
        #    e = x[:, :3]-torch.from_numpy(target)
        #    cost += torch.norm(e, p=2, dim=1)
        #    cost += torch.norm(u, p=2, dim=1)
        with torch.no_grad():
            costh = torch.cos(x[:,0])
            sinth = torch.sin(x[:,0])
            theta = torch.atan2(sinth,costh)
            cost += torch.pow(theta+np.pi, 2)
            cost += 0.1*torch.pow(x[:,1], 2)
            #cost += 0.1*torch.pow(x[:,2], 2)
            #cost += torch.pow(x[:,0], 2)
            #cost += 0.1*torch.pow(x[:,2], 2)
            #cost += 0.1*torch.pow(x[:,3], 2)
            #cost += 0.1*torch.pow(x[:,4], 2)
            cost += 0.001*torch.pow(u[:,0], 2)
        
        return -cost.data.numpy()
    

class VI_SV_model(nn.Module):
    def __init__(self, x_dim, q_dim, u_dim=0, h=0.1,encoder=False):
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
        self.u1 = nn.Linear(q_dim, 100)
        self.u2 = nn.Linear(100, 100)
        #self.uout = nn.Linear(100, 1, bias=False)
        self.uout = nn.Linear(100, q_dim, bias=False)
        if u_dim != 0:
            self.force1 = nn.Linear(u_dim, 100, bias=False)
            self.forceout = nn.Linear(100, q_dim, bias=False)
        
        self.fr1 = nn.Linear(2*q_dim, 100, bias=True)
        self.frout = nn.Linear(100, q_dim, bias=False)

    def predict_from_q(self, q_t, q_tt, u=None):
        #with torch.no_grad():
        if u is not None:
            u_t = np.expand_dims(u, axis=0)
            u_t = torch.from_numpy(u_t).float()
        else:
            u_t = u 
        q_t =q_t.float()
        q_tt = q_tt.float()
        q_next = self.forward(q_t, q_tt, u_t)
        
        return q_next, q_tt, self.decoder(q_next).detach().numpy()[0]
    

    def forward(self, q_t, q_tt, u=None):
        qt = q_t
        qtt = q_tt
        #qt = q_t[:,:self.q_dim]
        #qtt = q_tt[:, :self.q_dim]
        qtt.requires_grad_(True)
        #U = self.uout(torch.relu(self.u1(qtt)))
        #dUdq = torch.autograd.grad([a for a in U], [qtt], create_graph=True, only_inputs=True)[0]
        dV = self.uout(torch.relu(self.u2(torch.relu(self.u1(qtt)))))
        qttt = np.float(2.0)*qtt - qt - self.h2*dV
        if u is not None:
            qttt += self.h2*self.forceout(torch.relu(self.force1(u)))
        
        qu = torch.cat([qt,qtt],1)
        qttt += self.h2*self.frout(torch.relu(self.fr1(qu)))

        return qttt
        #q_t = q[:,:self.q_dim]
        #q_dot_t = q[:,-self.q_dim:]
        #q_dot_tt = torch.add(q_dot_t, torch.add(q_t, self.h*self.fout(F.relu(self.f1(q_t)))))
        #q_tt = torch.add(q_t, self.h*q_dot_tt)

        #return torch.cat((q_tt, q_dot_tt), axis=1)
    
    def reward(self, yt_, ytt_, u_):
        cost = 0
        xt = self.decoder(yt_).double()#x_.double()
        xtt = self.decoder(ytt_).double()
        u = u_.double()
        #with torch.no_grad():
        #    e = x[:, :3]-torch.from_numpy(target)
        #    cost += torch.norm(e, p=2, dim=1)
        #    cost += torch.norm(u, p=2, dim=1)
        with torch.no_grad():
            theta_t = torch.atan2(xt[:,1],xt[:,0])
            theta_tt = torch.atan2(xtt[:,1],xtt[:,0])
            cost += torch.pow(theta_tt,2)
            cost += 0.001*torch.pow(torch.norm((xtt - xt)/0.1), 2)
            #cost += 0.1*torch.pow(x[:,2], 2)
            
            #theta_t = torch.asin(xt[:,0])
            #theta_tt = torch.asin(xtt[:,0])
            #cost += torch.pow(theta_tt,2)
            #cost += 0.01*torch.pow((theta_tt-theta_t)/0.1,2)
            cost += 0.001*torch.pow(u[:,0], 2)
        return -cost.data.numpy()
    

class Encoder(nn.Module):
    
    def __init__(self, x_dim, q_dim, hid_units):
        super(Encoder, self).__init__()
        self.q_dim = q_dim
        self.x_dim = x_dim
            
        #self.zdot1 = nn.Linear(self.q_dim, hid_units)
        #self.zdot2 = nn.Linear(hid_units, hid_units)
        #self.zdotout = nn.Linear(hid_units, q_dim, bias=False)


        self.z1 = nn.Linear(x_dim-self.q_dim, hid_units)
        #self.z1 = nn.Linear(x_dim, hid_units)
        self.z2 = nn.Linear(hid_units, hid_units)
        self.zout = nn.Linear(hid_units, q_dim, bias=False)
    def forward(self, x_):
        x = x_.float()
        y = x[:,:self.x_dim - self.q_dim]
        qdot = x[:,self.x_dim- self.q_dim:]
        #hdot1 = torch.relu(self.zdot1(qdot))
        #hdot2 = torch.relu(self.zdot2(hdot1))
        
        h1 = torch.relu(self.z1(y))
        h2 = torch.relu(self.z2(h1))
        return torch.cat((self.zout(h2), qdot),1).double()
        #return self.zout(h2).double()

class Decoder(nn.Module):

    def __init__(self, x_dim, q_dim, hid_units):
        super(Decoder, self).__init__()
        self.q_dim = q_dim
        
        #self.zdot_inv1 = nn.Linear(self.q_dim, hid_units)
        #self.zdot_inv2 = nn.Linear(hid_units, hid_units)
        #self.zdot_invout = nn.Linear(hid_units, self.q_dim, bias=False)

        self.z_inv1 = nn.Linear(self.q_dim, hid_units)
        self.z_inv2 = nn.Linear(hid_units, hid_units)
        self.z_invout = nn.Linear(hid_units, x_dim-self.q_dim, bias=False)

    def forward(self, q_):
        q_ = q_.float()
        q = q_[:,:self.q_dim]
        qdot = q_[:,self.q_dim:]
        
        #hdot1 = torch.relu(self.zdot_inv1(q_dot))
        #hdot2 = torch.relu(self.zdot_inv2(hdot1))

        h1 = torch.relu(self.z_inv1(q))
        h2 = torch.relu(self.z_inv2(h1))
        return torch.cat((self.z_invout(h2), qdot),1).double()
        #return self.z_invout(h2).double()

class EncoderSV(nn.Module):
    
    def __init__(self, x_dim, q_dim, hid_units):
        super(EncoderSV, self).__init__()
        self.q_dim = q_dim
        self.x_dim = x_dim
            
        #self.zdot1 = nn.Linear(self.q_dim, hid_units)
        #self.zdot2 = nn.Linear(hid_units, hid_units)
        #self.zdotout = nn.Linear(hid_units, q_dim, bias=False)


        self.z1 = nn.Linear(x_dim, hid_units)
        #self.z1 = nn.Linear(x_dim, hid_units)
        self.z2 = nn.Linear(hid_units, hid_units)
        self.zout = nn.Linear(hid_units, q_dim, bias=False)
    def forward(self, x_):
        x = x_.float()
        #y = x[:,:self.x_dim - self.q_dim]
        #qdot = x[:,self.x_dim- self.q_dim:]
        #hdot1 = torch.relu(self.zdot1(qdot))
        #hdot2 = torch.relu(self.zdot2(hdot1))
        
        h1 = torch.relu(self.z1(x))
        h2 = torch.relu(self.z2(h1))
        #return torch.cat((self.zout(h2), qdot),1).double()
        return self.zout(h2).double()

class DecoderSV(nn.Module):

    def __init__(self, x_dim, q_dim, hid_units):
        super(DecoderSV, self).__init__()
        self.q_dim = q_dim
        
        #self.zdot_inv1 = nn.Linear(self.q_dim, hid_units)
        #self.zdot_inv2 = nn.Linear(hid_units, hid_units)
        #self.zdot_invout = nn.Linear(hid_units, self.q_dim, bias=False)

        self.z_inv1 = nn.Linear(self.q_dim, hid_units)
        self.z_inv2 = nn.Linear(hid_units, hid_units)
        self.z_invout = nn.Linear(hid_units, x_dim, bias=False)

    def forward(self, q_):
        q_ = q_.float()
        #q = q_[:,:self.q_dim]
        #qdot = q_[:,self.q_dim:]
        
        #hdot1 = torch.relu(self.zdot_inv1(q_dot))
        #hdot2 = torch.relu(self.zdot_inv2(hdot1))

        h1 = torch.relu(self.z_inv1(q_))
        h2 = torch.relu(self.z_inv2(h1))
        #return torch.cat((self.z_invout(h2), qdot),1).double()
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
    
