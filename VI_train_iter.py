import torch
from torch.utils import data
import torch.optim as optim
from data_loader import TrajDataset, gym_gen
from VI_model import VI_VV_loss, VI_VV_model, VI_SV_loss, VI_SV_model, Res_model#, Encoder, Decoder
from torch.nn.utils import clip_grad_norm
import gym
import gym_custom
import numpy as np
#import gym_softreacher
#from multi_env import MultiEnv
from VISV_cem import CEM
import pickle
from quad_dynamics import pid
#import roboschool

load = 0
train_control = True
save = 1
#fname = 'models/res_circ_quan_iter.pt'
fname = 'models/vv_10_policy_quan.pt'
#max_epochs= 10000
params ={   'batch_size': 2048,
            'shuffle': True,
            'num_workers' : 1}

#env = gym.make('QPendulum-v0')
#env = gym.make('Pendulum-v0')
#env = gym.make('PendulumPos-v0')
#env = gym.make('Manipulator-v0')
#env = gym.make('Quadrotor2D-v0')
#env = gym.make('CartpoleMod-v0')
#env = gym.make('AcrobotPos-v0')
#env = gym.make('Acrobot-v1')
#env = gym.make('SoftReacher-v0')
#env = gym.make('SpringMass-v0')
#env = gym.make('ImPendulum-v0')
#env = gym.make('SpringMassPos-v0')
#env = gym.make('CoupledMassSpring-v0')
#env = gym.make('RoboschoolReacher-v1')
q_dim = 2
h=0.004
x_dim = 5#len(env.observation_space.low)
u_dim = 1#len(env.action_space.low)
hid_units = 100
model_type = VI_VV_model
encoder = True

if load:
    model = torch.load(fname)
    #checkpoint = torch.load(fname)
    #model.load_state_dict(checkpoint)

else:
    model = model_type(x_dim, q_dim, u_dim=u_dim, h=h, encoder=encoder)
    #model = VI_SV_model(x_dim, q_dim, u_dim=u_dim, h=h, encoder=True)
    #model = Res_model(x_dim, q_dim, u_dim=u_dim, h=h, encoder=False)


def train_model(traj_dict, model, max_epochs, params, iteration):
    optimizer = optim.Adam(model.parameters(), lr=5e-4)#,weight_decay=1e-5)
    dataset = TrajDataset(traj_dict, 10)
    print('data set size: ',len(dataset))
    training_gen = data.DataLoader(dataset, **params)
    for epochs in range(max_epochs):
        epoch_loss = 0
        n = 0
        for init_states, controls, states in training_gen:
            N = states.shape[0]
            optimizer.zero_grad()
            loss = VI_SV_loss(model, init_states, states, controls)
    
            loss.backward()
    
            optimizer.step()
    
            epoch_loss += loss.item()/N
            n += 1
        print("iteration: "+ str(iteration), "epoch: " + str(epochs), "mean loss: ", str(epoch_loss/n))
        if (epochs%20==0 or epochs==max_epochs-1) and save:
            torch.save(model, fname)

    return model

def append_trajs(traj_dict, traj_dict_):
    traj_dict['states'] = np.concatenate((traj_dict['states'], traj_dict_['states']),axis=0)
    traj_dict['controls'] = np.concatenate((traj_dict['controls'], traj_dict_['controls']),axis=0)
    traj_dict['news'] = np.concatenate((traj_dict['news'], traj_dict_['news']),axis=0)
    return traj_dict
traj_dict = None
#model_pi = torch.load('models/vv_50_cart.pt')
#pi = CEM(model_pi, 500, 20, env)
max_epochs = 20000
#traj_dict = gym_gen(env, (50-1)*5, pi='uhlenbeck', seed=15)
with open('quansar_policy_10_short_train.pkl', 'rb') as f:
    traj_dict = pickle.load(f)
model = train_model(traj_dict, model, max_epochs, params, 0)
#model = torch.load('models/res_5_cart.pt')
max_epochs = 1000
for i in range(0):
    pi = CEM(model, 500, 20, env)
    #model_ = model_type(x_dim, q_dim, u_dim=u_dim, h=h, encoder=encoder)
    model_ = model
    traj_dict_ = gym_gen(env, (50-1)*1, pi=pi, stochastic=True, seed=i)
    if traj_dict is not None:
        traj_dict = append_trajs(traj_dict, traj_dict_)
    else:
        traj_dict = traj_dict_
    model = train_model(traj_dict, model_, max_epochs, params, i+1)
    


