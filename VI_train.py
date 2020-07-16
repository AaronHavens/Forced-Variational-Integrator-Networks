import torch
from torch.utils import data
import torch.optim as optim
from data_loader import TrajDataset, gym_gen
from VI_model import VI_VV_loss, VI_VV_model, Res_model#, Encoder, Decoder
from torch.nn.utils import clip_grad_norm
import gym
import gym_custom
#import dm_control2gym
import gym_softreacher
from multi_env import MultiEnv
from VI_cem import CEM

load = 0
train_control = True
save = 1
fname = 'models/res_reacher_forced_encoder_d1.pt'
max_epochs= 10000
params ={   'batch_size': 1024,
            'shuffle': True,
            'num_workers' : 1}

#env = gym.make('QPendulum-v0')
#env = gym.make('Pendulum-v0')
#env = gym.make('Acrobot-v1')
#env = gym.make('SoftReacher-v0')
env = gym.make('Springmass-v0')
q_dim = 1
h=0.1
x_dim = len(env.observation_space.low)
u_dim = len(env.action_space.low)
hid_units = 100


if load:
    model = torch.load(fname)
    #checkpoint = torch.load(fname)
    #model.load_state_dict(checkpoint)

else:
    model = VI_VV_model(x_dim, q_dim, u_dim=u_dim, h=h, encoder=True)
    #model = Res_model(x_dim, q_dim, u_dim=u_dim, h=h, encoder=False)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
#pi = CEM(model, 500, 10, env)
traj_dict = gym_gen(env, 10*50, pi='random')
dataset   = TrajDataset(traj_dict, 10)
print('data set size: ',len(dataset))

training_gen = data.DataLoader(dataset, **params)
#clip_grad_norm(model.parameters(),1000)
for epochs in range(max_epochs):
    epoch_loss = 0
    n = 0
    for init_states, controls, states in training_gen:
        N = states.shape[0]
        optimizer.zero_grad()
        loss = VI_VV_loss(model, init_states, states, controls)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()/N
        n += 1
    print("epoch: " + str(epochs), "mean loss: ", str(epoch_loss/n))
    if (epochs%20==0 or epochs==max_epochs-1) and save:
        torch.save(model, fname)
        #torch.save({
        #            'predict' : predict.state_dict(),
        #            'encode'  : encode.state_dict(),
        #            'decode'  : decode.state_dict()
        #            }, fname)

