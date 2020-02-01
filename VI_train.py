import torch
from torch.utils import data
import torch.optim as optim
from data_loader import TrajDataset, gym_gen
from VI_model import VI_loss, VI_model, Encoder, Decoder
from torch.nn.utils import clip_grad_norm
import gym
import dm_control2gym
import gym_softreacher

load = 0
save = not load
fname = 'models/vi_test.pt'
max_epochs = 500
params ={   'batch_size': 256,
            'shuffle': True,
            'num_workers' : 1}

env = gym.make('Pendulum-v0')

q_dim = 2
h=0.1
x_dim = len(env.observation_space.low)
u_dim = len(env.action_space.low)
hid_units = 64

encode = Encoder(x_dim, q_dim, hid_units)
decode = Decoder(x_dim, q_dim, hid_units)
model = VI_model(encode,decode, q_dim, h=0.1)
if load:
    checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint)
    #encode.load_state_dict(checkpoint['encode'])
    #decode.load_state_dict(checkpoint['decode'])

optimizer = optim.Adam(model.parameters(), lr=5e-4)
#optimizer_predict = optim.Adam(predict.parameters(), lr=5e-4)
#optimizer_encode  = optim.Adam(encode.parameters(), lr=5e-4)
#optimizer_decode  = optim.Adam(decode.parameters(), lr=5e-4)

traj_dict = gym_gen(env, 3000)
dataset   = TrajDataset(traj_dict, 3)
print('data set size: ',len(dataset))

training_gen = data.DataLoader(dataset, **params)
#clip_grad_norm(model.parameters(),1000)
for epochs in range(max_epochs):
    epoch_loss = 0
    n = 0
    for init_states, controls, states in training_gen:
        N = states.shape[0]
        optimizer.zero_grad()
        #optimizer_predict.zero_grad()
        #optimizer_encode.zero_grad()
        #optimizer_decode.zero_grad()

        loss = VI_loss(model, init_states, states)
        #loss = VI_loss(encode, decode, predict, init_states, states)
        loss.backward()

        optimizer.step()
        #optimizer_predict.step()
        #optimizer_encode.step()
        #optimizer_decode.step()

        epoch_loss += loss.item()/N
        n += 1
    print("epoch: " + str(epochs), "mean loss: ", str(epoch_loss/n))
    if (epochs%20==0 or epochs==max_epochs-1) and save:
        torch.save(model.state_dict(), fname)
        #torch.save({
        #            'predict' : predict.state_dict(),
        #            'encode'  : encode.state_dict(),
        #            'decode'  : decode.state_dict()
        #            }, fname)

