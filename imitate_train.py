import torch
from torch.utils import data
import torch.optim as optim
from data_loader import ImitateDataset, gym_gen
from model import imitation_loss, ImitatePolicy
from torch.nn.utils import clip_grad_norm
import gym
import gym_softreacher
from multi_env import MultiEnv
from cem import CEM

load = 0
save = not load
model_fname = 'models/theta_test_u.pt'
fname = 'models/softarm_policy.pt'

max_epochs = 1000
params ={   'batch_size': 256,
            'shuffle': True,
            'num_workers' : 1}

#env = gym.make('Pendulum-v0')
env = MultiEnv('SoftReacher-v0')
x_dim = len(env.observation_space.low)
u_dim = len(env.action_space.low)

model = torch.load(model_fname)
pi = CEM(model, 1000, 15, env.action_space.low, env.action_space.high)

pi_model = ImitatePolicy(x_dim, u_dim, 100)

optimizer = optim.Adam(pi_model.parameters(), lr=5e-4)

traj_dict = gym_gen(env, 2000, pi=pi)
dataset   = ImitateDataset(traj_dict, 10)
print(len(dataset))
training_gen = data.DataLoader(dataset, **params)


for epochs in range(max_epochs):
    epoch_loss = 0
    n = 0
    for init_states, controls in training_gen:
        N = init_states.shape[0]
        optimizer.zero_grad()
        loss = imitation_loss(pi_model, init_states, controls)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()/N
        n += 1
    print("epoch: " + str(epochs), "mean loss: ", str(epoch_loss/n))
    if (epochs%20==0 or epochs==max_epochs-1) and save:
        torch.save(pi_model, fname)
