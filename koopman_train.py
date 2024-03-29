import torch
from torch.utils import data
import torch.optim as optim
from data_loader import TrajDataset, gym_gen
from koopman_model import multistep_predict_loss, KoopmanPredict,VIKoopmanPredict, VIEncKoopmanPredict
from torch.nn.utils import clip_grad_norm
import gym
import dm_control2gym
import gym_softreacher
from multi_env import MultiEnv
import gym_custom


load = 0
save = not load
fname = 'models/pendulum_enc.pt'
max_epochs = 10000
params ={   'batch_size': 1024,
            'shuffle': True,
            'num_workers' : 1}

env = gym.make('QPendulum-v0')
#env = MultiEnv('SoftReacher-v0')
x_dim = len(env.observation_space.low)
u_dim = len(env.action_space.low)
z_dim = 500
q_dim = 1
if load:
    model = torch.load(fname)
else:
    model = VIKoopmanPredict(x_dim, u_dim, z_dim, q_dim, 100)

optimizer = optim.Adam(model.parameters(), lr=5e-4)#, weight_decay=0.01)


traj_dict = gym_gen(env, 10*50, pi=None)
dataset   = TrajDataset(traj_dict, 10)
print(len(dataset))
training_gen = data.DataLoader(dataset, **params)
#clip_grad_norm(model.parameters(),1000)
for epochs in range(max_epochs):
    epoch_loss = 0
    n = 0
    for init_states, controls, states in training_gen:
        N = states.shape[0]
        optimizer.zero_grad()
        loss = multistep_predict_loss(model, init_states, controls, states)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()/N
        n += 1
    print("epoch: " + str(epochs), "mean loss: ", str(epoch_loss/n))
    if (epochs%20==0 or epochs==max_epochs-1) and save:
        torch.save(model, fname)
