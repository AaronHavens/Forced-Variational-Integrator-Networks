import torch
from torch.utils import data
import torch.optim as optim
from data_loader import TrajDataset, gym_gen
import VI_model# import VI_VV_loss, VI_VV_model, VI_SV_loss, VI_SV_model, Res_model#, Encoder, Decoder
from torch.nn.utils import clip_grad_norm
import gym
import gym_custom
import numpy as np
from VISV_cem import CEM
import pickle
import argparse

def train_model(traj_dict, model, max_epochs, params, save_name, iteration):
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
            loss = VI_model.VI_SV_loss(model, init_states, states, controls)
    
            loss.backward()
    
            optimizer.step()
    
            epoch_loss += loss.item()/N
            n += 1
        print("iteration: "+ str(iteration), "epoch: " + str(epochs), "mean loss: ", str(epoch_loss/n))
        if (epochs%20==0 or epochs==max_epochs-1) and (save_name is not None):
            fname = 'models/' + save_name + '.pt'
            torch.save(model, fname)

    return model

def append_trajs(traj_dict, traj_dict_):
    traj_dict['states'] = np.concatenate((traj_dict['states'], traj_dict_['states']),axis=0)
    traj_dict['controls'] = np.concatenate((traj_dict['controls'], traj_dict_['controls']),axis=0)
    traj_dict['news'] = np.concatenate((traj_dict['news'], traj_dict_['news']),axis=0)
    return traj_dict

def main():
    parser = argparse.ArgumentParser(description='FVIN Train')
    parser.add_argument('--save_name','-m',type=str)
    parser.add_argument('--load_name','-r',type=str)
    parser.add_argument('--no_train', action='store_true', default=False)
    parser.add_argument('--epochs', '-e', default=10000, type=int)
    parser.add_argument('--n_traj', default=5, type=int)
    parser.add_argument('--model_type', default='VI_VV_model', type=str)
    parser.add_argument('--encoder', action='store_false', default=True)
    parser.add_argument('--n_hid_units',default=100,type=int)
    parser.add_argument('--batch_size',default=2048,type=int)
    parser.add_argument('--env', default='PendulumMod-v0',type=str)
    parser.add_argument('--render', action='store_true',default=False)
    parser.add_argument('--horizon', default=10, type=int)
    parser.add_argument('--iterations', default=0, type=int)
    parser.add_argument('--is_image', action='store_true', default=False)
    args = parser.parse_args()

    params ={   'batch_size': args.batch_size,
                'shuffle': True,
                'num_workers' : 1}
    
    env = gym.make(args.env)
    q_dim = env.env.q_dim
    h = env.env.dt
    x_dim = len(env.observation_space.low)
    u_dim = len(env.action_space.low)
    hid_units = args.n_hid_units
    model_type = getattr(VI_model, args.model_type)
    encoder = args.encoder
    traj_length = env._max_episode_steps

    if args.load_name is not None:
        load_name = 'models/' + args.load_name + '.pt'
        model = torch.load(load_name)
    
    else:
        model = model_type(x_dim, q_dim, u_dim=u_dim, h=h, encoder=encoder)    

    max_epochs = args.epochs
    traj_dict = gym_gen(env, (traj_length-1)*args.n_traj, pi='random', seed=15)
    model = train_model(traj_dict, model, max_epochs, params, args.save_name, 0)
    
    max_epochs = 1000
    # iterations with mpc policy to refine model 
    for i in range(args.iterations):
        pi = CEM(model, 500, 20, env)
        traj_dict_ = gym_gen(env, (traj_length-1)*1, pi=pi, stochastic=True, seed=i)
        traj_dict = append_trajs(traj_dict, traj_dict_)
        model = train_model(traj_dict, model, max_epochs, params, i+1)
        
    
if __name__ == '__main__':
    main()
