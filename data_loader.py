from torch.utils.data.dataset import Dataset
import numpy as np
from collections import deque
from itertools import islice
#import gym

class TrajDataset(Dataset):
    def __init__(self, traj_dict, T):
    
        self.initial_states, self.input_controls, self.out_states = self.parse_traj(traj_dict, T)

    def __getitem__(self, index):
        return self.initial_states[index], self.input_controls[index], self.out_states[index]

    def __len__(self):
        return len(self.initial_states)
    

    def parse_traj(self, traj_dict, T):
        states  = traj_dict['states']
        news    = traj_dict['news']
        controls = traj_dict['controls']

        states_deq = deque(maxlen=T)
        controls_deq = deque(maxlen=T)
        initial_states = []
        input_controls = []
        out_states = []

        for i in range(len(states)):
            if news[i]:
                states_deq.clear()
                controls_deq.clear()

            states_deq.append(states[i])
            controls_deq.append(controls[i])

            if len(states_deq) == T:
                initial_states.append(states_deq[0])
                input_controls.append(list(islice(controls_deq,0,T-2)))
                out_states.append(list(islice(states_deq,1, T-1)))

        
        return np.array(initial_states), np.array(input_controls), np.array(out_states)



def f(x,u):
    #A = np.transpose(np.array([[1,1],[0,0]]))
    #B = np.transpose(np.array([[0],[1]]))
    dt = 0.1

    return x*dt*1 + u*dt*0.1

def linear_gen_with_noise(env, L):
    T = L
    x_dim = len(env.observation_space.low)
    u_dim = len(env.action_space.low)
    states = np.zeros((T, x_dim))
    news = np.zeros(T, 'int32')
    controls = np.zeros((T,u_dim))
    x = env.reset()
    done = False
    for i in range(T):
        if done:
            x = env.reset()
            news[i] = 1
        u = env.action_space.sample()
        states[i,:] = x
        controls[i,:] = u
        x,r,done,_ = env.step(u)
        
    return {'states':states, 'controls':controls, 'news':news}



## test
#traj_dict = linear_gen_with_noise()
#dataset = TrajDataset(traj_dict,10)
#print(dataset[0])
