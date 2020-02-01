from torch.utils.data.dataset import Dataset
import numpy as np
from collections import deque
from itertools import islice
#import gym

class TrajDataset(Dataset):
    def __init__(self, traj_dict, T):
    
        self.initial_states, self.out_states = self.parse_traj(traj_dict, T)

    def __getitem__(self, index):
        return self.initial_states[index], self.out_states[index]

    def __len__(self):
        return len(self.initial_states)
    

    def parse_traj(self, traj_dict, T):
        states  = traj_dict['states']
        news    = traj_dict['news']

        states_deq = deque(maxlen=T)
        initial_states = []
        out_states = []

        for i in range(len(states)):
            if news[i]:
                states_deq.clear()

            states_deq.append(states[i])

            if len(states_deq) == T:
                initial_states.append(states_deq[0])
                out_states.append(list(islice(states_deq,1, T-1)))

        
        return np.array(initial_states), np.array(out_states)

