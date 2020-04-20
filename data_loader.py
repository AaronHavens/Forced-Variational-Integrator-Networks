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
                input_controls.append(list(islice(controls_deq,0,T-1)))
                out_states.append(list(islice(states_deq,1, T)))

        
        return np.array(initial_states), np.array(input_controls), np.array(out_states)

class ImitateDataset(Dataset):
    def __init__(self, traj_dict, T):
    
        self.initial_states, self.pi_controls = self.parse_traj(traj_dict, T)

    def __getitem__(self, index):
        return self.initial_states[index], self.pi_controls[index]

    def __len__(self):
        return len(self.initial_states)
    

    def parse_traj(self, traj_dict, T):
        states  = traj_dict['states']
        controls = traj_dict['controls']

        initial_states = []
        pi_controls = []

        for i in range(len(states)):
            initial_states.append(states[i])
            pi_controls.append(controls[i])

        
        return np.array(initial_states), np.array(pi_controls)


def gym_gen(env, L, pi=None):
    T = L
    x_dim = len(env.observation_space.low)
    u_dim = len(env.action_space.low)
    states = np.zeros((T, x_dim))
    news = np.zeros(T, 'int32')
    controls = np.zeros((T,u_dim))
#    x = env.reset()
    done = False
    env.seed(10)
    for i in range(T):
        if i==0 or done:
            x = env.reset()
            news[i] = 1
        # zero control for testing VI
        #u = np.zeros(env.action_space.sample().shape)
        #if pi is not None:
        #    u,_ = pi.predict(x)
        #else:
        u = env.np_random.uniform(-1, 1, size=(1,))#env.action_space.sample()
        if i%500==0:
            print('samples collected: ', i)
        print(x)
        states[i,:] = x
        controls[i,:] = u
        x,r,done,_ = env.step(u)
        #if i%60==0:
        #  done = True
        
    return {'states':states, 'controls':controls, 'news':news}
