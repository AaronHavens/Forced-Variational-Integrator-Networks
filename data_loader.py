from torch.utils.data.dataset import Dataset
import numpy as np
from collections import deque
from itertools import islice
import matplotlib.pyplot as plt
#import gym
import time
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
                #plt.plot(np.array(out_states)[-1, :,0])
                #plt.scatter(-1, np.array(initial_states)[-1, 0])
                #plt.show()
        
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

#def gym_imitate(model, pi):

    
def gym_gen(env, T, pi=None):
    x_dim = len(env.observation_space.low)
    u_dim = len(env.action_space.low)
    states = np.zeros((T, x_dim))
    news = np.zeros(T, 'int32')
    controls = np.zeros((T,u_dim))
#    x = env.reset()
    done = False
    env.seed(15)
    for i in range(T):
        if i==0 or done:
            xt = env.reset()
            xtt,_,_,_ = env.step(np.zeros(env.action_space.sample().shape))
            news[i] = 1
        # zero control for testing VI
        if pi=='random':
            u = env.action_space.sample()
        elif pi is not None:
            u,_ = pi.predict(xt, xtt)
        else:
            u = np.zeros(u_dim)
        #else:
        #u = env.env.np_random.uniform(-2, 2, size=(1,))#
        if i%500==0:
            print('samples collected: ', i)
        #print(x)
        #time.sleep(0.1)
        env.render()
        states[i,:] = xtt
        controls[i,:] = np.array([u])
        x,r,done,_ = env.step(u)
        print(done)
        xt = xtt
        xtt = x
        #if i!=0 and i%100==0:
        #    done = True
    #plt.plot(states[:,0])
    #plt.plot(states[:,1])
    #plt.show()
    env.close()
    return {'states':states, 'controls':controls, 'news':news}
