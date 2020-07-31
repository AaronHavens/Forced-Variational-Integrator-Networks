import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import scipy.integrate as integrate
import torch
class Quadrotor2DEnv(gym.Env):
  
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.g = 9.8
        self.m = 0.2
        self.I= 0.1
        self.tau = 0.005#0.001 # seconds between state updates
        self.dt = 0.1
        self.Nt = int(self.dt//self.tau)


        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

        self.action_space = spaces.Box(np.array([-20, -5]), np.array([20, 5]))#spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def f(self, t, y_):
        x = y_[0]
        y = y_[1]
        theta = y_[2]
        xdot = y_[3]
        ydot = y_[4]
        thetadot = y_[5]
        u1 = y_[6]
        u2 = y_[7]

        g = self.g
        m = self.m
        I = self.I
        xddot = -1/m*np.sin(theta) * u1
        yddot = 1/m*np.cos(theta) * u1 - g
        thetaddot = 1/I * u2
        return [xdot, ydot, thetadot, xddot, yddot, thetaddot, 0, 0]




    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, y, theta, x_dot, y_dot, theta_dot = state
        force = action#self.force_mag if action==1 else -self.force_mag
                
        for i in range(self.Nt):
            
            y_next = integrate.solve_ivp(self.f, [0.0, self.tau], 
                                    [x, y, theta, x_dot, y_dot, theta_dot,
                                    force[0], force[1]])
            x_next = y_next.y
            x = x_next[0][-1]
            y = x_next[1][-1]
            theta = x_next[2][-1]
            x_dot = x_next[3][-1]
            y_dot = x_next[4][-1]
            theta_dot = x_next[5][-1]
            
        self.state = (x, y, theta, x_dot, y_dot, theta_dot)
        
        ############ for testing
        reward = 0
        done = False
        ############
        return self.get_obs(np.array(self.state)), reward, done, {}
    
    def reward(self, model, y__, y_, u_):
        cost = 0
        x = model.decoder(y_).double()#x_.double()
        u = u_.double()
        with torch.no_grad():
            cost += torch.pow(x[:,0], 2)
            cost += 2*torch.pow(x[:,1], 2)
            cost += 10*torch.pow(x[:,2], 2)
            cost += 0.1*torch.pow(x[:,3], 2)
            cost += 0.1*torch.pow(x[:,4], 2)
            cost += 0.1*torch.pow(x[:,5], 2)
            cost += 0.001*torch.pow(u[:,0], 2)
            cost += 0.001*torch.pow(u[:,1], 2)

        return -cost.data.numpy()

    def reward_test(self, x__, x_, u_):
        cost = 0
        cost += x_[0]**2 + 2*x_[1]**2 + 10*x_[2]**2
        cost += 0.1*(x_[3]**2 + x_[4]**2 + x_[5]**2)
        cost += 0.001*(u_[0]**2 + u_[1]**2)

        return -cost

    def get_obs(self, x):
        return np.array([x[0], x[1], x[2], x[3], x[4], x[5]])

    def reset(self):
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(6,))
        return self.get_obs(np.array(self.state))

    def render(self, mode='human'):
        return 0

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
