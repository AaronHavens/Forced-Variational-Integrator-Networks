import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import scipy.integrate as integrate
import torch
class SpringMassEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.1
        self.viewer = None

        high = np.array([np.inf, self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def reward(self, model, y__, y_, u_):
        cost = 0
        x = model.decoder(y_).double()#x_.double()
        u = u_.double()
        with torch.no_grad():
            e = x[:,0] - 1
            cost += torch.pow(e, 2)
            cost += 0.1*torch.pow(x[:,1], 2)
            cost += 0.001*torch.pow(u[:,0], 2)

        return -cost.data.numpy()

    def reward_test(self, x__, x_, u_):
        cost = 0
        e = x_[0] - 1
        cost += (e)**2
        cost += 0.1*x_[1]**2
        cost += 0.001*u_[0]**2

        return -cost


    def step(self,u):
        th, thdot = self.state # th := theta

        k = 1.
        m = 1.
        l = 1.
        dt = self.dt
  

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        def f(t, y):
            theta, theta_dot = y
            return [theta_dot, u/m - k*theta/m - 0.2/m*theta_dot]
        costs = 0
        #costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
        y_next = integrate.solve_ivp(f, [0.0, dt], [th, thdot])
        #newthdot1 = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        #newth1 = th + newthdot1*dt
        newthdot = y_next.y[1][-1]
        newth = y_next.y[0][-1]
        #newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        self.state = np.array([newth, newthdot])
        return self.state, -costs, False, {}

    def reset(self):
        #sign = [-1, 1]
        #theta = np.pi - self.np_random.uniform(low=0.1, high=1./2.*np.pi)
        #theta = np.random.uniform(low=np.pi/3, high=2.*np.pi-np.pi/3)
        #theta = self.np_random.choice(sign)*theta
        theta = self.np_random.uniform(-1, 1)
        theta_dot = self.np_random.uniform(-1, 1)
        self.state = np.array([theta, theta_dot])
        #high = np.array([np.pi, 1])
        #self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self.state

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
