import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import scipy.integrate as integrate
import torch
import matplotlib.pyplot as plt
class ImPendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.1
        self.viewer = None

        high = np.ones(785)#np.array([np.inf, self.max_speed])
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
            costh = torch.cos(x[:,0])
            sinth = torch.sin(x[:,0])
            theta = torch.atan2(sinth,costh)
            
            cost += torch.pow(theta+np.pi, 2)
            cost += 0.1*torch.pow(x[:,1], 2)
            cost += 0.001*torch.pow(u[:,0], 2)
        
        return -cost.data.numpy()

    def reward_test(self, x__, x_, u_):
        cost = 0
        costh = np.cos(x_[0])
        sinth = np.sin(x_[0])
        theta = np.arctan2(sinth,costh)
            
        cost += (theta+np.pi)**2
        cost += 0.1*x_[1]**2
        cost += 0.001*u_[0]**2
        
        return -cost



    def step(self,u):
        th, thdot = self.state # th := theta

        g = 10.
        m = 1.
        l = 1.
        dt = self.dt#np.random.normal(self.dt, 0.01)
        
        
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        def f(t, y):
            theta, theta_dot = y
            return [theta_dot, -g/l * np.sin(theta) + 3./(m*l**2)*u - 0.2/l*theta_dot]

        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
        y_next = integrate.solve_ivp(f, [0.0, dt], [th, thdot])
        #newthdot1 = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        #newth1 = th + newthdot1*dt
        newthdot = y_next.y[1][-1]
        newth = y_next.y[0][-1]
        #newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        #sign = [-1, 1]
        #theta = np.pi - self.np_random.uniform(low=0.1, high=1./2.*np.pi)
        #theta = np.random.uniform(low=np.pi/3, high=2.*np.pi-np.pi/3)
        #theta = self.np_random.choice(sign)*theta
        #theta_dot = self.np_random.uniform(-0.5, 0.5)
        #self.state = np.array([theta, theta_dot])
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        length = 1.0
        q = self.state[0]
        qdot = self.state[1]
        x = length * np.sin(q)
        y = -length * np.cos(q)

        f, ax = plt.subplots(figsize=(1, 1), dpi=28)
        f_lim = length+0.2

        ys = []
        #for traj in range(q.shape[0]):
        #    y_traj = []
        #    for t in range(q.shape[1]):
        plt.cla()
        ax.set_xlim(-f_lim, f_lim)
        ax.set_ylim(-f_lim, f_lim)
        ax.plot([0, x],[0, y], linewidth=12, color="black")
        ax.axis('off')
        f.canvas.draw()
        y_t = np.array(f.canvas.renderer.buffer_rgba())[:, :, :1]
        y_t[y_t > 0] = 255
        y_t[y_t == 0] = 1.
        y_t[y_t == 255] = 0.
        #y_traj = np.float32(y_t[None])
        #y_traj = np.vstack(y_traj)
        #ys.append(y_traj[None])
        #ys = np.vstack(ys)
        plt.close()
        y = y_t.reshape(784)
        y = np.concatenate((y,np.array([qdot])),axis=0)
        #theta += np.random.normal(0,0.02)
        #thetadot += np.random.normal(0,0.3)
        return y

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
