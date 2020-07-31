"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import scipy.integrate as integrate
import torch
class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value between ±0.05

    Episode Termination:
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = -9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = 2*self.length#(self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.001 # seconds between state updates
        self.dt = 0.1
        self.Nt = int(self.dt//self.tau)
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([np.inf, 1,1,
            #self.x_threshold * 2,
            #self.theta_threshold_radians * 2,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(np.array([-10]), np.array([10]))#spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def f(self, t, y):
        x = y[0]
        theta = y[1]
        x_dot = y[2]
        theta_dot = y[3]
        force = y[4]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        l = 1
        g = -self.gravity
        m1 = self.masscart
        m2 = self.masspole
        Mat1 = np.matrix([[costheta, l],[m1 + m2, m2*l*costheta]])
        Mat2 = np.matrix([[g*sintheta],[force + m2*l*theta_dot**2*sintheta]])
        frict_vec = np.matrix([[0.1*x_dot],[0.05*theta_dot]])

        ddot = np.linalg.inv(Mat1)*(Mat2-frict_vec)
        #temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        #thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        #xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        #temp = (force+self.polemass_length*theta_dot**2*sintheta*self.masspole)

        #xacc = -self.gravity*self.polemass_length*costheta*sintheta*self.masspole
        #xacc += -self.polemass_length*temp
        #temp2 = self.polemass_length*(-self.total_mass+costheta**2*self.masspole)
        #xacc = xacc/temp2
        #thetaacc = -self.gravity*sintheta*(-self.total_mass)
        #thetaacc += costheta*temp
        #thetaacc = thetaacc/temp2
        xacc = ddot[0,0]
        thetaacc = ddot[1,0]
        return [x_dot, theta_dot, xacc, thetaacc, 0]
    
    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, theta, x_dot, theta_dot = state
        force = action#self.force_mag if action==1 else -self.force_mag
                
        for i in range(self.Nt):
            
            y_next = integrate.solve_ivp(self.f, [0.0, self.tau], 
                                    [x, theta, x_dot, theta_dot, force])
            x_next = y_next.y
            x = x_next[0][-1]
            theta = x_next[1][-1]
            x_dot = x_next[2][-1]
            theta_dot = x_next[3][-1]
            
        self.state = (x, theta, x_dot, theta_dot)
        #if self.kinematics_integrator == 'euler':
        #    x  = x + self.tau * x_dot
        #    x_dot = x_dot + self.tau * xacc
        #    theta = theta + self.tau * theta_dot
        #    theta_dot = theta_dot + self.tau * thetaacc
        #else: # semi-implicit euler
        #    x_dot = x_dot + self.tau * xacc
        #    x  = x + self.tau * x_dot
        #    theta_dot = theta_dot + self.tau * thetaacc
        #    theta = theta + self.tau * theta_dot
       #,ot,theta,theta_dot)
        #self.state = (x, theta, x_dot[0], theta_dot[0])
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        ############ for testing
        done = False
        ############
        return self.get_obs(np.array(self.state)), reward, done, {}
    
    def reward(self, model, y__, y_, u_):
        cost = 0
        x = model.decoder(y_).double()#x_.double()
        u = u_.double()
        with torch.no_grad():
            #costh = torch.cos(x[:,0])
            #sinth = torch.sin(x[:,0])
            #theta = torch.atan2(sinth,costh)
            
            theta = torch.atan2(x[:,2],x[:,1])
            cost += 1*torch.pow(x[:,0],2)
            cost += 5*torch.pow(theta, 2)
            cost += 0.1*torch.pow(x[:,3], 2)
            cost += 0.1*torch.pow(x[:,4], 2)
            cost += 0.01*torch.pow(u[:,0], 2)

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

    def get_obs(self, x):
        return np.array([x[0], np.cos(x[1]), np.sin(x[1]), x[2], x[3]])

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state[1] = self.np_random.uniform(-np.pi/4,-np.pi/4+2*(np.pi-np.pi/4))
        self.state[3] = self.np_random.uniform(-1, 1)
        self.steps_beyond_done = None
        return self.get_obs(np.array(self.state))

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
