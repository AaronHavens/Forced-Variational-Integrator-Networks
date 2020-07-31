import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


def dfdx2(t, y_):
    x = y_[0]
    y = y_[1]
    theta = y_[2]
    xdot = y_[3]
    ydot = y_[4]
    thetadot = y_[5]
    u1 = y_[6]
    u2 = y_[7]
    
    g = 9.81
    m = 0.2
    I = 0.1
    xddot = -1/m*np.sin(theta) * u1
    yddot = 1/m*np.cos(theta) * u1 - g
    thetaddot = 1/I * u2
    return [xdot, ydot, thetadot, xddot, yddot, thetaddot, 0, 0]


def step(x, u, tau):
    y = [x[0],x[1],x[2],x[3],x[4],x[5],u[0],u[1]]
    y_next = integrate.solve_ivp(dfdx2, [0.0, tau], y)
    x_next = y_next.y
    x = x_next[0][-1]
    y = x_next[1][-1]
    theta = x_next[2][-1]
    x_dot = x_next[3][-1]
    y_dot = x_next[4][-1]
    theta_dot = x_next[5][-1]
    
    return np.array([x, y, theta, x_dot, y_dot, theta_dot])

class pid():
    def __init__(self, env):
        self.I = env.env.I
        self.m = env.env.m
        self.g = env.env.g

    def predict(self, xt, x_):
        xdes = 0
        ydes = 0
        g = self.g
        m = self.m
        I = self.I

        kpy=50
        kdy=3
        kpx=0.8
        kdx=1
        kpth=20
        kdth=10
        x = x_[0]
        y = x_[1]
        theta = x_[2]
        xdot = x_[3]
        ydot = x_[4]
        thetadot = x_[5]

        u1 = m*(g + kdy*(-ydot) + kpy*(ydes-y))
        #u1 =  kdy*(-ydot) + kpy*(ydes-y)
        thetac = -1/g*(kdx*(-xdot) + kpx*(xdes-x))
        #thetac = kdx*(-xdot) + kpx*(xdes-x) 
        u2 = I*(kpth*(thetac-theta) + kdth * (-thetadot))
        #u2 = kpth*(thetac-theta) + kdth * (-thetadot)
        return np.array([u1, u2]), None


