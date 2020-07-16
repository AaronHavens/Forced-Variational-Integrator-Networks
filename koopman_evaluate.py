import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from koopman_cem import CEM
import gym_softreacher
from matplotlib import rc
from multi_env import MultiEnv
import control
from lqr_finite import MPC
import gym_custom

rc('text', usetex=True)

model = torch.load('models/pendulum_enc.pt')
#env = gym.make('Pendulum-v0')
#A = model.A.weight.data
#B = model.B.weight.data
#C = model.A.bias.data
#C = np.asmatrix(C).reshape(len(C),1)
#z = A.shape[0]
#Q = np.zeros((z,z))
#Q[0,0] = 1
#Q[1,1] = 1
#Q[2,2] = 1
#R = 1
#C = np.zeros(z)
#C[0] = np.sqrt(5)
#C[1] = np.sqrt(5)
#C[2] = np.sqrt(5)
#print(z)
#print(np.linalg.matrix_rank(control.ctrb(A,B)))
#print(np.linalg.matrix_rank(control.obsv(A,C)))
#pi = MPC(model, A, B, Q, R, z, 2, T=20, C=None, open_loop=False)
#pi.predict(np.zeros(5),0)
#env = MultiEnv('SoftReacher-v0')
env = gym.make('QPendulum-v0')
#sac_model = SAC.load('sac_cheetah_wexpert_iclr_1_3')
x_dim = len(env.observation_space.low)
u_dim = len(env.action_space.low)
#pi = CEM(model, 500, 10, env.action_space.low, env.action_space.high)
H = 200
#env.seed(15)
x = env.reset()
x_hat = np.copy(x)

xs = np.zeros((H,x_dim))
xs_hat = np.zeros((H,x_dim))
r_tot = 0
E = np.zeros(H-1)
E_hat = np.zeros(H-1)

for i in range(H):
    #u = env.action_space.sample()
    #u, _ = sac_model.predict(x)
    #u,_ = pi.predict(x)
    #u = pi.predict(x, i)# + np.random.normal(0,0.05, size=2)
    u = np.zeros(env.action_space.sample().shape)
    if i==0:
        z_next = model.predict_from_state(x_hat, u)
    else:
        z_next = model.predict_from_latent(z, u)
    x_hat_next = model.decode(z_next)
    x_next, r, done, _ = env.step(u)
    r_tot += r
    #env.render()
    xs[i,:] = x_next
    xs_hat[i,:] = x_hat_next
    
    #E[i-1] = 1/2*(env.env.state[1]**2) -10*np.cos(env.env.state[0]-np.pi)
    E[i-1] = 1/2*x_next[1]**2 - 10*np.cos(x_next[0])#+np.pi)
    #theta_ = np.arctan2(x_hat_next[1], x_hat_next[0])
    #thetadot_ = x_hat_next[2]
    #E_hat[i-1] = 1/2*thetadot_**2 - 10*np.cos(theta_-np.pi)
    E_hat[i-1] = 1/2*x_hat_next[1]**2 - 10*np.cos(x_hat_next[0])#+np.pi)


    x = x_next
    x_hat = x_hat_next
    z = z_next
   
   
   #print(np.linalg.norm(xs[i,:3]))
    #print(u)
    #if done:
    #    break
plt.plot(E,label='true energy')
plt.plot(E_hat, label='predicted energy')
plt.ylim([-10,15])
plt.legend()
plt.show()
plt.close()

print('total cost', r_tot)
#normed_true = np.linalg.norm(xs[:i,:3], axis=1)
#normed_predict = np.linalg.norm(xs_hat[:i,:3], axis=1)

plt.style.use('ggplot')
ax1 = plt.subplot(311)
plt.plot(xs[:i,0], label=r'true')
plt.plot(xs_hat[:i,0], label=r'predicted')
plt.ylabel(r'$\bar x_1$ distance [cm]')

ax2 = plt.subplot(312)
plt.plot(xs[:i,1], label=r'true')
plt.plot(xs_hat[:i,1], label=r'predicted')
plt.ylabel(r'$\bar x_2$ distance [cm]')

#ax1 = plt.subplot(313)
#plt.plot(xs[:i,2], label=r'true')
#plt.plot(xs_hat[:i,2], label=r'predicted')
#plt.xlabel(r'action step')
#plt.ylabel(r'$\bar x_3$ distance [cm]')
plt.legend()
plt.show()
