import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from VI_model import VI_SV_model, VI_VV_model, Res_model
import gym_softreacher
from matplotlib import rc
from multi_env import MultiEnv
from mpl_toolkits.mplot3d import Axes3D
import gym_custom

rc('text', usetex=True)

env = gym.make('QPendulum-v0')
fname = 'models/vi_base_forced.pt'
q_dim = 1
h = 0.1
x_dim = len(env.observation_space.low) 
u_dim = len(env.action_space.low) 
hid_units = 100
#encode = Encoder(x_dim, q_dim, hid_units) 
#decode = Decoder(x_dim, q_dim, hid_units) 
#model = VI_VV_model(q_dim, u_dim=1, h=h) 
model = Res_model(q_dim, u_dim=1, h=h)
checkpoint = torch.load(fname) 
#print(checkpoint)
model.load_state_dict(checkpoint)

H = 200
env.seed(16)
x = env.reset()
x_hat = np.copy(x)
#x,_,_,_ = env.step(np.zeros(np.shape(env.action_space.sample())))
#x_hatt = np.copy(x)
xs = np.zeros((H,x_dim))
xs_hat = np.zeros((H,x_dim))
xs[0,:] = x
q_t = torch.from_numpy(np.expand_dims(x_hat, axis=0)).float()
#q_tt = torch.from_numpy(np.expand_dims(x_hatt, axis=0)).float()
xs_hat[0,:] = x#model.decoder(q_t1.float()).detach().numpy()
E = np.zeros(H-1)
E_hat = np.zeros(H-1)
for i in range(1, H):
    #q_next, x_hat_next = model.predict_from_q(q_t, q_tt)
    #u = np.zeros(env.action_space.low.shape)
    u = env.np_random.uniform(-1, 1, size=(1,))#env.action_space.sample()
    q_next, x_hat_next = model.predict_from_q(q_t, u)

    x_next, r, done, _ = env.step(u)
    xs[i,:] = x_next
    xs_hat[i,:] = x_hat_next
    #x_hat = x_hat_next
    #q_t = q_tt
    #q_tt = q_next
    E[i-1] = 1/2*x_next[1]**2 - 10*np.cos(x_next[0])
    E_hat[i-1] = 1/2*x_hat_next[1]**2 - 10*np.cos(x_hat_next[0])
    q_t = q_next
    if done:
        break



#plt.scatter(np.sin(xs[:i,0]), -np.cos(xs[:i,0]))
plt.plot(E)
plt.plot(E_hat)
plt.ylim([-10,15])

plt.show()
plt.close()

plt.style.use('ggplot')
ax1 = plt.subplot(211)
plt.plot(xs[:i,0], label=r'true')
plt.plot(xs_hat[:i,0], label=r'predicted')

ax2 = plt.subplot(212)
plt.plot(xs[:i,1], label=r'true')
plt.plot(xs_hat[:i,1], label=r'predicted')

#ax3 = plt.subplot(313)
#plt.plot(xs[:i,2], label=r'true')
#plt.plot(xs_hat[:i,2], label=r'predicted')

#ax4 = plt.subplot(414,projection='3d')
#ax4.plot(qs[:i,0], qs[:i,1], zs=qs[:i,2])
#plt.xlabel(r'time step $h=0.05$ [s]')
#plt.ylim([-2,2])
plt.legend()
plt.show()
