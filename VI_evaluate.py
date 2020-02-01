import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from VI_model import VI_model, Encoder, Decoder
import gym_softreacher
from matplotlib import rc
rc('text', usetex=True)

env = gym.make('Pendulum-v0')
fname = 'models/vi_test.pt'
q_dim = 2
h=0.1 
x_dim = len(env.observation_space.low) 
u_dim = len(env.action_space.low) 
hid_units = 64

encode = Encoder(x_dim, q_dim, hid_units) 
decode = Decoder(x_dim, q_dim, hid_units) 
model = VI_model(encode,decode, q_dim, h=0.1) 

checkpoint = torch.load(fname) 
print(checkpoint)
model.load_state_dict(checkpoint)

H = 50
env.seed(0)
x = env.reset()
x_hat = np.copy(x)

xs = np.zeros((H,x_dim))
xs_hat = np.zeros((H,x_dim))

for i in range(H):
    if i==0:
        q_next, x_hat_next = model.predict_from_x(x_hat)
    else:
        q_next, x_hat_next = model.predict_from_q(q)

    u = np.zeros(env.action_space.sample().shape)
    x_next, r, done, _ = env.step(u)
    xs[i,:] = x_next
    xs_hat[i,:] = x_hat_next
    x = x_next
    x_hat = x_hat_next
    q = q_next

plt.style.use('ggplot')
plt.plot(xs[:,1], label=r'true')
plt.plot(xs_hat[:,1], label=r'predicted')
plt.xlabel(r'action step')
plt.ylabel(r'$||x||_2$ distance')
plt.ylim([-2,2])
plt.legend()
plt.show()
