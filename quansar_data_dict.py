import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt

index = 0
X = np.load('../quanser-openai-driver/quanser_nof_data_vv.npy')
n_traj = X.shape[0]
print(n_traj)
cutoff = 100
traj_length = X.shape[1]-cutoff
print(traj_length)
x_dim = X.shape[2]-1-2
u_dim = 1
states = np.zeros(((traj_length)*n_traj, x_dim))
controls = np.zeros(((traj_length)*n_traj, u_dim))
news = np.zeros((traj_length)*n_traj)
for traj in range(n_traj):
    Xi = X[traj,cutoff:]
    for rowx in range(traj_length):
        new = rowx==0
        states[index,:] = Xi[rowx,[1,3]]
        states[index,1] *= -1
        controls[index,:] = Xi[rowx,-1:]
        news[index] = new
        index += 1
    

nt = 1
s = 2
data_train = {'states': states[:nt*traj_length:s],
'controls':controls[:nt*traj_length:s], 'news': news[:nt*traj_length:s]}
data_test  = {'states': states[nt*traj_length:], 'controls':controls[nt*traj_length:], 'news':news[nt*traj_length:]}

f = open('quansar_nof_long_data_train.pkl', 'wb')
pickle.dump(data_train, f)
f.close()

f = open('quansar_nof_long_data_test.pkl', 'wb')
pickle.dump(data_test, f)
f.close()
