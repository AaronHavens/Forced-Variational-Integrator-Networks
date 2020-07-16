import csv
import numpy as np
import pickle

traj_length = 30
n_traj = 6
states = np.zeros(((traj_length-1)*n_traj*6, 6))
controls = np.zeros(((traj_length-1)*n_traj*6, 2))
news = np.zeros((traj_length-1)*n_traj*6)

index = 0
for angle in [270, 300, 330, 360, 30, 60]:
    angle_state = np.arcsin(np.sin(angle/360*2*np.pi))
    print(angle_state)
    for traj in range(n_traj):
        fname = './trajectories/'+str(angle) + '/traj' + str(traj+1) + '.csv'
        with open(fname) as trajfile:
            reader = csv.reader(trajfile, delimiter=',')
            trajectory = list(reader)
            Pb_old = 11
            Pr_old = 0

            for rowx in range(traj_length-1):
                values = trajectory[rowx]
                values = [float(i) for i in values]
                values_next = trajectory[rowx+1]
                values_next = [float(i) for i in values_next]
                Pb = values[0]
                Pr = values[1]
                Pb_next = values_next[0]
                Pr_next = values_next[1]
                X = values[2]
                Y = values[3]
                Z = values[4]
                ac_bend = Pb_next - Pb
                ac_rot = Pr_next - Pr
                new = rowx==0
                states[index,:] = np.array([X,Y,Z, Pb, Pr, angle_state])
                controls[index,:] = np.array([ac_bend, ac_rot])
                news[index] = new
                index += 1
        
#data_train = {'states': states, 'controls':controls, 'news': news}
data_train  = {'states': states[:29*n_traj*5],
'controls':controls[:29*n_traj*5], 'news':news[:29*n_traj*5]}
data_test = {'states': states[29*n_traj*5:],
'controls':controls[29*n_traj*5:], 'news':news[29*n_traj*5:]}

f = open('data_train_test.pkl', 'wb')
pickle.dump(data_train, f)
f.close()

f = open('data_test_test.pkl', 'wb')
pickle.dump(data_test, f)
f.close()
