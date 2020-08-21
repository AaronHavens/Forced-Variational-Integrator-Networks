import csv
import numpy as np
import pickle

traj_length = 30
n_traj = 3
states = np.zeros(((traj_length)*n_traj*7, 6))
controls = np.zeros(((traj_length)*n_traj*7, 2))
news = np.zeros((traj_length)*n_traj*7)

index = 0
for length in [3, 6, 9, 12, 15, 18, 21]:
    #angle_state = np.arcsin(np.sin(angle/360*2*np.pi))
    #print(angle_state)
    for traj in range(n_traj):
        fname = './data_vary_L/'+str(length) + 'cm_' + str(traj+1) + '.xls'
        with open(fname) as trajfile:
            reader = csv.reader(trajfile, delimiter=',')
            trajectory = list(reader)
            #Pb_old = 11
            #Pr_old = 0

            for rowx in range(traj_length):
                values = trajectory[rowx]
                values = [float(i) for i in values]
                #values_next = trajectory[rowx+1]
                #values_next = [float(i) for i in values_next]
                Pb = values[0]
                Pr = values[1]
                #Pb_next = values_next[0]
                #Pr_next = values_next[1]
                X = values[2]
                Y = values[3]
                Z = values[4]
                ac_bend = values[5]
                ac_rot = values[6]
                #ac_bend = Pb_next - Pb
                #ac_rot = Pr_next - Pr
                new = rowx==0
                states[index,:] = np.array([X,Y,Z, Pb, Pr, length])
                controls[index,:] = np.array([ac_bend, ac_rot])
                news[index] = new
                index += 1
        
#data_train = {'states': states, 'controls':controls, 'news': news}
n = 7
data_train  = {'states': states[:traj_length*n_traj*n],
'controls':controls[:traj_length*n_traj*n], 'news':news[:traj_length*n_traj*n]}
data_test = {'states': states[traj_length*n_traj*n:],
'controls':controls[traj_length*n_traj*n:], 'news':news[traj_length*n_traj*n:]}

f = open('data_train_test.pkl', 'wb')
pickle.dump(data_train, f)
f.close()

f = open('data_test_test.pkl', 'wb')
pickle.dump(data_test, f)
f.close()
