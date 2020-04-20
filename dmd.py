import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_softreacher

def gen_data(x0, A, B, U, T):
    n = np.shape(B)[0]
    m = np.shape(B)[1]
    X = np.zeros((n,T))
    X_ = np.zeros((n,T))
    
    A = np.asmatrix(A)
    B = np.asmatrix(B)
    U = np.asmatrix(U)
    #x = np.asmatrix(x0)
    x = x0
    for i in range(T):
        X[:,i] = x.reshape(n,)
        x = np.sqrt(np.abs(x)) + B*U[:,i]#A*x + B*U[:,i] + 0.1*np.tanh(x)
        X_[:,i] = x.reshape(n,)

    return np.asmatrix(X), np.asmatrix(X_)

def gen_data_gym(env, T, delay=1, seed=None):
    trajs = []
    Us = []
    done = False
    if seed is not None:
        env.seed(seed)

    x = env.reset()
    traj = []
    U = []
    for i in range(T):
        traj.append(x)
        u = env.action_space.sample()
        x , r, done, _ = env.step(u)
        U.append(u) 
        if done:
            traj.append(x)
            if len(traj) > delay*2:
                trajs.append(np.array(traj).transpose())
                Us.append(np.array(U).transpose())
            traj = []
            U = []
            x = env.reset()

    
    if not done and len(traj) > delay*2:
        trajs.append(np.array(traj).transpose())
        Us.append(np.array(U).transpose())
    
    X = None
    X_ = None
    U = None
    print('original_seq: ',trajs[0][:3,:2])
    print('original_u: ', Us[0][:,:2])
    for i in range(len(trajs)):
        (n, T) = np.shape(trajs[i])
        (m, _) = np.shape(Us[i])
        Td = T//delay
        print(Td)
        Z = np.zeros((n*delay, T-delay))
        ZU = np.zeros((m*delay, T-delay-1))
        print(np.shape(trajs[i]), np.shape(Us[i]))
        for t in range(T-delay):
            z = trajs[i][:,t]
            for k in range(1,delay):
                z = np.concatenate((z, trajs[i][:,t+k]), axis=0)
            Z[:,t] = z
            #if t != T-delay-1:
            #    zu = Us[i][:, t]
            #    for k in range(1,delay):
            #        zu = np.concatenate((zu, Us[i][:,t+k]), axis=0)
            #    ZU[:,t] = zu

        Xi = Z[:,:-1]
        Xi_ = Z[:,1:]
        Ui = Us[i]
        if i == 0:
            X = Xi
            X_ = Xi_
            U = Ui
        else:
            X = np.concatenate((X, Xi), axis=1)
            X_ = np.concatenate((X_, Xi_), axis=1)
            U = np.concatenate((U, Ui), axis=1)

    return X, X_, U
                    

def eval_dmd(x0, f, A, B, C, U, T):
    n = np.shape(A)[0]
    m = np.shape(B)[1]
    Y = np.zeros((n,T))
    Y_ = np.zeros((n,T))
    
    A = np.asmatrix(A)
    B = np.asmatrix(B)
    U = np.asmatrix(U)
    y = np.asmatrix(f(x0))
    for i in range(T):
        Y[:,i] = y.reshape(n,)
        y = A*y + B*U[:,i]
        Y_[:,i] = y.reshape(n,)

    return Y, Y_

def identity(X):
    return X

def p_n(power):
    def p_(X):
        (n,T) = X.shape
        Y = np.zeros((n*power, T))
        for t in range(T):
            x = X[:,t]
            y = x
            for i in range(2,power+1):
                yi = np.power(x,i)
                y = np.concatenate((y,yi),axis=0)

            Y[:,t] = y.reshape(n*power,)

        return np.asmatrix(Y)
    
    return p_

def p_n_sin(power):
    def p_(X):
        (n,T) = X.shape
        Y = np.zeros((n*(power+1), T))
        for t in range(T):
            x = X[:,t]
            y = x
            for i in range(1,power+1):
                yi = np.power(np.sin(x),i)
                y = np.concatenate((y,yi),axis=0)

            Y[:,t] = y.reshape(n*(power+1),)
        
        return np.asmatrix(Y)
    
    return p_

            
            
def get_A_B(X, X_, U, f, ):
    Y = f(X)
    Y_ = f(X_)

    Y_U = np.block([[Y],[U]])
    Y_X = np.block([[Y_],[X]])

    Z = Y_X*np.linalg.pinv(Y_U)
    z = Y.shape[0]
    A_ = Z[:z, :z]
    B_ = Z[:z, z:]

    return A_, B_

env = gym.make('Pendulum-v0')
#env = gym.make('SoftReacher-v0')
X, X_, U = gen_data_gym(env, 100, delay=2, seed=0)
print(np.shape(X), np.shape(X_), np.shape(U))
print(X[:3,:2])
print(U[:,:2])
#A = np.array([[.99,0.01,0],[0.01,0.98, 0.01],[0.5, 0.12, 0.97]])
#B = np.array([[1,0.1],[0, 0.1],[0, 0.1]])

#n = np.shape(B)[0]
#m = np.shape(B)[1]


#U = np.random.uniform(-1,1, size=(m, T_test))
#x0 = np.array([[1],[1],[1]])

#X, X_ = gen_data(x0, A, B, U, T_test)
#X_data = np.load('../Downloads/X_matrix.npy')

#n_t = 18
#n_a = 26
#split = 1
#X_data_new = np.zeros((26*split, 18*108//split))

#for i in range(108):
#    for j in range(18//split):
#        X_data_new[:,i*18//split + j] = X_data[j*26*split:(j+1)*26*split, i]

#(n, T) = X_data_new.shape
#for i in range(T//18//split):
#    Zi = X_data_new[:,i*18//split:(i+1)*18//split]
#    Xi = Zi[:,:-1]
#    Xi_ = Zi[:,1:]
#    if i == 0:
#        X = Xi
#        X_ = Xi_
#    else:
#        X = np.concatenate((X, Xi), axis=1)
#        X_ = np.concatenate((X_, Xi_), axis=1)
    

X = np.asmatrix(X)
X_ = np.asmatrix(X_)
U = np.asmatrix(U)
x0 = X[:,0]


T_test = X.shape[1]
T_train = T_test
print(X.shape, X_.shape)
power = 1
obser = p_n_sin(power)

A_, B_ = get_A_B(X[:,:T_train], X_[:, :T_train], U[:, :T_train], identity)
A_lift, B_lift = get_A_B(X[:, :T_train], X_[:, :T_train], U[:,:T_train], obser)
C_ = None
N = 20
print('rank A:', np.linalg.matrix_rank(A_))
X_hat, _ = eval_dmd(x0, identity, A_, B_, C_, U, N)
X = np.asarray(X)
X_hat = np.asarray(X_hat)

X_hat_eval, _ = eval_dmd(x0, obser, A_lift, B_lift, C_, U, N)
X_hat_eval = np.asarray(X_hat_eval)
# test 
dim = 0
t = np.arange(0,T_test,1)
plt.style.use('ggplot')
ax1 = plt.subplot(311)
plt.plot(X[dim,:N], label=r'true',c='r')
plt.plot(X_hat_eval[dim, :], label=r'eDMD', c='c')
plt.ylabel(r'$x_1$')

ax2 = plt.subplot(312)
plt.plot(X[dim+1,:N], label=r'true',c='r')
plt.plot(X_hat_eval[dim+1, :], label=r'eDMD', c='c')
plt.ylabel(r'$x_1$')

ax3 = plt.subplot(313)
plt.plot(X[dim+2,:N], label=r'true',c='r')
plt.plot(X_hat_eval[dim+2, :], label=r'eDMD', c='c')
plt.ylabel(r'$x_1$')

plt.legend()
plt.show()
