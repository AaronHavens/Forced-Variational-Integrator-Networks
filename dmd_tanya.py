import numpy as np
import matplotlib.pyplot as plt
import pydmd
import scipy.io as io

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
    print(x0)
    for i in range(T):
        X[:,i] = x.reshape(n,)
        x = np.sqrt(np.abs(x)) + B*U[:,i]#A*x + B*U[:,i] + 0.1*np.tanh(x)
        print(x)
        X_[:,i] = x.reshape(n,)

    return np.asmatrix(X), np.asmatrix(X_)

def eval_dmd(x0, f, A, T):
    n = np.shape(A)[0]
    print('n', n)
    Y = np.zeros((n,T))
    Y_ = np.zeros((n,T))
    
    A = np.asmatrix(A)
    y = np.asmatrix(f(x0))
    for i in range(T):
        Y[:,i] = y.reshape(n,)
        y = A*y
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



T_train = 50
T_test = 100

#U = np.random.uniform(-1,1, size=(m, T_test))
#x0 = np.array([[1],[1],[1]])

#X, X_ = gen_data(x0, A, B, U, T_test)
X_data = np.load('../Downloads/X_matrix.npy')
io.savemat('X_data.mat',{'data': X_data})
X = X_data[:, :-1]
X_ = X_data[:, 1:]
x0 = X[:,0]

T_test = X.shape[1]
T_train = T_test

#X_embedded = np.vstack((X_data[:,:-1], X_data[:,1:]))
d = pydmd.DMD(svd_rank=-1, exact=False).fit(X)
A = d.atilde
M = d.modes
print(M)
print(A.shape)

U, S, V = np.linalg.svd(X, full_matrices=False)
#V = V.conj().T
#A_ = X_*V2*np.linalg.inv(np.diag(Sig2))*U2.conj().T
#A_ = X_.dot(V).dot(U.conj().T) * np.reciprocal(S)
A_ = np.dot(np.dot(np.dot(U.conj().T, X_), V.conj().T),np.linalg.inv(np.diag(S)))
print(A_.shape)
eig, v = np.linalg.eig(A_)
phi = np.dot(X_, np.dot(V ,np.dot(np.linalg.inv(np.diag(S)),v)))
b = np.dot(np.linalg.pinv(phi),X[:,0])
Z_hat = np.zeros((107, T_train))
X_hat = np.zeros((468, T_train))
print('phi: ',phi.shape)
Z_hat[:,0] = b
X_hat[:,0] = X[:,0]
print('b: ',b.shape)
for i in range(1,T_train):
    Z_hat[:,i] = np.dot(np.diag(eig),b)
    X_hat[:,i] = np.dot(U, Z_hat[:,i])





#X_hat, _ = eval_dmd(x0, identity, A_,  T_test)
#X = np.asarray(X)
#X_hat = np.asarray(X_hat)

# test 

t = np.arange(0,T_test,1)

plt.style.use('ggplot')
ax1 = plt.subplot(111)
plt.plot(t, X[42], label=r'true',c='r')
plt.plot(X_hat[42], label=r'eDMD', c='c')
plt.ylabel(r'$x_1$')
plt.ylabel(r'$x_3$')
plt.legend()
plt.show()
