import numpy as np
import matplotlib.pyplot as plt




def relu(x):
    return max(0, x)


x = np.linspace(-1, 1, 1000)
y = np.zeros(len(x))

eps = 0.1
w1 = 1/eps
b1 = 1-eps

w2 = eps
b2 = 0
for i in range(len(x)):
    #y[i] = relu(-1/eps*relu(-x[i]+eps)+1)
    y[i] = -1/eps*relu(-x[i]+eps)+1

    #y[i] = relu(x[i], 1/eps, 0)

plt.plot(x,y)
plt.plot(np.zeros(100), np.linspace(-1,1,100))
plt.ylim(-1, 1)
plt.show()
