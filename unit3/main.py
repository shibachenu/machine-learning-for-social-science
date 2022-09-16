import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt
from activations import *

# test weighted average
X = np.matrix([1, 0])
w0 = -3
w = np.matrix([1, -1])

z = weighted_average(X, w, w0)
print("Weighted average is : " + str(z))

z_relu = activation_relu(z)
print("Relu z activated: " + str(z_relu))

z_tanh = activation_tanh(z)
print("Tanh z activated: " + str(z_tanh))



def plot_separation(X, w1, w01, w2, w02):
    f1 = activation_linear(weighted_average(X, w1, w01)).tolist()
    f2 = activation_linear(weighted_average(X, w2, w02)).tolist()

    plt.scatter(f1, f2, c=colors)

#examples
X = np.matrix([[-1, -1], [1, -1], [-1, 1], [1, 1]])
Y = np.array([1, -1, -1, 1])
colors = ['blue' if l == 1 else 'red' for l in Y]

#test different Ws
w1 = np.array([0, 0])
w01 = 0
w2 = np.array([0, 0])
w02 = 0

plot_separation(X, w1, w01, w2, w02)
#not seprable


w1 = np.array([2, 2])
w01 = 1
w2 = np.array([-2, -2])
w02 = 1

plot_separation(X, w1, w01, w2, w02)
#not seperable


w1 = np.array([-2, -2])
w01 = 1
w2 = np.array([2, 2])
w02 = 1

plot_separation(X, w1, w01, w2, w02)
#not seperable

#try different activation function
w1 = np.array([1, -1])
w01 = 1

z1 = weighted_average(X, w1, w01)

w2 = np.array([-1, 1])
w02 = 1

z2 = weighted_average(X, w2, w02)

#linear
f1 = activation_linear(z1).tolist()
f2 = activation_linear(z2).tolist()

plt.scatter(f1, f2, c=colors)

#reLU
f1 = activation_relu(z1).tolist()
f2 = activation_relu(z2).tolist()
plt.scatter(f1, f2, c=colors)

#tanh
f1 = activation_tanh(z1).tolist()
f2 = activation_tanh(z2).tolist()
plt.scatter(f1, f2, c=colors)


f_t = sigmoid(1+5)
i_t = sigmoid(1+5)
o_t = sigmoid(1 + 5)
c_t = f_t * 1 + i_t*activation_tanh(1+5)
h_t = o_t * activation_tanh(c_t)

print("h_t: output is: "+str(h_t))

