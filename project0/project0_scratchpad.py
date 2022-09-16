# -*- coding: utf-8 -*-

print("hello world")


import numpy as np

#create randomized matrix

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    return np.random.random([n,1])


rand10 = randomization(10)

#size of the matrix and transpose
rand10.shape
rand10.transpose()

x = np.matrix([[2,4,8],[5,7,6],[8,10,11]])
x.shape

y = x.transpose()

x * y


np.matmul(x, y)

np.exp(x)
np.sin(x)
np.cos(x)

np.tanh(x)

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    A = np.random.random([h, w])
    B = np.random.random([h, w])
    
    s = A + B
  
    return [A, B, s]


operations(2, 3)


#Max, min and norm of matrix


np.min(x)
np.max(y)

x.max()
y.min()

#norm2
np.linalg.norm(x)

def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    return np.linalg.norm(A+B)


norm(x, y)

def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    weighted = np.matmul(weights.transpose(), inputs)
    return np.tanh(weighted)

inputs = np.random.random([2, 1])
inputs.shape
weights = np.random.random([2, 1])
weights.shape

neural_network(inputs, weights)


#6. Vectorize function

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    if x<=y:
        return x*y
    else:
        return x/y


scalar_function(1, 2)
scalar_function(2, 2)
scalar_function(4, 2)

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    vec_function = np.vectorize(scalar_function)
    return vec_function(x, y)

print(vector_function(2, 2))

# 7. Introduction to ML packages

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 7]


x = np.linspace(-2*np.pi, 2*np.pi, 400)
y = np.tanh(x)
fig, ax = plt.subplots()
ax.plot(x, y)

x = np.linspace(0, 2*np.pi, 400)
y1 = np.tanh(x)
y2 = np.cos(x**2)
fig, axes = plt.subplots(1, 2, sharey=True)
axes[1].plot(x, y1)
axes[1].plot(x, -y1)
axes[0].plot(x, y2)

import sklearn

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, centers=2, random_state=0)
X[:5], y[:5]

fig, ax = plt.subplots()
for label in [0, 1]:
    mask = (y == label)
    ax.scatter(X[mask, 0], X[mask, 1])


def get_sum_metrics(predictions, metrics=[]):
    for i in range(3):
        metrics.append(lambda x: x + i)

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)

    return sum_metrics