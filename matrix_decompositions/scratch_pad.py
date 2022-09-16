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
