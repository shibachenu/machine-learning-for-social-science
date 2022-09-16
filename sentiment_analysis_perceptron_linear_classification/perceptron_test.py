import math

import numpy as np
from perceptron_algo import perceptron, perceptron_offset

"""
x1 = np.matrix([[-1, -1], [1, 0], [-1, 1.5]])
y1 = np.array([1, -1, 1])
t = 2

print("Starting from x1")
theta1, error = perceptron(x1, y1, t)
print("theta from x1: " + str(theta1) + ", error count: "+str(error))

x2 = np.matrix([[1, 0], [-1, -1], [-1, 1.5]])
y2 = np.array([-1, 1, 1])

print("Starting from x2")
theta2, error = perceptron(x2, y2, t)
print("theta from x2: " + str(theta2) + ", error count: "+str(error))


x3 = np.matrix([[-1, -1], [1, 0], [-1, 10]])
y3 = np.array([1, -1, 1])

t = 5

print("Starting from x1 in X3")
theta3, error = perceptron(x3, y3, t)
print("theta from x3 starting from x1: " + str(theta3) + ", error count: "+str(error))

x4 = np.matrix([[1, 0], [-1, -1],  [-1, 10]])
y4 = np.array([-1, 1, 1])

print("Starting from x2 in X3")
theta4, error = perceptron(x4, y4, t)
print("theta from x4 starting from x2: " + str(theta4) + ", error count: "+str(error))


x5 = np.matrix([[-4, 2], [-2, 1],  [-1, -1], [2, 2], [1, -2]])
y5 = np.array([1, 1, -1, -1, -1])
t = 5

theta5, error = perceptron_offset(x5, y5, t)
print("theta after perceptron with offset: " + str(theta5) + ", error count: "+str(error))


x6 = np.matrix([[-1, 1], [1, -1], [1, 1], [2, 2]])
y6 = np.array([1, 1, -1, -1])
t = 5

theta6, error = perceptron_offset(x6, y6, t)
print("theta after perceptron with offset: " + str(theta6) + ", error count: "+str(error))
"""


def testConvergence(d, t):
    x = np.zeros([d, d])
    y = np.ones(d).reshape([d, 1])
    for i in range(d):  # row
        for j in range(d):  # col
            x[i][j] = 0 if i != j else math.cos((i + 1) * math.pi)
    print("data with size: "+str(x.shape))
    theta, error = perceptron(x, y, t, False)
    print("theta after perceptron with offset: " + str(theta) + ", error count: " + str(error))


t = 3
#for d in range(2, 10):
 #   testConvergence(d, t)

d = 3
testConvergence(d, t)





feature_matrix = np.matrix([
    [-0.44600691, -0.48995308, 0.10672829, -0.23279704, 0.34922046, 0.49462688, 0.23413718, 0.24492144, 0.04536957, 0.29240331],
    [-0.40341641, 0.18935338, 0.49288347, 0.12559316, -0.25548025, -0.21518377, -0.49443982, 0.42027353, 0.23472971, -0.44255133],
    [-0.12251824, -0.39090683, 0.4144166, 0.33532461, 0.40672453, 0.12083048, -0.47791662, -0.10031615, -0.4681782, -0.43967674],
    [0.30205014, 0.2009804, 0.24596844, -0.38371268, 0.30504946, 0.1322984, 0.06530228, -0.28564335, 0.40193607, 0.10658088],
    [0.38805038, -0.35276849, -0.44475011, -0.34922998, 0.42465848, -0.21405048, -0.23617881, 0.48033352, 0.25019747, 0.16710515]])

labels = np.matrix([-1,1,1,-1,-1])

T = 5

perceptron(feature_matrix, labels, T)
