import numpy as np


def perceptron(x, y, t, debug=True):
    n = x.shape[0]
    p = x.shape[1]
    theta = np.zeros(p).reshape([p, 1])
    error = 0
    for t in range(t):
        if (debug):
            print("Round: " + str(t))
        for i in range(n):
            if (debug):
                print("Iteration: " + str(i))
                print("Theta before update: " + str(theta))
            assignment = y[i] * (np.matmul(x[i], theta))
            if assignment <= 0:
                increment = y[i] * x[i].transpose()
                theta_new = np.add(theta, increment.reshape([p, 1]))
            if (debug):
                print("Theta after update: " + str(theta_new))
            if not np.array_equal(theta, theta_new):
                error = error + 1
            theta = theta_new
    return theta, error


def perceptron_offset(x, y, t):
    n = x.shape[0]
    x = np.append(x, np.ones(n).reshape(n, 1), 1)
    p = x.shape[1]
    theta = np.zeros(p).reshape([p, 1])
    error = 0
    for t in range(t):
        print("Round: " + str(t))
        for i in range(n):
            print("Iteration: " + str(i))
            print("Theta before update: " + str(theta))
            assignment = y[i] * (np.matmul(x[i], theta))
            theta_new = theta
            if assignment <= 0:
                increment = y[i] * x[i].transpose()
                theta_new = np.add(theta, increment)
            if not np.array_equal(theta, theta_new):
                error = error + 1
            print("Theta after update: " + str(theta_new))
            theta = theta_new
    return theta, error
