import numpy as np
import math

def weighted_average(X, w, w0):
    '''
    :param X: n x d, inputs matrix
    :param w: 1 x d, theta, weights for each coordinates
    :param w0: intercept
    :return: z: weighted average
    '''
    z = np.matmul(X, w.T) + w0
    return z

def activation_relu(z):
    '''

    :param z: weighted average
    :return: ReLU transformaiton of z
    '''
    return np.maximum(z, 0)


def activation_tanh(z):
    '''

    :param z: weighted average
    :return: hyperbolic tangent function activation of z
    '''
    z_activated = 1 - 2 / (np.exp(2 * z) + 1)
    return z_activated


def activation_linear(z):
    z_activated = 2*z - 3
    return z_activated


def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig


def softmax(X):
    X_exp = np.exp(X)
    return X_exp/np.sum(X_exp, axis=1)
