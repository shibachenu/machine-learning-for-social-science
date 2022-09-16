#question 1: linear perceptron
import numpy as np

from perceptron_algo import *

x = np.matrix([[0, 0], [2, 0], [3, 0], [0, 2], [2, 2], [5, 1], [5, 2], [2, 4], [4, 4], [5, 5]])
y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])

x_pad = np.insert(x, 0, [1], axis=1)

mistakes = np.array([1, 9, 10, 5, 9, 11, 0, 3, 1, 1])

y_mistakes = np.multiply(mistakes, y)

x_y = np.multiply(y_mistakes.reshape([10,1]), x_pad)

theta = [0, 0, 0]
theta_updated = theta + np.sum(x_y, axis=0)

#validate through perceptron

theta,error = perceptron_offset(x, y, 50)

print("theta is: "+str(theta), "error is: "+str(error))

from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
svmModel = clf.fit(x, y)

intercept = svmModel.intercept_
theta = svmModel.coef_

print("intercept: "+str(intercept), "coeff: "+str(theta))

margin = 1/np.linalg.norm(theta, 'fro')

print("margin is: "+str(margin))

from sklearn.metrics import hinge_loss

y_pred = svmModel.predict(x)
total_loss = hinge_loss(y, y_pred)

print("total hinge loss is: "+str(total_loss))

theta_half = theta/2
intercept_half = intercept/2

y_pred_half = np.matmul(x, theta_half.T) + intercept_half
total_loss = hinge_loss(y, y_pred_half)

print("total hinge loss after halving theta is: "+str(total_loss))

