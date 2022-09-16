# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print("Hi, {0}".format(name))  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#Hinge loss vs Empirical Risk
import numpy as np
x = np.matrix([[1, 0, 1],[1, 1, 1],[1, 1, -1], [-1, 1, 1]])
y = np.array([2, 2.7, -0.7, 2])

theta = np.array([0, 1, 2])

def hinge_loss(x, y, theta):
    z = y - np.matmul(x, theta)
    def loss(z):
        return max(0, 1-z)
    vloss = np.vectorize(loss)
    loss = vloss(z)
    return loss.mean()

def risk_loss(x, y, theta):
    z = y - np.matmul(x, theta)
    risk = np.power(z, 2)/2
    return risk.mean()

hinge_loss = hinge_loss(x, y, theta)
print("Hinge loss is: "+str(hinge_loss))

risk_loss = risk_loss(x, y, theta)
print("Risk loss is: "+str(risk_loss))

import math

def polyterm_count(n, power):
    term_count = 0
    for d in range(1, power+1):
        current_term_count = math.comb(n+d-1, d)
        term_count += current_term_count
    print("Term count for dimension: "+str(n)+", power: "+str(power)+", is: "+str(term_count))

polyterm_count(150, 3)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(color_codes=True)

def plot_poly():
    x = np.arange(0.1, 50, step=0.5)
    sns.regplot(x, x**2 + x**3 + x**4, order=4)

plot_poly()