import numpy as np
import scipy.stats as stats


x1 = np.array([[-1.2, -0.8], [-1, -1.2], [-0.8, -1]])
x2 = np.array([[1.2, 0.8], [1, 1.2], [0.8, 1]])

np.mean(x1, axis=0)

mu1 = -3
mu2 = 2
var1 = 4
var2 = 4

p1 = 0.5
p2 = 0.5

x1 = 0.2
x2 = -0.9
x3 = -1
x4 = 1.2
x5 = 1.8

import math

def poster_clusters_prob(x, p1, p2, mu1, mu2, var1, var2):
    sd1 = math.sqrt(var1)
    sd2 = math.sqrt(var2)
    p1_1 = p1 * stats.norm(mu1, sd1).pdf(x) / (p1 * stats.norm(mu1, sd1).pdf(x) + p2 * stats.norm(mu2, sd2).pdf(x))
    p1_2 = 1 - p1_1
    return p1_1, p1_2

p11, p12 = poster_clusters_prob(x1, p1, p2, mu1, mu2, var1, var2)
p21, p22 = poster_clusters_prob(x2, p1, p2, mu1, mu2, var1, var2)
p31, p32 = poster_clusters_prob(x3, p1, p2, mu1, mu2, var1, var2)
p41, p42 = poster_clusters_prob(x4, p1, p2, mu1, mu2, var1, var2)
p51, p52 = poster_clusters_prob(x5, p1, p2, mu1, mu2, var1, var2)

print("p11, p12, p21, p22, p31, p32, p41, p42, p51, p52")

n1 = p11 + p21 + p31 + p41 +p51
n2 = p12 + p22 + p32 + p42 + p52
p1_updated = n1/(n1+n2)

mu1_updated = (p11*x1 + p21*x2 + p31*x3 + p41*x4 + p51*x5)/n1
var1_updated = (p11*((x1-mu1_updated)**2) + p21*((x2-mu1_updated)**2) + p31*((x3-mu1_updated)**2) + p41*((x4-mu1_updated)**2) + p51*((x5-mu1_updated)**2))/n1

print("mean, variance")

d = "ABABBCABAABCAC"
A_count = d.count("A")
B_count = d.count("B")
C_count = d.count("C")

prob_A = A_count/len(d)
prob_B = B_count/len(d)
prob_C = C_count/len(d)


prob_ABC = prob_A * prob_B * prob_C
prob_BBB = prob_B**3
prob_ABB = prob_A*prob_B**2
prob_AAC = prob_A**2*prob_C

print("probs")