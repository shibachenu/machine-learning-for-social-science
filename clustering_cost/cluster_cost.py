import math
import numpy as np

def cost_single_cluster(x, z):
    """
    Cost function for one cluster based on Euclidean norm
    :param x: all data points in the cluster, each row of x is one multi-dim data point
    :param z: rep data point of the cluster
    :return: total cost of the cluster
    """
    return np.linalg.norm(x-z)**2

def cost_all_clusters(clusters, reps):
    """
    Cost function for the whole partitioning
    :param clusters: list of clusters, each cluster data set is matrix
    :param reps: rep for each cluster
    :return: total cost of the partitioning
    """
    sum = 0
    for i in range(reps.shape[0]):
        cluster = clusters[i]
        rep = reps[i]
        cost = cost_single_cluster(cluster, rep)
        sum += cost
    return sum


# testing the cost functions
x1 = np.array([[-1, 2], [-2, 1], [-1, 0]])
z1 = np.array([-1, 1])

cost1 = cost_single_cluster(x1, z1)
print("cost1 is: " + str(cost1))

x2 = np.array([[2, 1], [3, 2]])
z2 = np.array([2, 2])
cost2 = cost_single_cluster(x2, z2)
print("cost2 is : " + str(cost2))
