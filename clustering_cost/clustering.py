import numpy as np

def distance_l1_norm(x1, x2):
    return np.linalg.norm(x1 - x2, ord=1)

def distance_l2_norm(x1, x2):
    return np.linalg.norm(x1 - x2)

def cluster_assignment(x, z_s, distance):
    closet_cluster_index = 0
    closet_distance = 100000
    for k in range(len(z_s)):
        cur_dist = distance(x, z_s[k])
        if(cur_dist<closet_distance):
            closet_distance = cur_dist
            closet_cluster_index = k
    return closet_cluster_index, closet_distance

def centroid(x):
    return np.median(x, axis=0)

def k_medroid(x, z0, distance_function):
    x_assignment = np.zeros(x.shape[0])

    for i in range(len(x)):
        cluster, distance = cluster_assignment(x[i], z0, distance_function)
        x_assignment[i] = cluster

    print("l1 norm: x assignments: " + str(x_assignment))

    x_s_cluster1 = x[x_assignment == 0]
    x_s_cluster2 = x[x_assignment == 1]

    return x_s_cluster1, x_s_cluster2


def k_means(x, z0, t, distance_function):
    for j in range(t):
        x_assignment = np.zeros(x.shape[0])
        for i in range(len(x)):
            cluster, distance = cluster_assignment(x[i], z0, distance_function)
            x_assignment[i] = cluster
        x_s_cluster1 = x[x_assignment == 0]
        z0[0] = centroid(x_s_cluster1)
        x_s_cluster2 = x[x_assignment == 1]
        z0[1] = centroid(x_s_cluster2)
    x_s_cluster1 = x[x_assignment == 0]
    x_s_cluster2 = x[x_assignment == 1]

    return z0, x_s_cluster1, x_s_cluster2


x = np.array([[0, -6],[4,4],[0,0],[-5,2]])
z0 = np.array([[-5,2], [0,-6]])


cluster1, cluster2 = k_medroid(x, z0, distance_l1_norm)
print("K medroid with l1 norm: cluster1: "+str(cluster1), "cluster2: "+str(cluster2))

cluster1, cluster2 = k_medroid(x, z0, distance_l2_norm)
print("K medroid with l2 norm: cluster1: "+str(cluster1), "cluster2: "+str(cluster2))

centroids, cluster1, cluster2 = k_means(x, z0, 1, distance_l1_norm)
print("K means with l1 norm: cluster1: "+str(cluster1), "cluster2: "+str(cluster2), "centroid: "+str(centroids))

