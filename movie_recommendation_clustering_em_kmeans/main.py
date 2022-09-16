import math

import numpy as np
import kmeans
import common
import em
import test_utils

X = np.loadtxt("toy_data.txt")
test_utils.test_kmeans(X)

# retain the best
mixture, softcount = common.init(X, K=4, seed=0)
mixture, post, cost = kmeans.run(X, mixture, softcount)

test_utils.test_native_em(X)
test_utils.test_em(X, mixture, debug=True)

X = np.loadtxt("netflix_incomplete.txt")
for K in [1, 12]:
    max_ll = -math.inf
    for seed in range(5):
        print("Test EM for K=" + str(K), "seed = " + str(seed))
        mixture, softcount = common.init(X, K, seed)
        mixture, post, ll = em.run(X, mixture, softcount)
        if ll > max_ll:
            max_ll = ll
    print("max likelihood for K=" + str(K) + ", is: " + str(max_ll))
