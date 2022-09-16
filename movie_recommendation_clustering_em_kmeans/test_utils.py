import math
import kmeans
import common
import naive_em
import em
import numpy as np

def test_kmeans(X, debug=False, plot=False):
    # K means first to initiliaze parameters for EM algo
    for K in range(1, 5):
        if debug:
            print("Running k-means for K =" + str(K))
        min_cost = math.inf
        for seed in range(0, 1):
            mixture, softcount = common.init(X, K, seed)
            mixture, post, cost = kmeans.run(X, mixture, softcount)
            if plot:
                common.plot(X, mixture, post, "with K="+str(K)+", seed="+str(seed))
            if cost < min_cost:
                min_cost = cost
        if debug:
            print("min cost for K=" + str(K) + " is " + str(min_cost))


def test_native_em(X, debug=False, plot=False):
    for K in range(1, 5):
        if debug:
            print("Running EM for K =" + str(K))
        min_cost = math.inf
        max_bic = -math.inf
        for seed in range(0, 1):
            mixture, softcount = common.init(X, K, seed)
            mixture, post, cost = naive_em.run(X, mixture, softcount)
            if plot:
                common.plot(X, mixture, post, "with K="+str(K)+", seed="+str(seed))
            if cost < min_cost:
                min_cost = cost
            bic = common.bic(X, mixture, cost)
            if bic > max_bic:
                max_bic = bic
        if debug:
            print("min cost for K=" + str(K) + " is " + str(min_cost), "max BIC is : "+str(max_bic))


def test_em(X, mixture, debug=False, plot=False):
    post, l = em.estep(X, mixture)
    if debug:
        print("EM likelihood: " + str(l))

    post, l = naive_em.estep(X, mixture)
    if debug:
        print("Naive EM likelihood: "+str(l))

