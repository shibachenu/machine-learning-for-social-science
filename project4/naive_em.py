"""Mixture model using EM"""
import math
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    import math
    n = X.shape[0]
    k = mixture.p.shape[0]

    post = np.zeros([n, k])
    weighted_log_post = 0.

    for i in range(n):
        x_i = X[i]
        cu_i = x_i != 0
        p_j_i_nom = np.zeros(k)
        for j in range(k):
            p_j = mixture.p[j]
            x_cu = x_i[cu_i]
            mu_cu = mixture.mu[j][cu_i]
            var = mixture.var[j]
            p_norm_j = ((2*math.pi)**len(cu_i)*var**len(cu_i))**(-1/2)*math.exp((-1/2)*(1/var)*np.matmul((x_cu-mu_cu).T, (x_cu-mu_cu)))
            p_j_i_nom[j] = p_j * p_norm_j

        p_j_i_nom_normalized = p_j_i_nom/np.sum(p_j_i_nom)
        post[i] = p_j_i_nom_normalized

        weighted_log_post_i = np.matmul(p_j_i_nom_normalized.T, np.log(p_j_i_nom/p_j_i_nom_normalized))
        weighted_log_post += weighted_log_post_i
    return post, weighted_log_post


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    d = X.shape[1]
    k = post.shape[1]

    mu = np.zeros([k, d])
    var = np.zeros(k)
    p = np.zeros(k)

    for j in range(k):
        p_j = post[:, j] #n x 1
        p_j_sum = np.sum(p_j)
        p[j] = np.mean(p_j)
        mu[j] = np.matmul(p_j.T, X)/p_j_sum #1xd
        var[j] = np.matmul(p_j.T, np.sum(np.multiply(X - mu[j], X - mu[j]), axis=1))/(d * p_j_sum)

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """

    delta = 1

    while delta > 10 ** (-6):
        post, old_likelihood = estep(X, mixture)
        mixture = mstep(X, post)
        new_post, new_likelihood = estep(X, mixture)
        delta = (new_likelihood - old_likelihood) / abs(new_likelihood)

    return mixture, new_post, new_likelihood
