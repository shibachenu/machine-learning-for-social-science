"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
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
        f_u = np.zeros(k)
        for j in range(k):
            p_j = mixture.p[j]
            x_cu = x_i[cu_i]
            mu_cu = mixture.mu[j][cu_i]
            var_j = mixture.var[j]

            f_u_j = math.log(p_j + 1e-16) - 0.5 * len(x_cu) * (math.log(2 * math.pi) + math.log(var_j)) - (
                        1 / (2 * var_j)) * np.matmul((x_cu - mu_cu).T, (x_cu - mu_cu))

            f_u[j] = f_u_j
        #logsum of f_u exp across j
        f_u_max = np.max(f_u)
        f_u_logsum = f_u_max + np.log(np.sum(np.exp(f_u - f_u_max)))

        l_i = f_u - f_u_logsum
        post[i] = np.exp(l_i)

        weighted_log_post_i = np.matmul(post[i].T, f_u - l_i)
        weighted_log_post += weighted_log_post_i
    return post, weighted_log_post



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    d = X.shape[1]
    k = post.shape[1]
    cu = X != 0

    mu_hat = np.zeros([k, d])
    var_hat = np.zeros(k)
    p_hat  = np.zeros(k)

    for j in range(k):
        p_j = post[:, j]  # n x 1

        p_hat[j] = np.mean(p_j)
        cu_j = cu[j]

        p_j_cu = p_j[cu_j]
        p_j_cu_sum = np.sum(p_j_cu)

        X_cu_j = X[cu_j]
        mu_hat[j] = np.matmul(p_j_cu.T, X_cu_j) / p_j_cu_sum #cu x 1

        var_hat[j] = np.matmul(p_j_cu.T, np.sum(np.multiply(X_cu_j - mu_hat[j], X_cu_j - mu_hat[j]), axis=1)) / (
                    X_cu_j.shape[1] * np.sum(p_j))
        var_hat[j] = max(min_variance, var_hat[j])

    return GaussianMixture(mu_hat, var_hat, p_hat)


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
        mixture = mstep(X, post, mixture)
        new_post, new_likelihood = estep(X, mixture)
        delta = (new_likelihood - old_likelihood) / abs(new_likelihood)

    return mixture, new_post, new_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
