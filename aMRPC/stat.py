#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:21:16 2020

@author: kroeker
https://orcid.org/0000-0003-0360-5307

Provides statistic and information theoretic tools.

"""
from math import sqrt, pi
import numpy as np
try:
    from numba import jit_module, jit  # jit, njit  # jit_module
    NJM = True
except ImportError:
    NJM = False
    pass


# @njit()
def cmp_norm_likelihood_cf(std, number_of_measurments):
    """
    Computes the coefficient of the Gaussian likelihood function

    Parameters
    ----------
    std : float
        standard deviation.
    number_of_measurments : int
        number of observations.

    Returns
    -------
    float
        likelihood coefficient.

    """
    return 1/pow(sqrt(2*pi)*std, number_of_measurments)


# @njit
def cmp_log_likelihood_cf(std, number_of_measurments):
    """
    Computes the logarithhm of the coefficient of the Gaussian likelihood
    function

    Parameters
    ----------
    std : float
        standard deviation.
    number_of_measurments : int
        number of observations.

    Returns
    -------
    float
        likelihood coefficient.

    """
    return -(np.log(2*pi)/2 + np.log(std)) * number_of_measurments


# @jit(nopython=True)
def cmp_norm_likelihood_cf_mv(covariance_matrix):
    """
    Computes normalizing coefficient for Gaussian likelihood function

    Parameters
    ----------
    covariance_matrix : np.array
        Covariance matrix.

    Returns
    -------
    float
        normalizing coefficient.

    """
    dim = covariance_matrix.shape[0]
    if np.all(covariance_matrix == np.diag(np.diag(covariance_matrix))):
        det = np.prod(np.diag(covariance_matrix))
    else:
        det = np.prod(np.diag(np.linalg.cholesky(covariance_matrix)))
    return 1/sqrt(pow(2*pi, dim)*det)


# @njit  # (nopython=True)
def cmp_log_likelihood_cf_mv(covariance_matrix):
    """
    computes logarithm normalizing coefficient for Gaussian likelihood function

    Parameters
    ----------
    covariance_matrix : np.array
        Covariance matrix.

    Returns
    -------
    float
        log(normalizing coefficient).

    """
    dim = covariance_matrix.shape[0]
    if np.all(covariance_matrix == np.diag(np.diag(covariance_matrix))):
        logdet = np.log(np.diag(covariance_matrix)).sum()/2
        # print("cov_mx:", covariance_matrix.diagonal())
    else:
        # logdet = np.linalg.slogdet(covariance_matrix)[1]
        logdet = np.log(np.diag(np.linalg.cholesky(covariance_matrix))).sum()
    # return (-np.log(2*pi)*dim/2 - np.log(np.diag(np.linalg.cholesky(covariance_matrix))).sum())
    return -np.log(2*pi)*dim/2 - logdet


# @njit
def cmp_norm_likelihood_core(observation, response_surface, covariance_matrix):
    """
    Computes the core part of the Gaussian likelihood function with cov. matrix

    Parameters
    ----------
    observation : numpy array
        Array of observations.
    response_surface : numpy array
        Array of evaluations of response surface.
    covariance_matrix : numpy array
        Covariance matrix.

    Returns
    -------
    numpy array
        Likelihood for each observation / realization.

    """
    cov_inv = np.linalg.pinv(covariance_matrix)
    if response_surface.ndim == 1:
        deviation = observation - response_surface
        return np.exp(-0.5*deviation.T @ cov_inv @ deviation)
    else:
        deviation = observation.reshape((-1, 1)) - response_surface
        deviation_shape = deviation.shape
        ret_array = np.zeros(deviation_shape[1], dtype=np.float64)
        for i in range(deviation_shape[1]):
            devi = np.ascontiguousarray(deviation[:, i])
            ret_array[i] = np.exp(-0.5*devi.T @ cov_inv @ devi)
        return ret_array


# @njit
def cmp_norm_likelihood_core_inv(observation, response_surface, cov_inv):
    """
    Computes the core part of the Gaussian likelihood function with cov. matrix

    Parameters
    ----------
    observation : numpy array
        Array of observations.
    response_surface : numpy array
        Array of evaluations of response surface.
    covariance_matrix : numpy array
        Covariance matrix.

    Returns
    -------
    numpy array
        Likelihood for each observation / realization.

    """
    if response_surface.ndim == 1:
        deviation = observation - response_surface
        return np.exp(-0.5*deviation.T @ cov_inv @ deviation)
    else:
        n_o, n_s = response_surface.shape
        deviation = observation - response_surface
        # deviation_shape = deviation.shape
        ret_array = np.zeros(n_s, dtype=np.float64)
        for i in range(n_s):
            devi = np.ascontiguousarray(deviation[:, i])
            ret_array[i] = np.exp(-0.5*devi.T @ cov_inv @ devi)
        return ret_array


# @njit
def cmp_log_likelihood_core(observation, response_surface, covariance_matrix):
    """
    Computes the core part of the Gaussian log-likelihood function with
    cov. matrix

    Parameters
    ----------
    observation : numpy array
        Array of observations.
    response_surface : numpy array
        Array of evaluations of response surface.
    covariance_matrix : numpy array
        Covariance matrix.

    Returns
    -------
    numpy array
        Log-likelihood for each observation / realization.

    """
    # ret = np.inf
    #    cov_inv = np.linalg.inv(covariance_matrix)
    cov_inv = np.linalg.pinv(covariance_matrix)

#    L = np.linalg.cholesky(covariance_matrix)
#    cov_inv = np.linalg.inv(L.T) @ np.linalg.inv(L)
    if response_surface.ndim == 1:
        deviation = observation - response_surface
        return np.float64(-0.5*deviation.T @ cov_inv @ deviation)
    else:
        deviation = observation.reshape((-1, 1)) - response_surface
        deviation_shape = deviation.shape
        ret_array = np.zeros(deviation_shape[1], dtype=np.float64)
        for i in range(deviation_shape[1]):
            devi = np.ascontiguousarray(deviation[:, i])
            ret_array[i] = np.float64(-0.5*devi.T @ cov_inv @ devi)
        return ret_array


# @njit  # (nopython=True)
def cmp_log_likelihood_core_inv(observation, response_surface, cov_matrix_inv):
    """
    Computes the core part of the Gaussian log-likelihood function with
    cov. matrix

    Parameters
    ----------
    observation : numpy array
        Array of observations.
    response_surface : numpy array
        Array of evaluations of response surface.
    cov_matrix_inv : numpy array
        inverse of the covariance matrix.

    Returns
    -------
    numpy array
        Log-likelihood for each observation / realization.

    """
    # ret = np.inf
    if response_surface.ndim == 1:
        deviation = observation - response_surface
        ret = -0.5*deviation.T @ cov_matrix_inv @ deviation
    else:
        deviation = observation.reshape((-1, 1)) - response_surface
        deviation_shape = deviation.shape
        ret_array = np.zeros(deviation_shape[1], dtype=np.float64)
        for i in range(deviation_shape[1]):
            devi = np.ascontiguousarray(deviation[:, i])
            ret_array[i] = (-0.5*devi.T @ cov_matrix_inv @ devi)
        ret = ret_array
    return ret


# @njit
def cmp_log_likelihood_core_sinv(observation, response_surface, cov_matrix_inv):
    """
    Computes the core part of the Gaussian log-likelihood function with
    cov. matrix

    Parameters
    ----------
    observation : numpy array
        Array of observations.
    response_surface : numpy array
        Array of evaluations of response surface.
    cov_matrix_inv : numpy array
        inverse of the covariance matrix.

    Returns
    -------
    numpy array
        Log-likelihood for each observation / realization.

    """
    # ret = np.inf
    deviation = np.ascontiguousarray(observation - response_surface)

    return -0.5*deviation.T @ cov_matrix_inv @ deviation


# @njit
def bme_norm_response(observation, response_surfaces, covariance_matrix):
    """
    Computes BME (Bayesian Model Evidence)

    Parameters
    ----------
    observation : np.array
        observation trajectory.
    response_surfaces : np.array
        surrogate / model response.
    covariance_matrix : np.array
        covariance matrix.

    Returns
    -------
    float
        BME.

    """
    n, m = response_surfaces.shape[0:2]
    if m == len(observation):
        sample_cnt = n
    else:
        sample_cnt = m
        response_surfaces = response_surfaces.T

    lh_cf = cmp_norm_likelihood_cf_mv(covariance_matrix)
    covariance_matrix_inv = np.linalg.pinv(covariance_matrix)
    return np.mean(np.array([lh_cf * cmp_norm_likelihood_core_inv(observation,
                                                                  response_surfaces[sample, :],
                                                                  covariance_matrix_inv)
                             for sample in range(sample_cnt)], dtype=np.float64))
    # lhs = np.zeros(sample_cnt)
#    for sample in range(sample_cnt):
#        lhs[sample] = lh_cf * cmp_norm_likelihood_core(observation,
#                                                       response_surfaces[sample, :],
#                                                       covariance_matrix)
#    return lhs.mean()


# @njit
def lbme_norm_response(observation, response_surfaces, covariance_matrix,
                       eps=0):
    """
    Computes log(BME) (Bayesian Model Evidence)

    Parameters
    ----------
    observation : np.array
        observation trajectory.
    response_surfaces : np.array
        surrogate / model response.
    covariance_matrix : np.array
        covariance matrix.

    Returns
    -------
    float
        BME.

    """
    n, m = response_surfaces.shape[0:2]
    if m == len(observation):
        sample_cnt = n
    else:
        sample_cnt = m
        response_surfaces = response_surfaces.T

    llh_cf = cmp_log_likelihood_cf_mv(covariance_matrix)
    covariance_matrix_inv = np.linalg.pinv(covariance_matrix)
    mean_lh = np.mean(np.array([cmp_norm_likelihood_core_inv(observation,
                                                             response_surfaces[sample, :],
                                                             covariance_matrix_inv) + eps
                                for sample in range(sample_cnt)], dtype=np.float64))
    return llh_cf + np.log(mean_lh) if mean_lh > 0 else -np.inf


def d_kl_norm_prior_response(observation, response_surfaces, covariance_matrix,
                             eps=0):
    """
    computes Kullback-Leibler divergence

    Parameters
    ----------
    observation : np.array
        observation.
    response_surfaces : np.array
        surrogate / model response.
    covariance_matrix : np.array
        covariance matrix.
    **kwargs : dict
        eps - error bad condition compensation.

    Returns
    -------
    float.

    """
    # eps = kwargs.get('eps', 0)
    n, m = response_surfaces.shape
    if m == len(observation):
        sample_cnt = n
    else:
        sample_cnt = m
        response_surfaces = response_surfaces.T
#    lhs = np.zeros(sample_cnt)
    covariance_matrix_inv = np.linalg.pinv(covariance_matrix)
    llhs = np.array([cmp_log_likelihood_core_sinv(observation,
                                                  response_surfaces[sample, :],
                                                  covariance_matrix_inv)
                     for sample in range(sample_cnt)])
#    llhs = np.zeros(sample_cnt)
#    for sample in range(sample_cnt):
##        lhs[sample] = cmp_norm_likelihood_core(observation,
##                                               response_surfaces[sample, :],
##                                               covariance_matrix)
#        llhs[sample] = cmp_log_likelihood_core(observation,
#                                               response_surfaces[sample, :],
#                                               covariance_matrix)
    # mask = np.exp(llhs) >= np.exp(llhs.max()) * np.random.uniform(0, 1, llhs.shape)
    mask = llhs - llhs.max() >= np.log(np.random.uniform(0, 1, llhs.shape))
    bme = np.exp(llhs).mean() + eps
#    return lh_cf*np.mean(llhs[mask]*lhs[mask])/bme - np.log(bme)
    return np.mean(llhs[mask]) - np.log(bme) if bme > 0 else np.nan


# if NJM:
#     jit_module(nopython=True, error_model="numpy")


# @jit
def entropy_norm_response_j(observation, response_surfaces, covariance_matrix,
                            eps, eps_bme, cov_diag):
    """
    Computes entropy

    Parameters
    ----------
    observation : np.array
        observation.
    response_surfaces : np.array
        surrogate / model response.
    covariance_matrix : np.array
        covariance matrix.
    **kwargs : dict
        eps : error bad condition of cov-mx compensation, def: eps=1e-15.
        eps_bme : lowest bound for bme. default: eps_bme=1.0e-300:
        cov_diag: bool, uses only diag of the est. cov-mx. default: False.

    Returns
    -------
    float
        entropy value.

    """
    # eps = kwargs.get('eps', 1e-15)
    # eps_bme = kwargs.get('eps_bme', 1.0e-300)
    # cov_diag = kwargs.get('cov_diag', False)
    n, m = response_surfaces.shape
    if m == len(observation):
        sample_cnt = n
        measurs = m
        response_surfaces = np.ascontiguousarray(response_surfaces)
    else:
        measurs = n
        sample_cnt = m
        response_surfaces = np.ascontiguousarray(response_surfaces.T)

    covariance_matrix_inv = np.linalg.pinv(covariance_matrix)
    llhs = np.array([cmp_log_likelihood_core_sinv(observation,
                                                  response_surfaces[sample, :],
                                                  covariance_matrix_inv)
                     for sample in range(sample_cnt)])

    mask = llhs - llhs.max() >= np.log(np.random.uniform(0, 1, llhs.shape))
    bme = np.exp(llhs).mean() + eps_bme
    rs_mean = np.array([response_surfaces[:, i].mean()
                        for i in range(measurs)])

    rs_cov = np.cov(response_surfaces, rowvar=False)
    if cov_diag:
        rs_cov = np.diag(np.diag(rs_cov))
    # eps_v = rs_cov.diagonal() < eps
    if eps > 0.0:
        rs_cov += eps * np.eye(measurs)
        # print("stat.entr: reg. cov_diag", rs_cov.diagonal())
        # eps_mx = np.diag(eps_v)
        # rs_cov[eps_mx] = eps

    rs_cov_cf = cmp_log_likelihood_cf_mv(rs_cov)
    # try:
    rs_cov_inv = np.linalg.pinv(covariance_matrix)
    # except:
    #     L = np.linalg.cholesky(covariance_matrix)
    #     L_inv = np.ascontiguousarray(np.linalg.pinv(L))
    #     rs_cov_inv = np.ascontiguousarray(L_inv.T @ L_inv)

    llhs_gs = cmp_log_likelihood_core_inv(rs_mean,
                                          response_surfaces[mask, :].T,
                                          rs_cov_inv)

    return (np.log(bme) - np.mean(llhs[mask]) - np.mean(llhs_gs)
            - rs_cov_cf if bme > 0 else np.nan)
    # return np.log(bme) - np.mean(llhs[mask]) - np.mean(llhs_gs[mask]) if bme > 0 else np.nan


def entropy_prior_response(observation, response_surfaces, covariance_matrix,
                            pr_dens, eps):
    """
    Computes entropy

    Parameters
    ----------
    observation : np.array
        observation.
    response_surfaces : np.array
        surrogate / model response.
    covariance_matrix : np.array
        covariance matrix.
    pr_dens: np.array
        prior probability density values of the samples
   eps : lowest bound for bme. default: eps_bme=1.0e-300:
 

    Returns
    -------
    float
        entropy value.

    """
    n, m = response_surfaces.shape
    if m == len(observation):
        sample_cnt = n
        measurs = m
        response_surfaces = np.ascontiguousarray(response_surfaces)
    else:
        measurs = n
        sample_cnt = m
        response_surfaces = np.ascontiguousarray(response_surfaces.T)

    covariance_matrix_inv = np.linalg.pinv(covariance_matrix)
    llhs = np.array([cmp_log_likelihood_core_sinv(observation,
                                                  response_surfaces[sample, :],
                                                  covariance_matrix_inv)
                     for sample in range(sample_cnt)])

    mask = llhs - llhs.max() >= np.log(np.random.uniform(0, 1, llhs.shape))
    bme = np.exp(llhs).mean() + eps

    return (np.log(bme) - np.mean(llhs[mask]) - np.mean(np.log(pr_dens[mask]))
            if bme > 0 else np.nan)


if NJM:
    jit_module(nopython=True, error_model="numpy", nogil=True)


def entropy_norm_response(observation, response_surfaces, covariance_matrix,
                          **kwargs):
    """
    Computes entropy

    Parameters
    ----------
    observation : np.array
        observation.
    response_surfaces : np.array
        surrogate / model response.
    covariance_matrix : np.array
        covariance matrix.
    **kwargs : dict
        eps : error bad condition of cov-mx compensation, def: eps=1e-15.
        eps_bme : lowest bound for bme. default: eps_bme=1.0e-300:
        cov_diag: bool, uses only diag of the est. cov-mx. default: False.

    Returns
    -------
    float
        entropy value.

    """
    eps = np.float64(kwargs.get('eps', 1e-15))
    eps_bme = np.float64(kwargs.get('eps_bme', 1.0e-300))
    cov_diag = np.bool8(kwargs.get('cov_diag', False))

    return entropy_norm_response_j(observation,
                                   response_surfaces,
                                   covariance_matrix, eps, eps_bme,
                                   cov_diag)

    # n, m = response_surfaces.shape
    # if m == len(observation):
    #     sample_cnt = n
    # else:
    #     sample_cnt = m
    #     response_surfaces = response_surfaces.T

    # covariance_matrix_inv = np.linalg.pinv(covariance_matrix)
    # llhs = np.array([cmp_log_likelihood_core_sinv(observation,
    #                                               response_surfaces[sample, :],
    #                                               covariance_matrix_inv)
    #                   for sample in range(sample_cnt)])

    # mask = llhs - llhs.max() >= np.log(np.random.uniform(0, 1, llhs.shape))
    # bme = np.exp(llhs).mean() + eps_bme
    # rs_mean = response_surfaces.mean(axis=0)
    # rs_cov = np.cov(response_surfaces, rowvar=False)
    # if cov_diag:
    #     rs_cov = np.diag(rs_cov.diagonal())

    # if eps > 0:
    #     rs_cov += eps * np.eye(len(observation))

    # rs_cov_cf = cmp_log_likelihood_cf_mv(rs_cov)
    # rs_cov_inv = np.linalg.pinv(covariance_matrix)

    # f_gs = lambda sid: (cmp_log_likelihood_core_sinv(rs_mean,
    #                                                   response_surfaces[sid, :],
    #                                                   rs_cov_inv))

    # llhs_gs_it = map(f_gs, np.arange(sample_cnt)[mask])
    # return (np.log(bme) - np.mean(llhs[mask]) - np.mean(np.fromiter(llhs_gs_it,
    #                                                                 dtype=np.float64))
    #         - rs_cov_cf if bme > 0 else np.nan)
