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

def cmp_log_likelihood_cf(std, number_of_measurments):
    """
    Computes the logarithhm of the coefficient of the Gaussian likelihood function

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
    if np.all(covariance_matrix == np.diag(np.diagonal(covariance_matrix))):
        det = np.multiply.reduce(np.diag(covariance_matrix))
    else:
        det = np.multiply.reduce(np.diag(np.linalg.cholesky(covariance_matrix)))
    return 1/sqrt(pow(2*pi, dim)*det)

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
    if np.all(covariance_matrix == np.diag(np.diagonal(covariance_matrix))):
        logdet = np.log(np.diagonal(covariance_matrix)).sum()
        # print("cov_mx:", covariance_matrix.diagonal())
    else:
        # logdet = np.linalg.slogdet(covariance_matrix)[1]
        logdet = np.log(np.diag(np.linalg.cholesky(covariance_matrix))).sum()
    # return (-np.log(2*pi)*dim/2 - np.log(np.diag(np.linalg.cholesky(covariance_matrix))).sum())
    return -np.log(2*pi)*dim/2 - logdet

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
    deviation = observation - response_surface
    deviation_shape = deviation.shape
    cov_inv = np.linalg.pinv(covariance_matrix)
    if len(deviation_shape) == 1:
        return np.exp(-0.5*deviation.T @ cov_inv @ deviation)
    else:
        ret_array = np.zeros(deviation_shape[1])
        for i in range(deviation_shape[1]):
            ret_array[i] = np.exp(-0.5*deviation[:, i].T @
                                  cov_inv @ deviation[:, i])
        return ret_array

def cmp_log_likelihood_core(observation, response_surface, covariance_matrix):
    """
    Computes the core part of the Gaussian log-likelihood function with cov. matrix

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
    ret = np.inf
    deviation = observation - response_surface
    deviation_shape = deviation.shape
#    cov_inv = np.linalg.inv(covariance_matrix)
    cov_inv = np.linalg.pinv(covariance_matrix)
#    L = np.linalg.cholesky(covariance_matrix)
#    cov_inv = np.linalg.inv(L.T) @ np.linalg.inv(L)
    if len(deviation_shape) == 1:
        ret = -0.5*deviation.T @ cov_inv @ deviation
    else:
        ret_array = np.zeros(deviation_shape[1])
        for i in range(deviation_shape[1]):
            ret_array[i] = (-0.5*deviation[:, i].T
                            @ cov_inv @ deviation[:, i])
        ret = ret_array
    return ret

def cmp_log_likelihood_core_inv(observation, response_surface, cov_matrix_inv):
    """
    Computes the core part of the Gaussian log-likelihood function with cov. matrix

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
    ret = np.inf
    deviation = observation - response_surface
    deviation_shape = deviation.shape
#    cov_inv = np.linalg.inv(covariance_matrix)
    if len(deviation_shape) == 1:
        ret = -0.5*deviation.T @ cov_matrix_inv @ deviation
    else:
        ret_array = np.zeros(deviation_shape[1])
        for i in range(deviation_shape[1]):
            ret_array[i] = (-0.5*deviation[:, i].T
                            @ cov_matrix_inv @ deviation[:, i])
        ret = ret_array
    return ret

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
    return np.mean([lh_cf * cmp_norm_likelihood_core(observation,
                                                     response_surfaces[sample, :],
                                                     covariance_matrix)
                    for sample in range(sample_cnt)])
    #lhs = np.zeros(sample_cnt)
#    for sample in range(sample_cnt):
#        lhs[sample] = lh_cf * cmp_norm_likelihood_core(observation,
#                                                       response_surfaces[sample, :],
#                                                       covariance_matrix)
#    return lhs.mean()


def lbme_norm_response(observation, response_surfaces, covariance_matrix):
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
    mean_lh = np.mean([cmp_norm_likelihood_core(observation,
                                                response_surfaces[sample, :],
                                                covariance_matrix)
                       for sample in range(sample_cnt)])
    return llh_cf + np.log(mean_lh) if mean_lh > 0 else -np.inf


def d_kl_norm_prior_response(observation, response_surfaces, covariance_matrix,
                             **kwargs):
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
    eps = kwargs.get('eps', 0)
    n, m = response_surfaces.shape
    if m == len(observation):
        sample_cnt = n
    else:
        sample_cnt = m
        response_surfaces = response_surfaces.T
#    lhs = np.zeros(sample_cnt)
    llhs = np.array([cmp_log_likelihood_core(observation,
                                             response_surfaces[sample, :],
                                             covariance_matrix)
                     for sample in range(sample_cnt)])
#    llhs = np.zeros(sample_cnt)
#    for sample in range(sample_cnt):
##        lhs[sample] = cmp_norm_likelihood_core(observation,
##                                               response_surfaces[sample, :],
##                                               covariance_matrix)
#        llhs[sample] = cmp_log_likelihood_core(observation,
#                                               response_surfaces[sample, :],
#                                               covariance_matrix)
    #mask = np.exp(llhs) >= np.exp(llhs.max()) * np.random.uniform(0, 1, llhs.shape)
    mask = llhs - llhs.max() >= np.log(np.random.uniform(0, 1, llhs.shape))
    bme = np.exp(llhs).mean() + eps
#    return lh_cf*np.mean(llhs[mask]*lhs[mask])/bme - np.log(bme)
    return np.mean(llhs[mask]) - np.log(bme) if bme > 0 else np.nan

def entropy_norm_response(observation, response_surfaces, covariance_matrix, **kwargs):
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
        eps - error bad condition compensation.

    Returns
    -------
    float
        entropy value.

    """
    eps = kwargs.get('eps', 1e-15)
    n, m = response_surfaces.shape
    if m == len(observation):
        sample_cnt = n
    else:
        sample_cnt = m
        response_surfaces = response_surfaces.T

    llhs = np.array([cmp_log_likelihood_core(observation,
                                             response_surfaces[sample, :],
                                             covariance_matrix)
                     for sample in range(sample_cnt)])

    mask = llhs - llhs.max() >= np.log(np.random.uniform(0, 1, llhs.shape))
    bme = np.exp(llhs).mean() + eps
    rs_mean = response_surfaces.mean(axis=0)
    rs_cov = np.cov(response_surfaces, rowvar=False)
    #eps_v = rs_cov.diagonal() < eps
    if eps > 0:
        rs_cov += eps * np.eye(len(observation))
        #print("stat.entr: reg. cov_diag", rs_cov.diagonal())
        # eps_mx = np.diag(eps_v)
        # rs_cov[eps_mx] = eps

    rs_cov_cf = cmp_log_likelihood_cf_mv(rs_cov)
    f_gs = lambda sid : (cmp_log_likelihood_core(rs_mean,
                                                 response_surfaces[sid, :],
                                                 rs_cov)
                         + rs_cov_cf)
    llhs_gs_it = map(f_gs, np.arange(sample_cnt)[mask])
    #llhs_gs = np.fromiter(llhs_gs_it, dtype=float)
    return (np.log(bme) - np.mean(llhs[mask]) - np.mean(np.fromiter(llhs_gs_it, dtype=float))
            if bme > 0 else np.nan)
    # return np.log(bme) - np.mean(llhs[mask]) - np.mean(llhs_gs[mask]) if bme > 0 else np.nan
