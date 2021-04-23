#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:21:16 2020

@author: kroeker

Provides statistic tools
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

def cmp_norm_likelihood_cf_mv(covariance_matrix):
    dim = covariance_matrix.shape[0]
    return 1/sqrt(pow(2*pi, dim)*np.linalg.det(covariance_matrix))

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
    if len(deviation_shape) == 1:
        return np.exp(-0.5*deviation.T @ np.linalg.inv(covariance_matrix) @ deviation)
    else:
        ret_array = np.zeros(deviation_shape[1])
        for i in range(deviation_shape[1]):
            ret_array[i] = np.exp(-0.5*deviation[:, i].T @
                                  np.linalg.inv(covariance_matrix) @ deviation[:, i])
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
    deviation = observation - response_surface
    deviation_shape = deviation.shape
    if len(deviation_shape) == 1:
        return -0.5*deviation.T @ np.linalg.inv(covariance_matrix) @ deviation
    else:
        ret_array = np.zeros(deviation_shape[1])
        for i in range(deviation_shape[1]):
            ret_array[i] = (-0.5*deviation[:, i].T
                            @ np.linalg.inv(covariance_matrix) @ deviation[:, i])
        return ret_array

def cmp_norm_bme_response(observation, response_surfaces, covariance_matrix):
    dim = len(observation)
    n, m = response_surfaces.shape
    if m == dim:
        sample_cnt = n
    else:
        sample_cnt = m
        response_surfaces = response_surfaces.T
    lhs = np.zeros(sample_cnt)
    lh_cf = cmp_norm_likelihood_cf_mv(covariance_matrix)
    for sample in range(sample_cnt):
        lhs[sample] = lh_cf * cmp_norm_likelihood_core(observation,
                                                       response_surfaces[sample, :],
                                                       covariance_matrix)
    return lhs.mean()

def d_kl_prior_response(observation, response_surfaces, covariance_matrix):
    lh_cf = cmp_norm_likelihood_cf_mv(covariance_matrix)
    llh_cf = np.log(lh_cf)
    dim = len(observation)
    n, m = response_surfaces.shape
    if m == dim:
        sample_cnt = n
    else:
        sample_cnt = m
        response_surfaces = response_surfaces.T
    lhs = np.zeros(sample_cnt)
    llhs = np.zeros(sample_cnt)
    for sample in range(sample_cnt):
        lhs[sample] = cmp_norm_likelihood_core(observation,
                                               response_surfaces[sample, :],
                                               covariance_matrix)
        llhs[sample] = cmp_log_likelihood_core(observation,
                                               response_surfaces[sample, :],
                                               covariance_matrix) + llh_cf
    mask = lhs >= lhs.max() * np.random.uniform(0, 1, lhs.shape)
    bme = lh_cf*lhs.mean()
#    return lh_cf*np.mean(llhs[mask]*lhs[mask])/bme - np.log(bme)
    return np.mean(llhs[mask]) - np.log(bme)