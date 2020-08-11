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
    return np.exp(-0.5*deviation.T @ np.linalg.inv(covariance_matrix) @ deviation)
