#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:07:47 2020

@author: kroeker
"""
from math import exp
import numpy as np
from scipy.stats import norm
import context
import aMRPC.stat as st

EPS = 1.0e-7

NUMBER_OF_MEASURMENTS = 20
STD = 0.2
NUMBER_OF_REALIZATIONS = 15




def gen_observations(eval_points, params):
    """
    simple example of synthetic physical model response for testing purpose

    Parameters
    ----------
    eval_points : np.array
        points in space or time.
    params : np.array
        Problem parameters.

    Returns
    -------
    np.array
        synthetic physical model evaluation on eval_points.

    """
    return (pow(params[0]*params[0] + params[1] -1, 2)
            + 0.1 * params[0] * exp(params[1]) - np.sqrt(0.5*eval_points)*2*params[1]
            + 1 + np.sin(5*eval_points))

def test_scalar_norm_likelihood():
    points = np.linspace(0, 1, NUMBER_OF_MEASURMENTS)
    synthetic_parameters = np.array([0, 0])
    observations = gen_observations(points, synthetic_parameters)
    cov_mx = STD*STD* np.eye(NUMBER_OF_MEASURMENTS)
    response_surface = observations + np.sin(points*30)/10
    likelihood = st.cmp_norm_likelihood_core(observations, response_surface, cov_mx)
    lh_cf = st.cmp_norm_likelihood_cf(STD, NUMBER_OF_MEASURMENTS)
    
    norm_likelihood = norm.pdf(observations, response_surface, STD)
    #print(likelihood, lh_cf, lh_cf*likelihood)
    assert abs(lh_cf*likelihood - norm_likelihood.prod()) < EPS
    
def test_vector_norm_likelihood():
    points = np.linspace(0, 1, NUMBER_OF_MEASURMENTS)
    synthetic_parameters = np.array([0, 0])
    cov_mx = STD*STD* np.eye(NUMBER_OF_MEASURMENTS)
    observations = gen_observations(points, synthetic_parameters)
    
    vec_observations = np.zeros([NUMBER_OF_MEASURMENTS, NUMBER_OF_REALIZATIONS])
    vec_response_surface = np.zeros([NUMBER_OF_MEASURMENTS, NUMBER_OF_REALIZATIONS])
    ref_likelihood = np.zeros(NUMBER_OF_REALIZATIONS)
    
    for i in range(NUMBER_OF_REALIZATIONS):
        vec_observations[:, i] = observations
        vec_response_surface[:, i] = observations + np.sin(points*30)/(10+i)
        ref_likelihood[i] = norm.pdf(observations, vec_response_surface[:, i], STD).prod()
    lh_cf = st.cmp_norm_likelihood_cf(STD, NUMBER_OF_MEASURMENTS)
    vec_lh = st.cmp_norm_likelihood_core(vec_observations, vec_response_surface, cov_mx)
    #print(vec_lh*lh_cf)
    #print(ref_likelihood)
    for i in range(NUMBER_OF_REALIZATIONS):
        assert abs(lh_cf*vec_lh[i] - ref_likelihood[i]) < EPS
