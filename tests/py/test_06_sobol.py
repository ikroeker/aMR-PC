#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:28:39 2020

@author: kroeker

We use Ishigami function for testing, see
B. Sudret, Global sensitivity analysis using polynomial chaos expansion
doi:10.1016/j.ress.2007.04.002
"""
import math
import numpy as np
import context
import aMRPC.sobol as sob

# Problem params
SAMPLE_CNT = 1000
P = 8
# Ishigami function:
A = 7
B = 0.1
ISH_FCT = lambda X: np.sin(X[:, 0]) + A*np.sin(X[:, 1])**2 + B*(X[:, 2]**4)*np.sin(X[:, 0])


def gen_rv(sample_cnt):
    X = np.random.uniform(-math.pi, math.pi, (sample_cnt, 3))
    Y = ISH_FCT(X)
    return X, Y

def ishigami_exact_sensitivity(a, b):
    D = a**2 / 8 + b*math.pi**4 / 5 + b**2 * math.pi**8 /18 + 0.5
    DS = {}
    DS[frozenset([1])] = b*(math.pi**4) / 5 + (b**2) * (math.pi**8) / 50 +0.5
    DS[frozenset([2])] = (a**2) / 8
    DS[frozenset([3])] = 0
    DS[frozenset([1, 2])] = 0
    DS[frozenset([2, 3])] = 0
    DS[frozenset([1, 3])] = 8*b**2 * math.pi**8 / 225
    DS[frozenset([1, 2, 3])] = 0
    sob_cfs = {}
    for idx, var_cf in DS.items():
        sob_cfs[idx] = var_cf / D
    return D, sob_cfs

def test_sobol():
    X, Y = gen_rv(SAMPLE_CNT)
    y_var, sob_cfs = ishigami_exact_sensitivity(A, B)
    