#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:15:45 2020

@author: kroeker
"""
import itertools as it
import numpy as np

# %% Sobol idx for aPC
def sobol_idx_pc(pc_coefs, alphas, idx_set):
    var_pc = pc_coefs[1:] @ pc_coefs[1:]
    p_max, dim = alphas.shape
    srcs = list(range(dim))
    assert max(idx_set) < dim
    not_in_idx_set = list(set(srcs)-set(idx_set))
    sobol = 0
    for pidx in range(p_max):
        alpha = alphas[pidx, :]
        chk_in = alpha[idx_set].min() > 0
        if not not_in_idx_set:
            chk_out = 0
        else:
            chk_out = alpha[not_in_idx_set].max() == 0
        if chk_in and chk_out:
            sobol += pc_coefs[pidx]**2 / var_pc
    return sobol

def gen_idx_subsets(dim):
    items = list(range(dim))
    sub_idx = [frozenset(t) for length in range(1, dim+1)
               for t in it.combinations(items, length)]
    return set(sub_idx)

def sobol_tot_sen_pc(pc_coefs, alphas, src_idxs, idx_set):
    tot = 0
    for idx_it in src_idxs:
        if set(idx_set) <= idx_it:
            tot += sobol_idx_pc(pc_coefs, alphas, idx_set)
    return tot
