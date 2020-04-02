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
import pandas as pd
import context
import aMRPC.sobol as sob
import aMRPC.datatools as dt
import aMRPC.utils as u

# Problem params
TOL = 1e-3 # error tolerance
SAMPLE_CNT = 1000000
P = 9

METHOD = 1 # gen roots, weights, polynomials method
SRCS = [0, 1, 2]
# params
DIM = len(SRCS)
ALPHAS = u.gen_multi_idx(P, DIM)  #powers of multivar. polynomials
P_MR = ALPHAS.shape[0] # number of coefficients
# Ishigami function:
A = 7
B = 0.1

ISH_FCT = lambda X: A*np.sin(X[:, 1])**2 + (1+B*np.power(X[:, 2], 4))*np.sin(X[:, 0])

def ishigami_exact_sensitivity(a, b):
    ival = 0
    D = (a**2) / 8 + b*(math.pi**4) / 5 + (b**2) * (math.pi**8) /18 + 0.5
    DS = {}
    DS[frozenset([ival])] = b*(math.pi**4) / 5 + (b**2) * (math.pi**8) / 50 +0.5
    DS[frozenset([ival+1])] = (a**2) / 8
    DS[frozenset([ival+2])] = 0
    DS[frozenset([ival, ival+1])] = 0
    DS[frozenset([ival+1, ival+2])] = 0
    DS[frozenset([ival, ival+2])] = 8*b**2 * math.pi**8 / 225
    DS[frozenset([ival, ival+1, ival+2])] = 0
    sob_cfs = {}
    for cf_idx, var_cf in DS.items():
        sob_cfs[cf_idx] = var_cf / D
    return D, sob_cfs


def gen_rv(sample_cnt):
    x = np.random.uniform(-math.pi, math.pi, (sample_cnt, 3))
    y = ISH_FCT(x)
    return x, y


def test_sobol():
    x, _ = gen_rv(SAMPLE_CNT)
    dataset = pd.DataFrame(x) #  dataframe
    # %% gen aPC polynomials and roots
    pc_nr_range = [0]
    h_dict = dt.genHankel(dataset, SRCS, pc_nr_range, P)
    pc_roots, pc_weights = dt.gen_roots_weights(h_dict, METHOD)
    pc_dict = dt.gen_pcs(h_dict, METHOD)
    npc_dict = dt.gen_npcs_mm(pc_dict, h_dict)
    nrb_dict = dt.gen_nr_range_bds(dataset, SRCS, pc_nr_range)
    #MK_LST = dt.gen_mkey_list(PC_ROOTS, SRCS)
    # get roots and weights and long MK-list for the output
    roots_eval, _, mk_lst = dt.get_rw_4nrs(np.zeros(DIM), SRCS,
                                           pc_roots, pc_weights)
    _, mk2sid = dt.gen_mkey_sid_rel(roots_eval, mk_lst, nrb_dict)
    pol_vals = dt.gen_pol_on_samples_arr(roots_eval, npc_dict, ALPHAS, mk2sid)
    y_rt = ISH_FCT(roots_eval)

    cf_ls_4s = np.zeros((len(y_rt), P_MR)) # Fct coefs on each sample, (sid, p, x): by LS

    for mkey, sids in mk2sid.items():
        phi = pol_vals[:, sids].T

        # v, resid, rank, sigma = linalg.lstsq(A,y)
        # solves Av = y using least squares
        # sigma - singular values of A
        v_ls, _, _, _ = np.linalg.lstsq(
            phi, y_rt[sids], rcond=None) # LS - output
        cf_ls_4s[sids, :] = v_ls

    # %% Compute Sobol coefs
    pc_cfs = cf_ls_4s[0] # pol cfs 4 Nr=0, (identical for all samples)
    src_idx = sob.gen_idx_subsets(DIM)
    _, sob_cfs = ishigami_exact_sensitivity(A, B)
    for src in src_idx:
        idx = list(src)
        sob_idx = sob.sobol_idx_pc(pc_cfs, ALPHAS, idx)
        sob_idx_ex = sob_cfs[src]
        assert abs(sob_idx - sob_idx_ex) < TOL
