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
NR = 3
NO = 2

METHOD = 1 # gen roots, weights, polynomials method
SRCS = [0, 1, 2]
# params
DIM = len(SRCS)
ALPHAS = u.gen_multi_idx(P, DIM)  #powers of multivar. polynomials
P_PC = ALPHAS.shape[0] # number of coefficients
ALPHAS_MR = u.gen_multi_idx(NO, DIM)  #powers of multivar. polynomials
P_MR = ALPHAS_MR.shape[0] # number of coefficients

# Ishigami function:
A = 7
B = 0.1

ISH_FCT = lambda X: A*np.sin(X[:, 1])**2 + (1+B*np.power(X[:, 2], 4))*np.sin(X[:, 0])

def ishigami_exact_sensitivity(a, b):
    """
    generates analytical Sobol sensitivy indexes of Ishigami function

    Parameters
    ----------
    a : float
        parameter a.
    b : float
        parameter b.

    Returns
    -------
    D : float
        variance.
    sob_cfs : dictionary[frozenset(sources)] = index
        dictionary of analytocal Sobol indexes.

    """
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
    """ generates random vector and apply Ishany function on """
    np.random.seed(3)
    x = np.random.uniform(-math.pi, math.pi, (sample_cnt, 3))
    y = ISH_FCT(x)
    return x, y


def test_sobol():
    """
    Tests Sobool indexes for aPc agains analytical solution.

    Returns
    -------
    None.

    """
    x, _ = gen_rv(SAMPLE_CNT)
    dataset = pd.DataFrame(x) #  dataframe
    #%% gen aPC polynomials and roots
    pc_nr_range = [0]
    h_dict = dt.genHankel(dataset, SRCS, pc_nr_range, P)
    pc_roots, pc_weights = dt.gen_roots_weights(h_dict, METHOD)
    pc_dict = dt.gen_pcs(h_dict, METHOD)
    npc_dict = dt.gen_npcs_mm(pc_dict, h_dict)
    nrb_dict = dt.gen_nr_range_bds(dataset, SRCS, pc_nr_range)
    # get roots and weights and long MK-list for the output
    roots_eval, _, mk_lst_l = dt.get_rw_4nrs(np.zeros(DIM), SRCS,
                                             pc_roots, pc_weights)
    mk_lst = list(set(mk_lst_l))
    _, mk2sid = dt.gen_mkey_sid_rel(roots_eval, mk_lst, nrb_dict)
    pol_vals = dt.gen_pol_on_samples_arr(roots_eval, npc_dict, ALPHAS, mk2sid)
    y_rt = ISH_FCT(roots_eval)

    cf_ls_4s = np.zeros((len(y_rt), P_PC)) # Fct coefs on each sample, (sid, p, x): by LS

    for sids in mk2sid.values():
        phi = pol_vals[:, sids].T

        # v, resid, rank, sigma = linalg.lstsq(A,y)
        # solves Av = y using least squares
        # sigma - singular values of A
        v_ls, _, _, _ = np.linalg.lstsq(
            phi, y_rt[sids], rcond=-1) # LS - output
        cf_ls_4s[sids, :] = v_ls

    #%% Compute Sobol coefs
    pc_cfs = cf_ls_4s[0] # pol cfs 4 Nr=0, (identical for all samples)
    src_idx = sob.gen_idx_subsets(DIM)
    _, sob_cfs = ishigami_exact_sensitivity(A, B)
    for src in src_idx:
        idx = list(src)
        sob_idx = sob.sobol_idx_pc(pc_cfs, ALPHAS, idx)
        sob_idx_ex = sob_cfs[src]
        assert abs(sob_idx - sob_idx_ex) < TOL

def test_amr_sobol():
    """
    Tests Sobol indexes for aMR-PC again analytical solution

    Returns
    -------
    None.

    """
    x, _ = gen_rv(SAMPLE_CNT)
    dataset = pd.DataFrame(x) #  dataframe
    nr_range = np.arange(NR, NR+1)
    h_dict = dt.genHankel(dataset, SRCS, nr_range, NO)
    roots, weights = dt.gen_roots_weights(h_dict, METHOD)
    pc_dict = dt.gen_pcs(h_dict, METHOD)
    npc_dict = dt.gen_npcs_mm(pc_dict, h_dict)
    nrb_dict = dt.gen_nr_range_bds(dataset, SRCS, nr_range)
    # get roots and weights and long MK-list for the output
    roots_eval, _, mk_lst_l = dt.get_rw_4nrs(NR*np.ones(DIM), SRCS,
                                             roots, weights)
    mk_lst = list(set(mk_lst_l))
    rsc_dict = dt.gen_rcf_dict(mk_lst)# rescaling coefficients for proj-> Nr=n_r

    _, mk2sid = dt.gen_mkey_sid_rel(roots_eval, mk_lst, nrb_dict)
    pol_vals = dt.gen_pol_on_samples_arr(roots_eval, npc_dict, ALPHAS_MR, mk2sid)
    y_rt = ISH_FCT(roots_eval)

    cf_ls_4s = np.zeros((len(y_rt), P_MR)) # Fct coefs on each sample, (sid, p, x): by LS

    for sids in mk2sid.values():
        phi = pol_vals[:, sids].T

        # v, resid, rank, sigma = linalg.lstsq(A,y)
        # solves Av = y using least squares
        # sigma - singular values of A
        v_ls, _, _, _ = np.linalg.lstsq(
            phi, y_rt[sids], rcond=-1) # LS - output
        cf_ls_4s[sids, :] = v_ls

    #%% Compute Sobol coefs
    #pc_cfs = cf_ls_4s[0] # pol cfs 4 Nr=0, (identical for all samples)
    src_idx = sob.gen_idx_subsets(DIM)
    _, sob_cfs = ishigami_exact_sensitivity(A, B)
    tmp_dict = {}
    for src in src_idx:
        idx = list(src)
        sob_idx = sob.sobol_idx_amrpc_helper(cf_ls_4s, rsc_dict, mk2sid, ALPHAS_MR, idx)
        tmp_dict[src] = sob_idx
    for src in src_idx:
        idx = list(src)
        sob_idx = sob.sobol_idx_amrpc(tmp_dict, src)
        sob_idx_ex = sob_cfs[src]
        assert abs(sob_idx - sob_idx_ex) < TOL

def test_amr_sobol_comb():
    """
    Tests Sobol indexes for combinatoric aMR-PC again analytical solution

    Returns
    -------
    None.

    """
    x, _ = gen_rv(SAMPLE_CNT)
    dataset = pd.DataFrame(x) #  dataframe
    nr_range = np.arange(NR, NR+1)
    h_dict = dt.genHankel(dataset, SRCS, nr_range, NO)
    roots, weights = dt.gen_roots_weights(h_dict, METHOD)
    pc_dict = dt.gen_pcs(h_dict, METHOD)
    npc_dict = dt.gen_npcs_mm(pc_dict, h_dict)
    nrb_dict = dt.gen_nr_range_bds(dataset, SRCS, nr_range)
    # get roots and weights and long MK-list for the output
    roots_eval, _, mk_lst_l = dt.get_rw_4nrs(NR*np.ones(DIM), SRCS,
                                             roots, weights)
    mk_lst = list(set(mk_lst_l))
    rsc_dict = dt.gen_rcf_dict(mk_lst)# rescaling coefficients for proj-> Nr=n_r

    _, mk2sid = dt.gen_mkey_sid_rel(roots_eval, mk_lst, nrb_dict)
    pol_vals = dt.gen_pol_on_samples_arr(roots_eval, npc_dict, ALPHAS_MR, mk2sid)
    y_rt = ISH_FCT(roots_eval)

    cf_ls_4s = np.zeros((len(y_rt), P_MR)) # Fct coefs on each sample, (sid, p, x): by LS

    for sids in mk2sid.values():
        phi = pol_vals[:, sids].T

        # v, resid, rank, sigma = linalg.lstsq(A,y)
        # solves Av = y using least squares
        # sigma - singular values of A
        v_ls, _, _, _ = np.linalg.lstsq(
            phi, y_rt[sids], rcond=-1) # LS - output
        cf_ls_4s[sids, :] = v_ls

    #%% Compute Sobol coefs
    #pc_cfs = cf_ls_4s[0] # pol cfs 4 Nr=0, (identical for all samples)
    src_idx = sob.gen_idx_subsets(DIM)
    _, sob_cfs = ishigami_exact_sensitivity(A, B)
    #tmp_dict = {}
    tmp_dict = sob.gen_sobol_amrpc_dict(cf_ls_4s, rsc_dict, mk2sid, ALPHAS_MR, SRCS)
    sob_idx, sobol_dict = sob.sobol_idx_amrpc_comb(tmp_dict, frozenset(SRCS), {})
    for src in src_idx:
        sob_idx = sobol_dict[src]
        sob_idx_ex = sob_cfs[src]
        assert abs(sob_idx - sob_idx_ex) < TOL

def test_amr_sobol_dynamic():
    """
    Tests Sobol indexes for dynamic aMR-PC again analytical solution

    Returns
    -------
    None.

    """
    x, _ = gen_rv(SAMPLE_CNT)
    dataset = pd.DataFrame(x) #  dataframe
    nr_range = np.arange(NR, NR+1)
    h_dict = dt.genHankel(dataset, SRCS, nr_range, NO)
    roots, weights = dt.gen_roots_weights(h_dict, METHOD)
    pc_dict = dt.gen_pcs(h_dict, METHOD)
    npc_dict = dt.gen_npcs_mm(pc_dict, h_dict)
    nrb_dict = dt.gen_nr_range_bds(dataset, SRCS, nr_range)
    # get roots and weights and long MK-list for the output
    roots_eval, _, mk_lst_l = dt.get_rw_4nrs(NR*np.ones(DIM), SRCS,
                                             roots, weights)
    mk_lst = list(set(mk_lst_l))
    rsc_dict = dt.gen_rcf_dict(mk_lst)# rescaling coefficients for proj-> Nr=n_r

    _, mk2sid = dt.gen_mkey_sid_rel(roots_eval, mk_lst, nrb_dict)
    pol_vals = dt.gen_pol_on_samples_arr(roots_eval, npc_dict, ALPHAS_MR, mk2sid)
    y_rt = ISH_FCT(roots_eval)

    cf_ls_4s = np.zeros((len(y_rt), P_MR)) # Fct coefs on each sample, (sid, p, x): by LS

    for sids in mk2sid.values():
        phi = pol_vals[:, sids].T

        # v, resid, rank, sigma = linalg.lstsq(A,y)
        # solves Av = y using least squares
        # sigma - singular values of A
        v_ls, _, _, _ = np.linalg.lstsq(
            phi, y_rt[sids], rcond=-1) # LS - output
        cf_ls_4s[sids, :] = v_ls

    #%% Compute Sobol coefs
    #pc_cfs = cf_ls_4s[0] # pol cfs 4 Nr=0, (identical for all samples)
    src_idx = sob.gen_idx_subsets(DIM)
    _, sob_cfs = ishigami_exact_sensitivity(A, B)
    sobol_dict = {}
    help_sobol_dict = {}
    for src in src_idx:
        sob_idx, sobol_dict, help_sobol_dict = sob.sobol_idx_amrpc_dynamic(src, 
                                                                           cf_ls_4s, 
                                                                           rsc_dict, 
                                                                           mk2sid, 
                                                                           ALPHAS_MR,
                                                                           sobol_dict,
                                                                           help_sobol_dict)
        sob_idx_ex = sob_cfs[src]
        assert abs(sob_idx - sob_idx_ex) < TOL
