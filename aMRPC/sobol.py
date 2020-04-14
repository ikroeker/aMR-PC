#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:15:45 2020

@author: kroeker
"""
import itertools as it
import numpy as np
import aMRPC.datatools as dt
import aMRPC.utils as u

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
            chk_out = True
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

def sobol_tot_sen_pc(pc_coefs, alphas, src_idxs, idx_list):
    tot = 0
    idx_set = set(idx_list)
    for idx_it in src_idxs:
        if idx_set <= idx_it:
            tot += sobol_idx_pc(pc_coefs, alphas, list(idx_it))
    return tot

def sobol_idx_amrpc(pc_coefs, rsc_dict, mk2sid, alphas, idx_set):
    mean, var = dt.cf_2_mean_var(pc_coefs, rsc_dict, mk2sid)
    p_max, dim = alphas.shape
    srcs = list(range(dim))
    assert max(idx_set) < dim
    not_in_idx_set = list(set(srcs)-set(idx_set))
    #loc_rsc_cf = 2**(- len(not_in_idx_set)*4)
    sobol_ns = 0

    for mkey, sids in mk2sid.items():
        loc_pc = pc_coefs[sids[0], :]
        sobol_mk = 0 #loc_pc[0]**2
        r_cf = rsc_dict[mkey]
        cf = u.gen_corr_rcf(mkey, not_in_idx_set)
        for a_mkey, a_sids in mk2sid.items():
            loc_pc_a = pc_coefs[a_sids[0], :]
            #idx_diff = u.multi_key_intersect_srcs(mkey, a_mkey)
            #if set(idx_diff) <= set(idx_set):
            #mk_chk = mkey == a_mkey
            #    sobol_mk += loc_pc[0]**2
            #else:
            mk_chk = u.compare_multi_key_for_idx(mkey, a_mkey, idx_set)
            if mk_chk:
                #sobol_mk += loc_pc[0] * loc_pc_a[0]
                for pidx in range(p_max):
                    alpha = alphas[pidx, :]
                    chk_in = alpha[idx_set].min() > 0
                    if not not_in_idx_set:
                        chk_out = True
                    else:
                        chk_out = alpha[not_in_idx_set].max() == 0
                    if chk_out:
                        sobol_mk += loc_pc[pidx] * loc_pc_a[pidx]
        sobol_mk *= r_cf * cf
        sobol_ns += sobol_mk
    sobol_ns -= mean**2
    #print(mean, var, 1/rsc_dict[mkey])
    return sobol_ns / var
