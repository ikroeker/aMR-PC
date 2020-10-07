#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:15:45 2020

@author: kroeker
"""
import math
import itertools as it
import numpy as np
import aMRPC.datatools as dt
import aMRPC.utils as u

#  Sobol idx for aPC
def sobol_idx_pc(pc_coefs, alphas, idx_set, eps=1e-15):
    """
    computes Sobol indexes from polynomial chaos (PC) expansion

    Parameters
    ----------
    pc_coefs : numpy.ndarray
        PC coefficients.
    alphas : numpy.ndarray
        polynomial degrees related to pc_coefs.
    idx_set : list
        list of sources to be considered.
    eps : float, optional
        threshold for minimal value for variance. Set var=1 vor var<=eps.
        The default is 1e-15.

    Returns
    -------
    sobol : float / numpy.ndarray
        Sobol indexes for idx_set.

    """
    cf_tup = pc_coefs.shape
    if len(cf_tup) == 1 or cf_tup[1] == 1:
        var_pc = pc_coefs[1:] @ pc_coefs[1:]
        sobol = 0
    else:
        sobol = np.zeros(cf_tup[1])
        var_pc = np.add.reduce(pc_coefs[1:] * pc_coefs[1:], axis=0)
        var_thresh = var_pc <= eps
        if max(var_thresh):
            var_pc[var_thresh] = 1

    p_max, dim = alphas.shape
    srcs = list(range(dim))
    assert max(idx_set) < dim
    not_in_idx_set = list(set(srcs)-set(idx_set))
    #sobol = 0
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
    """
    Generates all possible source combination for dim-sources

    Parameters
    ----------
    dim : int
        number/dimension of sources.

    Returns
    -------
    TYPE
        frozenset of frozensets.

    """
    items = list(range(dim))
    sub_idx = [frozenset(t) for length in range(1, dim+1)
               for t in it.combinations(items, length)]
    return set(sub_idx)

def gen_sidx_subsets(sidx):
    items = list(sidx)
    sub_idx = [frozenset(t) for length in range(1, len(sidx)+1)
               for t in it.combinations(items, length)]
    return set(sub_idx)

def sobol_tot_sen_pc(pc_coefs, alphas, src_idxs, idx_list):
    """
    Computes total sensitivity coeficients for polynomial-chaos (PC) expansion

    Parameters
    ----------
    pc_coefs : numpy.ndarray
        PC coefficients.
    alphas : numpy.ndarray
        polynomial degrees related to pc_coefs.
    src_idxs : frozenset
        all possbile index combinations.
    idx_list : list
        idx_list to be considered.

    Returns
    -------
    tot : float / numpy.ndarray
        total sensitivity for sources in idx_list.

    """
    tot = 0
    idx_set = set(idx_list)
    for idx_it in src_idxs:
        if idx_set <= idx_it:
            tot += sobol_idx_pc(pc_coefs, alphas, list(idx_it))
    return tot

def sobol_tot_sens(sob_dict, src_idxs, idx_list):
    """
    total sensitivity

    Parameters
    ----------
    sob_dict : dictionary
        dictionary of Soblo coeffcients.
    src_idxs : frozenset
        set of all possbile source combinations.
    idx_list : list
        list of source-indexes to be considered.

    Returns
    -------
    tot : float / numpy.ndarray
        total sensitivity for sources in idx_list.

    """
    tot = 0
    idx_set = set(idx_list)
    for idx_it in src_idxs:
        if idx_set <= idx_it:
            tot += sob_dict[idx_it]
    return tot

def sobol_idx_amrpc_helper(pc_coefs, rsc_dict, mk2sid, alphas, idx_list, eps=1e-15):
    """
    Helper function for computation of Sobol indexes for aMR-PC expansion
    results are required by sobol_idx_amrpc(...)

    Parameters
    ----------
    pc_coefs : numpy.ndarray
        polynomial coeficiens of the aMR-PC expansion.
    rsc_dict : dictionary
        dictionary of rescaling coefficients.
    mk2sid : dictionary
        multi-key -> sample id.
    alphas : numpy.ndarray
        polynomial degrees.
    idx_list : frozenset
        set of all source combinations.
    eps : float, optional
        variance threshold. The default is 1e-15.

    Returns
    -------
    float / numpy.ndarray
        input data for solbol_idx_amprc.

    """
    mean, var = dt.cf_2_mean_var(pc_coefs, rsc_dict, mk2sid)
    p_max, dim = alphas.shape
    srcs = list(range(dim))
    assert max(idx_list) < dim
    not_in_idx_set = list(set(srcs)-set(idx_list))
    #loc_rsc_cf = 2**(- len(not_in_idx_set)*4)
    # omit normalization for var <=eps
    var_thresh = var <= eps
    if max(var_thresh):
        var[var_thresh] = 1
    # replace np.array by scalars for len(mean) == 1
    qnt_len = mean.shape[0]
    if qnt_len == 1:
        mean = mean[0]
        var = var[0]
        sobol_ns = 0
    else:
        sobol_ns = np.zeros(mean.shape)

    for mkey, sids in mk2sid.items():
        loc_pc = pc_coefs[sids[0], :]
        sobol_mk = 0 #loc_pc[0]**2
        r_cf = rsc_dict[mkey]
        c_cf = u.gen_corr_rcf(mkey, not_in_idx_set)
        for a_mkey, a_sids in mk2sid.items():
            loc_pc_a = pc_coefs[a_sids[0], :]
            a_r_cf = rsc_dict[a_mkey]
            a_c_cf = u.gen_corr_rcf(a_mkey, not_in_idx_set)
            #idx_diff = u.multi_key_intersect_srcs(mkey, a_mkey)
            #if set(idx_diff) <= set(idx_set):
            #mk_chk = mkey == a_mkey
            #    sobol_mk += loc_pc[0]**2
            #else:
            mk_chk = u.compare_multi_key_for_idx(mkey, a_mkey, idx_list)
            if mk_chk:
                #sobol_mk += loc_pc[0] * loc_pc_a[0]
                for pidx in range(p_max):
                    alpha = alphas[pidx, :]
                    #chk_in = alpha[idx_list].min() > 0
                    if not not_in_idx_set:
                        chk_out = True
                    else:
                        chk_out = alpha[not_in_idx_set].max() == 0
                    if chk_out:
                        sobol_mk += (loc_pc[pidx] * loc_pc_a[pidx]
                                     * math.sqrt(a_r_cf*a_c_cf))
        sobol_mk *= math.sqrt(r_cf * c_cf)
        sobol_ns += sobol_mk
    sobol_ns -= mean**2
    #print(mean, var, 1/rsc_dict[mkey])
    return sobol_ns / var
    #return sobol_ns

def sobol_idx_amrpc(sobol_dict, idx_set):
    """
    Computes Sobol indexes for aMR-PC expansion using dictionary provided by
    sobol_idx_amrpc_helper(...)

    Parameters
    ----------
    sobol_dict : dicitonary
        output of sobol_idx_amrpc_helper(...).
    idx_set : list
        list of sources to be considered.

    Returns
    -------
    ret_val : float / numpy.ndarray
        Sobol sensitivities for sources in idx_set.

    """
    idx_set_len = len(idx_set)
    ret_val = 0 #sobol_dict[idx_set]
    if idx_set_len > 1:
        sub_idx = gen_sidx_subsets(idx_set)
        #print(idx_set, sub_idx)
        for idx in sub_idx:
            #print(((-1)**(len(idx_set-idx))), idx_set, idx)
            ret_val += ((-1)**(len(idx_set-idx)))*sobol_dict[idx]
    else:
        ret_val = sobol_dict[idx_set]
    return ret_val

def sobol_idx_amrpc_jj(pc_coefs, rsc_dict, mk2sid, alphas, idx_list, eps=1e-15):
    mean, var = dt.cf_2_mean_var(pc_coefs, rsc_dict, mk2sid)
    p_max, dim = alphas.shape
    srcs = list(range(dim))
    assert max(idx_list) < dim

    not_in_idx_set = list(set(srcs)-set(idx_list))

    var_thresh = var <= eps
    if max(var_thresh):
        var[var_thresh] = 1
    # replace np.array by scalars for len(mean) == 1
    qnt_len = mean.shape[0]
    if qnt_len == 1:
        mean = mean[0]
        var = var[0]
        sobol_ns = 0
    else:
        sobol_ns = np.zeros(mean.shape)

    for mkey, sids in mk2sid.items():
        loc_pc = pc_coefs[sids[0], :]
        sobol_mk = 0 #loc_pc[0]**2
        r_cf = rsc_dict[mkey]
        c_cf = u.gen_corr_rcf(mkey, not_in_idx_set)
        #c_cf = u.gen_corr_rcf(mkey, not_in_idx_set)
        for a_mkey, a_sids in mk2sid.items():
            mk_chk = u.compare_multi_key_for_idx(mkey, a_mkey, idx_list)

            if mk_chk:
                loc_pc_a = pc_coefs[a_sids[0], :]
                a_r_cf = rsc_dict[a_mkey]
                a_c_cf = u.gen_corr_rcf(a_mkey, not_in_idx_set)
                for pidx in range(p_max):
                    alpha = alphas[pidx, :]
                    #chk_in = alpha[idx_list].min() > 0
                    if not not_in_idx_set:
                        chk_out = True
                    else:
                        chk_out = alpha[not_in_idx_set].max() == 0

                    if chk_out:
                        sobol_mk += (loc_pc[pidx] * loc_pc_a[pidx]
                                     * math.sqrt(a_r_cf*a_c_cf))

        sobol_mk *= math.sqrt(r_cf*c_cf)
        sobol_ns += sobol_mk
    sobol_ns -= mean**2
    #var_ths = var >= eps
    return sobol_ns / var



def gen_sobol_amrpc_dict(pc_coefs, rsc_dict, mk2sid, alphas, idx_list, eps=1e-15):
    # idx_list_len = len(idx_list)
    # sub_idx = [frozenset(t) for length in range(1, idx_list_len+1)
    #            for t in it.combinations(idx_list, length)]
    sub_idx = gen_sidx_subsets(set(idx_list))
    sobol_dict = {}
    for j in sub_idx:
        #print(j)
        sobol_dict[j] = sobol_idx_amrpc_jj(pc_coefs, rsc_dict,
                                           mk2sid, alphas,
                                           list(j), eps)
    return sobol_dict

def sobol_idx_amrpc_comb(help_sobol_dict, idx_set, tmp_sobol_dict):
    """
    Computes Sobol indexes for aMR-PC expansion using dictionary provided by
    gen_sobol_amrpc_dict(...)

    Parameters
    ----------
    sobol_dict : dicitonary
        output of sobol_idx_amrpc_jk(...).
    idx_set : set
        list of sources to be considered.

    Returns
    -------
    ret_val : float / numpy.ndarray
        Sobol sensitivities for sources in idx_list.

    """
    idx_set_len = len(idx_set)
    ret_val = help_sobol_dict[idx_set]
    if idx_set_len > 1:
        #sub_idx = gen_sidx_subsets(list(idx_set)) - idx_set
        items = list(idx_set)
        sub_idx = [frozenset(t) for length in range(1, idx_set_len)
                    for t in it.combinations(items, length)]
        #print(idx_set, sub_idx)
        for j_idx in sub_idx:
            j_len = len(j_idx)
            #print('jj:',j_idx, help_sobol_dict[j_idx], j_len)
            if j_idx not in tmp_sobol_dict.keys():
                if j_len == 1:
                    tmp_sobol_dict[j_idx] = help_sobol_dict[j_idx]
                else:
                    _, tmp_sobol_dict = sobol_idx_amrpc_comb(help_sobol_dict,
                                                             j_idx, tmp_sobol_dict)
            ret_val -= tmp_sobol_dict[j_idx]
            #print('ret=', ret_val)

    tmp_sobol_dict[idx_set] = ret_val
    return ret_val, tmp_sobol_dict

def sobol_idx_amrpc_dynamic(idx_set, pc_coefs, rsc_dict, mk2sid, alphas,
                            sobol_dict, help_sobol_dict, eps=1e-15):
    idx_set_len = len(idx_set)
    if idx_set in sobol_dict.keys():
        return sobol_dict[idx_set], sobol_dict, help_sobol_dict
    
    if idx_set not in help_sobol_dict.keys():
        help_sobol_dict[idx_set] = sobol_idx_amrpc_jj(pc_coefs, rsc_dict,
                                                      mk2sid, alphas,
                                                      list(idx_set), eps)
    ret_val = help_sobol_dict[idx_set]
    if idx_set_len > 1:
        items = list(idx_set)
        sub_idx = [frozenset(t) for length in range(1, idx_set_len)
                   for t in it.combinations(items, length)]
        #print(idx_set, sub_idx)
        for j_idx in sub_idx:
            j_len = len(j_idx)
            #print('jj:',j_idx, help_sobol_dict[j_idx], j_len)
            if j_idx not in sobol_dict.keys():
                if j_len == 1:
                    if j_idx not in help_sobol_dict.keys():
                        help_sobol_dict[j_idx] = sobol_idx_amrpc_jj(pc_coefs, rsc_dict,
                                                                    mk2sid, alphas,
                                                                    list(j_idx), eps)
                    sobol_dict[j_idx] = help_sobol_dict[j_idx]
                else:
                    _, sobol_dict, help_sobol_dict = sobol_idx_amrpc_dynamic(j_idx, pc_coefs,
                                                                                 rsc_dict, mk2sid,
                                                                                 alphas,
                                                                                 sobol_dict,
                                                                                 help_sobol_dict,
                                                                                 eps)
            ret_val -= sobol_dict[j_idx]
            #print('ret=', ret_val)
    sobol_dict[idx_set] = ret_val
    return ret_val, sobol_dict, help_sobol_dict
