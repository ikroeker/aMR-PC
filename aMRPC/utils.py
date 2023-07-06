"""
utils.py - provides functions and dictionary for index and key management
to handle multi-resolution

@author: kroeker
https://orcid.org/0000-0003-0360-5307

"""

import itertools as it
# from math import comb
import numpy as np
from scipy.special import comb
try:
    from numba import jit, njit  # , jit_module
    NJM = True
except ImportError:
    NJM = False
    pass

# import math

# Data format for roots, weights and details: DictName(Nr,aNr,Nri,src)
# ParPos={'Nr':0,'aNr':1,'Nri':2,'src':3}
ParPos = {'Nr': 0, 'aNr': 0, 'Nri': 1, 'src': 2}


def gen_multi_idx_old(n_o, dim):
    """
    generates mulit-indices of multi-variate polynomial base
    uses graded lexicographic ordering (p. 156, Sullivan)
    old version
    """
    p_cnt = comb(n_o+dim, dim)
    alphas = np.zeros((np.uint32(p_cnt), dim), dtype=np.int32)
    tmp_arr = np.zeros(dim)
    l_idx = 1
    pmax = (n_o+1)**dim
    for p_idx in range(1, pmax):
        p_a = p_idx
        for d_a in range(dim):
            m_d = (n_o+1)**(dim-d_a-1)
            val = p_a//m_d
            tmp_arr[d_a] = val
            p_a = p_a-val*m_d
        if sum(tmp_arr) <= n_o:
            alphas[l_idx, :] = tmp_arr
            l_idx = l_idx+1
            if l_idx == p_cnt:
                break
    return alphas


# @jit(debug=True)
def gen_multi_idx(n_o, dim):
    """
    generates mulit-indices of multi-variate polynomial base
    uses graded lexicographic ordering (p. 156, Sullivan)
    """
    p_cnt = np.int64(comb(n_o+dim, dim, exact=True))
    alphas = np.zeros((p_cnt, dim), dtype=np.uint32)
    l_idx = 0
    tmp_it = it.product(range(n_o+1), repeat=dim)
    for p_it in tmp_it:
        if sum(p_it) <= n_o:
            alphas[l_idx, :] = np.array(p_it, dtype=np.uint32)
            l_idx = l_idx+1
            if l_idx == p_cnt:
                break
    return alphas


@jit(nopython=True, nogil=True)
def gen_midx_mask(alphas, no_max):
    """
    generates a mask for alphas, such that all multi-index polynomial degrees
    are below ( <=)no_max
    """
    p_cnt = alphas.shape[0]
    a_mask = np.empty(p_cnt, dtype=np.bool8)
    for i in range(p_cnt):
        a_mask[i] = alphas[i, :].sum() <= no_max
    return a_mask


@jit(nopython=True, nogil=True)
def gen_midx_mask_part(alphas, no_min, no_max, idx_set):
    """
    generates a mask for alphas, such that all multi-index polynomial degrees
    are below ( <=)no_max for idxs in idx_set only
    for idx not in idx_st pol degree <=no_min
    """
    p_cnt, dim = alphas.shape
    a_mask = np.zeros(p_cnt, dtype=np.bool8)
    for i in range(p_cnt):
        a_mask[i] = alphas[i, :].sum() <= no_max
        for _d in range(dim):
            if _d not in idx_set:
                a_mask[i] = a_mask[i] and alphas[i, _d] <= no_min
    return a_mask


@jit(nopython=True, nogil=True)
def gen_midx_mask_hyp(alphas, no_max, p_norm):
    """
    generates a mask for alphas, such that all P-Norms of
    multi-index polynomial degrees
    are below ( <=) no_max
    see Hyperbolic trunction in
    Sparse Polynomial Chaos Expansions: Literature Survey and Benchmark
    Nora Lüthen, Stefano Marelli, and Bruno Sudret
    """
    p_cnt = alphas.shape[0]
    a_mask = np.empty(p_cnt, dtype=np.bool8)
    for i in range(p_cnt):
        i_arr = alphas[i, :].astype(np.float32)
        a_mask[i] = np.linalg.norm(i_arr, p_norm) <= no_max
    return a_mask


def gen_nri_range(nrs):
    """
    generates an array with Nri-entries
    input:
    Nrs : list of Nr for each dim, e.g. [Nr, Nr]

    return:
    Nris :  np.array with Nri-(integer) entries
    NriCnt : length of the array
    """
    dim = len(nrs)
    nri_cnts = np.zeros(dim)
    divs = np.zeros(dim)
    nri_cnt = 1
    for d_idx in range(dim):
        nri_cnts[d_idx] = 2**(nrs[d_idx])
        divs[d_idx] = nri_cnts[0:d_idx].prod()
    nri_cnt = int(nri_cnts.prod())
    nris = np.zeros((nri_cnt, dim), dtype=np.uint32)
    for nri in range(nri_cnt):
        for d_idx in range(dim):
            val = (nri//divs[d_idx] % nri_cnts[d_idx])
            nris[nri, d_idx] = val
    return nris, nri_cnt


def gen_nri_range_4mkset(mkey_set, dim):
    """
    generates an np.array with Nri-entries
    input:
    mkey_set -- set of multi-keys
    dim -- dimension, also number of keys in each mkey
    return:
    nris -- np.array with Nri-(integer) entries
    nri_cnt -- length of the array
    """
    nri_cnt = len(mkey_set)  # number of multi-keys
    pos = ParPos['Nri']
    assert nri_cnt > 0
    nris = np.zeros((nri_cnt, dim), dtype=np.uint64)
    cnt = 0  # counter
    for mkey in mkey_set:
        for d_i in range(dim):
            nris[cnt, d_i] = int(mkey[d_i][pos])
        cnt += 1
    assert cnt == nri_cnt
    return nris, nri_cnt


@njit(nogil=True)
def midx4quad(ar_lens):
    """ generates indexes for eval. points etc. """
    # n_lens = np.array(ar_lens, dtype=np.int32)
    cols = len(ar_lens)
    lines = ar_lens.prod()
    idx_mx = np.zeros((lines, cols), dtype=np.uint32)
    divs = np.zeros(cols)
    for col in range(cols):
        divs[col] = ar_lens[0:col].prod()
    # print(divs)
    for l_idx in range(lines):
        for col in range(cols):
            val = (l_idx//divs[col] % ar_lens[col])
            idx_mx[l_idx, col] = val
    return idx_mx


def gen_dict_key(anr, nri, src=None):
    """
    generates dictionary key for
    aNr - actually Nr, Nri , src
    according to ParPos
    """
    # chkNri = Nri < 2**aNr
    # assert(chkNri)
    if src is None:
        ret = (anr, nri)
    else:
        ret = (anr, nri, src)
    return ret


def get_dict_entry(adict, anr, nri, src=-1):
    """ returns dictionary entry """
    key = gen_dict_key(anr, nri, src)
    return adict[key]


def gen_multi_key(anrs, nris, srcs):
    """
    generates a tuple of tuples that will be used as a key
    srcs - source numbers
    aNrs - Nr- levels for each entree in srcs
    Nris - Nr indices for each entree in src
    """
    dims = len(srcs)
    return tuple(gen_dict_key(anrs[d], nris[d], srcs[d]) for d in range(dims))


def multi_key2srcs(mkey):
    """ generates srcs list from multi-key """
    src_pos = ParPos['src']
    return [c[src_pos] for c in mkey]


def multi_key_diff_srcs(mkey_one, mkey_two):
    """
    Computes difference between two multi keys

    Parameters
    ----------
    mkey_one : tuple
        multi-key one.
    mkey_two : tuple
        multie-key two.

    Returns
    -------
    list
        difference between mkey_one and mkey_two.

    """
    mk_len = len(mkey_one)
    assert mk_len == len(mkey_two)
    # spos = ParPos['src']
    srcs = multi_key2srcs(mkey_one)
    return [srcs[src] for src in range(mk_len) if mkey_one[src] != mkey_two[src]]


def multi_key_intersect_srcs(mkey_one, mkey_two):
    """
    Computes intersection between two multi keys

    Parameters
    ----------
    mkey_one : tuple
        multi-key one.
    mkey_two : tuple
        multie-key two.

    Returns
    -------
    list
        intersection between mkey_one and mkey_two.

    """
    mk_len = len(mkey_one)
    assert mk_len == len(mkey_two)
    # spos = ParPos['src']
    srcs = multi_key2srcs(mkey_one)
    return [srcs[src] for src in range(mk_len) if mkey_one[src] == mkey_two[src]]


def compare_multi_key_for_idx(mkey_one, mkey_two, srcs):
    """
    Comapres two multi-keys for indexes given in srcs

    Parameters
    ----------
    mkey_one : tuple
        multi-key one.
    mkey_two : tuple
        multi-key tow.
    srcs : list
        input indexes to compare.

    Returns
    -------
    ret : bool
        true if all equal.

    """
    mkey_len = len(mkey_two)
    chk_a = len(mkey_one) == mkey_len
    chk_b = max(srcs) < mkey_len
    assert chk_a and chk_b
    ret = True
    for src in srcs:
        ret = ret and (mkey_one[src] == mkey_two[src])
    return ret


def gen_corr_rcf(mkey, srcs):
    """
    Generates multi-resolution correcting/re-scalling coefficient for a multi-key

    Parameters
    ----------
    mkey : tuple
        multi-key.
    srcs : list
        indexes to be considered.

    Returns
    -------
    correct_cf : float
        correcting/res-calling cofficient.

    """
    nr_pos = ParPos['aNr']
    correct_cf = 1
    for src in srcs:
        correct_cf *= 2**(-mkey[src][nr_pos])
    return correct_cf


def get_multi_entry(adict, anrs, nris, srcs):
    """ returns dictionary entry """
    key = gen_multi_key(anrs, nris, srcs)
    return adict[key]


def choose_cols(arr, ccols, zero_cols=-1):
    """ picks the ccols columns from arr """
    ret = arr[:, ccols]
    if zero_cols != -1:
        ret[:, zero_cols] = 0
    return ret


def inv_src_arr(srcs):
    """ generates dictionary with positions of srcs """
    isrc = {}
    i = 0
    for src in srcs:
        isrc[src] = i
        i = i+1
    return isrc


# if NJM:
#     # jit_module(nopython=True, error_model="numpy")
#     jit_module(error_model="numpy")

if __name__ == "__main__":
    print("test utils.py")
    NO = 3
    DIM = 2
    ALPHAS = gen_multi_idx(NO, DIM)
    print(ALPHAS)
    D = {}
    AKEY = gen_dict_key(3, 2, 0)
    D[AKEY] = ALPHAS
