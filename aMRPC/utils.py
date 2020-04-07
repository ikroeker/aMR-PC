"""
utils.py - provides functions and dictionary for index and key management
to handle multi-resolution

@author: kroeker
"""

import numpy as np
from scipy.special import comb
#import math

# Data format for roots, weights and details: DictName(Nr,aNr,Nri,src)
#ParPos={'Nr':0,'aNr':1,'Nri':2,'src':3}
ParPos = {'Nr':0, 'aNr':0, 'Nri':1, 'src':2}

def gen_multi_idx(n_o, dim):
    """
    generates mulit-indices of multi-variate polynomial base
    uses graded lexicographic ordering (p. 156, Sullivan)
    """
    p_cnt = comb(n_o+dim, dim)
    alphas = np.zeros((int(p_cnt), dim), dtype=int)
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
    return alphas

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
    nris = np.zeros((nri_cnt, dim), dtype=int)
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
    nri_cnt = len(mkey_set) # number of multi-keys
    pos = ParPos['Nri']
    assert nri_cnt > 0
    nris = np.zeros((nri_cnt, dim), dtype=int)
    cnt = 0 # counter
    for mkey in mkey_set:
        for d_i in range(dim):
            nris[cnt, d_i] = int(mkey[d_i][pos])
        cnt += 1
    assert cnt == nri_cnt
    return nris, nri_cnt

def midx4quad(ar_lens):
    """ generates indexes for eval. points etc. """
    n_lens = np.array(ar_lens)
    cols = len(ar_lens)
    lines = n_lens.prod()
    idx_mx = np.zeros((lines, cols), dtype=int)
    divs = np.zeros(cols)
    for col in range(cols):
        divs[col] = n_lens[0:col].prod()
    #print(divs)
    for l_idx in range(lines):
        for col in range(cols):
            val = (l_idx//divs[col] % n_lens[col])
            idx_mx[l_idx, col] = val
    return idx_mx


def gen_dict_key(anr, nri, src=None):
    """
    generates dictionary key for
    aNr - actually Nr, Nri , src
    according to ParPos
    """
    #chkNri = Nri < 2**aNr
    #assert(chkNri)
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
    key_list = [gen_dict_key(anrs[d], nris[d], srcs[d]) for d in range(dims)]
    return tuple(key_list)

def multi_key2srcs(mkey):
    """ generates srcs list from multi-key """
    src_pos = ParPos['src']
    return [c[src_pos] for c in mkey]

def multi_key_diff_srcs(mkey_one, mkey_two):
    mk_len = len(mkey_one)
    assert mk_len == len(mkey_two)
    #spos = ParPos['src']
    srcs = multi_key2srcs(mkey_one)
    diff = [srcs[src] for src in range(mk_len) if mkey_one[src] != mkey_two[src]]
    return diff

def get_multi_entry(adict, anrs, nris, srcs):
    """ returns dictionary entriy """
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

if __name__ == "__main__":
    print("test utils.py")
    NO = 3
    DIM = 2
    ALPHAS = gen_multi_idx(NO, DIM)
    print(ALPHAS)
    D = {}
    AKEY = gen_dict_key(3, 2, 0)
    D[AKEY] = ALPHAS
