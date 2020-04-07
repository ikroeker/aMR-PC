import numpy as np

import context
import aMRPC.utils as u

NO = 3
NR = 3
DIM = 3

def test_midx():
    dim = DIM
    alphas = u.gen_multi_idx(NO, dim)
    m = alphas.shape[0]
    fline = alphas[0, :]
    lline = alphas[m-1, :]
    assert sum(fline) == 0
    assert sum(lline) == NO

def test_multikeys():
    srcs = [0, 1, 3]
    dim = len(srcs)
    anrs = [NR]*dim
    nri_max = 2**(NR*dim)
    nris, nri_cnt = u.gen_nri_range(anrs)
    assert nri_max == nri_cnt
    for nri in nris:
        mk = u.gen_multi_key(anrs, nri, srcs)
        nsrcs = u.multi_key2srcs(mk)
        for d in range(dim):
            assert srcs[d] == nsrcs[d]

def test_multiidx():
    """ checks if each No multi-index is unique"""
    alphas = u.gen_multi_idx(NO, DIM)
    for p in range(NO+1):
        for q in range(NO+1):
            if p != q:
                chk = True
                for d in range(DIM):
                    chk = chk and (alphas[p, d] == alphas[q, d])
                assert not chk

def test_mk_diff():
    "checks multi-key diff fct"
    srcs = [0, 1, 3]
    dim = len(srcs)
    anrs = [NR]*dim
    nri_max = 2**(NR*dim)
    nris, nri_cnt = u.gen_nri_range(anrs)
    assert nri_max == nri_cnt
    for nri_a in nris:
        mk_a = u.gen_multi_key(anrs, nri_a, srcs)
        for nri_b in nris:
            mk_b = u.gen_multi_key(anrs, nri_b, srcs)
            src_list = u.multi_key_diff_srcs(mk_a, mk_b)
            if mk_a == mk_b:
                assert not src_list
            else:
                assert src_list
