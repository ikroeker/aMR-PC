import numpy as np

import context
import aMRPC.utils as u

No = 3
DIM = 3

def test_midx():
    dim = DIM
    Alphas=u.genMultiIdx(No, dim)
    m,n=Alphas.shape
    fline=Alphas[0,:]
    lline=Alphas[m-1,:]
    assert sum(fline)==0
    assert sum(lline)==No
    
def test_multikeys():
    srcs = [0,1,3]
    aNr = 2
    dim = len(srcs)
    aNrs = [aNr]*dim
    NriMax = 2**(aNr*dim)
    Nris,NriCnt = u.genNriRange(aNrs)
    assert(NriMax == NriCnt)
    for nri in Nris:
        mk = u.genMultiKey(aNrs,nri,srcs)
        nsrcs = u.MultiKey2srcs(mk)
        for d in range(dim):
            assert(srcs[d] == nsrcs[d])

def test_MultiIdx():
    """ checks if each No multi-index is unique"""
    dim = DIM
    Alphas = u.genMultiIdx(No, dim)
    for p in range(No+1):
        for q in range(No+1):
            if p != q:
                chk=True
                for d in range(dim):
                    chk = chk and (Alphas[p, d] == Alphas[q, d])
                assert not chk
