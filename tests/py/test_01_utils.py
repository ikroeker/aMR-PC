import numpy as np

import context
import aMRPC.utils as u

def test_midx():
    No=3
    dim=2
    Alphas=u.genMultiIdx(No,dim)
    m,n=Alphas.shape
    fline=Alphas[0,:]
    lline=Alphas[m-1,:]
    assert sum(fline)==0
    assert sum(lline)==No
    
def test_multikes():
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
