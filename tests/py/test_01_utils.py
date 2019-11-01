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
    
