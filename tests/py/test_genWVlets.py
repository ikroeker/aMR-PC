import numpy as np
import context
import aMRPC.wavetools as wt

def genWV(P=4,qdeg=-1):
    wv=wt.WaveTools(P,qdeg)
    wv.genWVlets()
    return wv

def test_quad_one():
    P=4
    tol=1e-8
    wv=genWV(P)
    i=1
    j=i
    for i in range(P):
        j=i
        q=np.dot((wv.fpsi(i,wv.roots)* wv.fpsi(j,wv.roots)) , wv.weights)
        assert abs(q-1)<tol
    
    
    
