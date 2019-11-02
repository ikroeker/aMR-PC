import numpy as np
import context
import aMRPC.wavetools as wt

Nr=4
P=5
tol=1e-9
qdeg=-1

def genWV(P):
    wv=wt.WaveTools(P,qdeg)
    wv.genWVlets()
    return wv

def test_quad_one():
    """delta_ii test for  multi-wavelets on [0,1] """
    wv=genWV(P)
    for i in range(P):
        j=i
        q=np.dot((wv.fpsi(i,wv.roots)* wv.fpsi(j,wv.roots)) , wv.weights)
        assert abs(q-1)<tol
    
    
def test_quad_zero_one():
    """ delta_ij test for [0,1]-multi-wavelets"""
    wv=genWV(P)
    for i in range(P):
        for j in range (P):
            q=np.dot((wv.fpsi(i,wv.roots)* wv.fpsi(j,wv.roots)) , wv.weights)
            if i==j:
                assert abs(q-1)<tol
            else:
                assert abs(q)<tol

def test_pols():
    """ <x^i, \psi_j(x)>=0 test for [0,1]-multi-wavelets"""
    wv=genWV(P)
    x=wv.roots
    for i in range(P):
        for j in range(P):
            q = (x**i)*wv.fpsi(j,x) @ wv.weights
            #print(i,j,q)
            assert abs(q)<tol

def test_quad_MR():
    """ delta_ij test for rescaled-multi-wavelets"""
    wv=genWV(P)
    for aNr in range(Nr+1):
        for Nri in range(2**aNr):
            x=wv.rescY(wv.roots,aNr,Nri)
            for i in range(P):
                for j in range(P):
                    q= (wv.rfpsi(x,i,aNr,Nri)*wv.rfpsi(x,j,aNr,Nri)*wv.rqCf(aNr))@wv.weights
                    if i==j:
                        assert abs(q-1)<tol
                    else:
                        assert abs(q)<tol
    
def test_pols_MR():
    """ <x^i,psi_j>=0 test for rescaled-multi-wavelets"""
    wv=genWV(P)
    for aNr in range(Nr+1):
        for Nri in range(2**aNr):
            x=wv.rescY(wv.roots,aNr,Nri)
            for i in range(P):
                for j in range(P):
                    q= ((x**i)*wv.rfpsi(x,j,aNr,Nri)*wv.rqCf(aNr))@wv.weights
                    assert abs(q)<tol
