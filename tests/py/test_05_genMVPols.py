import numpy as np
import pandas as pd
import context
import aMRPC.iodata as iod
import aMRPC.datatools as dt
import aMRPC.polytools as pt
import aMRPC.utils as u

iod.inDir='./tests/data'
iod.outDir=iod.inDir
fname='InputParameters.txt'
Nr=4
No=3
srcs=[0, 1, 2, 3]
method=0
tol=1e-5

NrRange=np.arange(Nr+1)
dim=len(srcs)

def load():
    return iod.loadEvalPoints(fname)
    

def test_load():
    """ test if data are correct loaded, dataset has 10k lines """
    data=load()
    n,m=data.shape
    assert n==10000

def test_MultiIdx():
    """ checks if each No multi-index is unique"""
    Alphas=u.genMultiIdx(No,dim)
    for p in range(No+1):
        for q in range(No+1):
            if p!=q:
                chk=True
                for d in range(dim):
                    chk=chk and (Alphas[p,d]==Alphas[q,d])
                assert not chk

def test_innerProdFct():
    """ tests inner product for two polynomials as functions"""
    dataframe=load()
    Hdict=dt.genHankel(dataframe,srcs,NrRange,No)
    R,W=dt.genRootsWeights(Hdict,method)
    PCdict=dt.genPCs(Hdict,method)
    nPCdict=dt.genNPCs(PCdict,R,W)
    Alphas=u.genMultiIdx(No,dim)
    for Nra in NrRange:
        aNrs=Nra*np.ones(dim)
        for Nri in range(2**Nra):
            Nris=Nri*np.ones(dim)
            mk=u.genMultiKey(aNrs,Nris,srcs)
            aR,aW=dt.genRW4mkey(mk,R,W)
            for p in range(No+1):
                pCfs=dt.PCfs4eval(nPCdict,mk,Alphas[p])
                F=lambda x:pt.PCeval(pCfs,x)
                for q in range(No+1):
                    qCfs=dt.PCfs4eval(nPCdict,mk,Alphas[q])
                    G=lambda x:pt.PCeval(qCfs,x)
                    q_pq=dt.innerProdFct(F,G,aR,aW)
                    q_mk=dt.innerProdMultiIdx(F,G,mk,R,W)
                    if p!=q:
                        assert abs(q_pq)< tol
                        assert abs(q_mk)< tol
                    else:
                        assert abs(1-q_pq)<tol
                        assert abs(1-q_mk)<tol



