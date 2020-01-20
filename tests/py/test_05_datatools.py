import numpy as np
import pandas as pd
from scipy.special import comb
import context
import aMRPC.iodata as iod
import aMRPC.datatools as dt
import aMRPC.polytools as pt
import aMRPC.utils as u

iod.inDir='./tests/data'
iod.outDir=iod.inDir
fname='InputParameters.txt'
Nr=2
No=2
srcs=[0, 1, 2, 3]
method=0
tol=1e-4

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
    P=int(comb(No+dim,dim))
    for Nra in NrRange:
        aNrs=Nra*np.ones(dim)
        NriRange,Lines=u.genNriRange(aNrs)
        for Nri in range(Lines):
            Nris=NriRange[Nri,:]
            mk=u.genMultiKey(aNrs,Nris,srcs)
            aR,aW=dt.genRW4mkey(mk,R,W)
            for p in range(P):
                pCfs=dt.PCfs4eval(nPCdict,mk,Alphas[p])
                F=lambda x:pt.PCeval(pCfs,x)
                for q in range(P):
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

                        
def test_sample2mkey():
    """ tests the point to multi-key relationship """
    dataframe=load()
    myNrRange=[Nr]
    # Generate Hankel matrices
    Hdict=dt.genHankel(dataframe,srcs,NrRange,No)
    # Roots and Weights
    R,W=dt.genRootsWeights(Hdict,method)
    # Generates dictionary of MR-elements bounds
    NRBdict=dt.genNrRangeBds(dataframe,srcs,myNrRange)
    # get roots and weights for the output
    mkLst=dt.genMkeyList(NRBdict,srcs)
    tR, tW, mkLstLong=dt.getRW4mKey(mkLst,R,W)
    n=len(mkLstLong)
    for i in range(n):
        mk=mkLstLong[i]
        sample=tR[i,:]
        mks=dt.sample2mKey(sample,mkLst,NRBdict,True)
        assert(len(mks)==1)
        assert(mk==mks[0])
    

def test_MkeySidRel():
    """ tests the point to multi-key relationship in dictionaries"""
    dataframe=load()
    myNrRange=[Nr]
    # Generate Hankel matrices
    Hdict=dt.genHankel(dataframe,srcs,NrRange,No)
    # Roots and Weights
    R,W=dt.genRootsWeights(Hdict,method)
    # Generates dictionary of MR-elements bounds
    NRBdict=dt.genNrRangeBds(dataframe,srcs,myNrRange)
    # get roots and weights for the output
    mkLst=dt.genMkeyList(NRBdict,srcs)
    tR, tW, mkLstLong=dt.getRW4mKey(mkLst,R,W)
    sid2mk, mk2sid=dt.genMkeySidRel(tR,mkLst,NRBdict)
    n=len(mkLstLong)
    for sid in range(n):
        mk=mkLstLong[sid]
        mkl=sid2mk[sid]
        assert(mk==mkl[0])
        sids=mk2sid[mk]
        assert(sid in sids)
        
def test_cmpRescCf():
    """ tests sum cfs =1 """
    dataframe=load()
    myNrRange=[Nr]
    NRBdict=dt.genNrRangeBds(dataframe,srcs,myNrRange)
    mkLst=dt.genMkeyList(NRBdict,srcs)
    sum=0
    dim=len(srcs)
    assert(abs(dt.cmpRescCf(mkLst[0])-dt.cmpRescCfL([Nr]*dim))<tol)
    rCdict=dt.genRCfDict(mkLst)
    for mk in mkLst:
        cf=dt.cmpRescCf(mk)
        assert(cf == rCdict[mk])
        sum+=cf
    assert(abs(sum-1)<tol)
