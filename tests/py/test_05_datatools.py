import numpy as np
import pandas as pd
from scipy.special import comb
import context
import aMRPC.iodata as iod
import aMRPC.datatools as dt
import aMRPC.polytools as pt
import aMRPC.utils as u
import aMRPC.wavetools as wt

iod.inDir = './tests/data'
iod.outDir = iod.inDir
fname = 'InputParameters.txt'
Nr = 2
No = 2
srcs = [0, 1, 2, 3]
srcs = [0, 2, 3]
method = 0
tol = 1e-6 * 10**Nr
eps = 0.01 # multi-wavelet threshold
#eps = 0.005 # multi-wavelet threshold

NrRange = np.arange(Nr+1)
dim = len(srcs)

def load():
    return iod.loadEvalPoints(fname)
    

def test_load():
    """ test if data are correct loaded, dataset has 10k lines """
    data = load()
    n,m = data.shape
    assert n == 10000


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
    
def test_u_nri_ranges():
    """ 
    tests both utils.genNriRange functions
    """
    dataframe = load()
    myNrRange = [Nr]
    # Generate Hankel matrices
    Hdict = dt.genHankel(dataframe, srcs, NrRange, No)
    # Roots and Weights
    R, W = dt.genRootsWeights(Hdict, method)
    # Generates dictionary of MR-elements bounds
    NRBdict = dt.genNrRangeBds(dataframe, srcs, myNrRange)
    # get roots and weights for the output
    mkLst = dt.genMkeyList(NRBdict, srcs)
    mkey_set = set(mkLst)
    assert(len(mkey_set) > 0)
    aNrs = Nr*np.ones(dim) 
    Nri_arr, Nri_cnt = u.genNriRange(aNrs)
    Nri_arr_set, Nri_cnt_set = u.genNriRange_4mkset(mkey_set, dim)
    assert(Nri_cnt == Nri_cnt_set)
    sum_nri = np.sum(Nri_arr, axis=0)
    sum_nri_set = np.sum(Nri_arr_set, axis=0)
    for d in range(dim):
        assert(sum_nri[d] == sum_nri_set[d])
        
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

def testPolOnSamples():
    """ tests genPolOnSamplesArr(...)"""
    dataframe = load()
    myNrRange = [Nr]
    Hdict = dt.genHankel(dataframe,srcs,NrRange,No)
    R,W = dt.genRootsWeights(Hdict,method)
    PCdict = dt.genPCs(Hdict,method)
    nPCdict = dt.genNPCs(PCdict,R,W)
    Alphas = u.genMultiIdx(No,dim)
    P = int(comb(No+dim,dim))
    # Generates dictionary of MR-elements bounds
    NRBdict = dt.genNrRangeBds(dataframe,srcs,myNrRange)
    # get roots and weights for the output
    mkLst = dt.genMkeyList(NRBdict,srcs)
    tR, tW, mkLstLong = dt.getRW4mKey(mkLst,R,W)
    sid2mk, mk2sid = dt.genMkeySidRel(tR,mkLst,NRBdict)
    polVals = dt.genPolOnSamplesArr(tR,nPCdict,Alphas,mk2sid)
    Gws = np.prod(tW,axis=1)
    n = tR.shape[0]
    for mk in mk2sid:
        sids = mk2sid[mk]
        for p in range(P):
            pV = polVals[p,sids]
            for q in range(P):
                qV = polVals[q,sids]
                p_pq = np.inner(pV*qV,Gws[sids])
                if p == q:
                    assert(abs(1-p_pq)<tol)
                else:
                    assert(abs(p_pq)<tol)
    
    
def test_cmpRescCf():
    """ tests sum cfs =1 """
    dataframe = load()
    myNrRange = [Nr]
    NRBdict = dt.genNrRangeBds(dataframe, srcs, myNrRange)
    mkLst = dt.genMkeyList(NRBdict, srcs)
    sum = 0
    #dim = len(srcs)
    assert(abs(dt.cmpRescCf(mkLst[0])-dt.cmpRescCfL([Nr]*dim)) < tol)
    rCdict = dt.genRCfDict(mkLst)
    for mk in mkLst:
        cf = dt.cmpRescCf(mk)
        assert(cf == rCdict[mk])
        sum += cf
    assert(abs(sum-1) < tol)

def test_wavelet_adapted():
    """ 
    generates wavelet adapted multi-element dataset
    test if each sid in one multi-element only
    and all sid's are contained 
    """
    dataframe = load()
    minNr = 2
    maxNr = Nr
    minNo = 1
    maxNo = No
    for nr in range(minNr, maxNr+1):
        nrs = nr*np.ones(dim)
        myNrRange = np.arange(nr+1)
        NRBdict = dt.genNrRangeBds(dataframe, srcs, myNrRange)
        # multi-key list
        #mkLst = dt.genMkeyList(NRBdict, srcs)
        # rescaling coefficients for proj-> Nr=0
        #rCdict = dt.genRCfDict(mkLst)
        for no in range(minNo, maxNo+1):
            # generate roots, weights and also polynomials
            # compare with Rooots-n-Weights
            # Generate Hankel matrices
            Hdict = dt.genHankel(dataframe, srcs, myNrRange, no)
            # Roots and Weights
            R, W = dt.genRootsWeights(Hdict, method)
            # (monic) orthogonal polynomial coefficients
            PCdict = dt.genPCs(Hdict, method)
            # normed orthogonal polynomials
            nPCdict = dt.genNPCs(PCdict, R, W)

            # generate wavelets and compute details
            P = no+1
            wv = wt.WaveTools(P)
            wv.genWVlets()
            # compute quantiles on roots on rescaled (0,1)-Legendre polynomials
            Qdict = dt.genQuantDict(dataframe, srcs, myNrRange, wv)
            # details
            Dtdict = dt.genDetailDict(Qdict, wv)
            # stoch elements to keep (details >= threshold)
            Kdict = dt.markDict4keep(Dtdict, eps)
            # highest level
            topKeys = dt.getTopKeys(Kdict, srcs)
            # multi-keys generated from topKeys
            mkLst = dt.genMkeyList(topKeys, srcs)
            #print(topKeys)
            # get roots and weights for the output
            tR, tW, mkLstLong = dt.getRW4mKey(mkLst, R ,W)
            sid2mk, mk2sid = dt.genMkeySidRel(tR, mkLst, NRBdict)
            for sid, mk in sid2mk.items():
                assert(len(mk) == 1)
