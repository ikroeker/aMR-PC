import pandas as pd
import numpy as np
from . import polytools as pt
from . import utils as u
from . import wavetools as wt

def genHankel(dataframe,srcs,NrRange,No):
    """ generates Hankel matrixes for each Nri, writes in Hdict """
    #Nr=max(NrRange)
    Hdict={}
    for src in srcs:
        data=dataframe[src]
        for aNr in NrRange:
            for Nri in range(2**aNr):
                lb,rb=cmpLRB(aNr,Nri)
                qlb=data.quantile(lb)
                qrb=data.quantile(rb)
                bd=cmpQuantDomain(data,qlb,qrb)
                qdata=data[bd]
                H=pt.Hankel(No+1,qdata)
                key=u.genDictKey(aNr,Nri,src)
                Hdict[key]=H
                #print(H)
    return Hdict

def genNrRangeBds(dataframe,srcs,NrRange):
    """ generates dictionary with boundaries of MR-elements """
    NRBdict={}
    for src in srcs:
        data=dataframe[src]
        for aNr in NrRange:
            for Nri in range(2**aNr):
                lb,rb=cmpLRB(aNr,Nri)
                qlb=data.quantile(lb)
                qrb=data.quantile(rb)
                key=u.genDictKey(aNr,Nri,src)
                NRBdict[key]=(qlb,qrb)
    return NRBdict

def genRootsWeights(Hdict,method):
    """ 
    generates dictionaries with roots and weights
    generated by methods 0- Gautschi / 1-Karniadakis & Kirby
    """
    Roots={}
    Weights={}
    for key in Hdict:
        r,w=pt.genRW(Hdict[key],method)
        Roots[key]=r
        Weights[key]=w
    return Roots,Weights

def cmpLRB(Nr,Nri):
    """ computes left and right bounds for use in dataframe.quantile() """
    cf=2**(Nr)
    lb=Nri/cf
    rb=(Nri+1)/cf
    return lb, rb

def cmpQuantDomain(data,qlb,qrb):
    """ generates bool array with 1 for x in [qlb,qrb], 0 else """
    b = (data>=qlb) & (data<=qrb)
    return b

def cmpMVQuantDomain(Roots,NRBdict,Nrs,Nris,cols):
    """ 
    generates bool array with 1 for r inside of 
    [a_0,b_0]x..x[a_d,b_d], 0 else
    """
    n=Roots.shape[0]
    ndim=len(Nrs)
    assert ndim == len(Nris)
    B=np.ones(n,dtype=bool)
    for d,c in enumerate(cols):
        key=u.genDictKey(Nrs[d],Nris[d],c)
        qlb,qrb=NRBdict[key]
        B=B & cmpQuantDomain(Roots[c],qlb,qrb)
    return B

def cmpMVQuantDomainMK(Roots,NRBdict,mkey):
    """ 
    generates bool array with 1 for r inside of 
    [a_0,b_0]x..x[a_d,b_d], 0 else
    for given multikey mkey
    """
    n, sdim = Roots.shape
    assert(sdim == len(mkey))
    B = np.ones(n, dtype=bool)
    for d in range(sdim):
        key = mkey[d]
        qlb, qrb = NRBdict[key]
        B = B & cmpQuantDomain(Roots[:,d], qlb, qrb)
    return B

def genPCs(Hdict,method):
    """
    generated dictionaries with matrices of monic orthogonal
    polynomials, method 0: Gautschi style, 1: Sergey Style.
    """
    Cfs={}
    for key in Hdict:
        Cfs[key]=pt.genPCmx(Hdict[key],method)
    return Cfs

def genNPCs(PCDict,Roots,Weights):
    """ generates dictionary with coeficients of orthonormal polynomials """
    nCfs={}
    for key in PCDict:
        nCfs[key]=pt.genNPCmx(PCDict[key],Roots[key],Weights[key])
    return nCfs

def PCfs4eval(PCdict,mkey,alpha):
    """ provides PC-Cfs for multi-polynomial with degrees in alpha """
    mdeg=max(alpha)
    alen=len(alpha)
    assert(alen == len(mkey))
    RCfs=np.zeros((alen,mdeg+1))
    for d in range(alen):
        ad=alpha[d]
        Cfs=PCdict[mkey[d]]
        RCfs[d,:]=Cfs[ad,0:mdeg+1]
    return RCfs
    
def GaussQuad(func,roots,weights):
    """ Gauss quadrature with roots and weights """
    assert(len(roots)==len(weights))
    return np.inner(func(roots),weights)

def innerProd(f,g,roots,weights):
    """ inner product <f,g,>, computed by Gauss quadrature """
    return GaussQuad(lambda x: f(x)*g(x),roots,weights)

def GaussQuadIdx(F,multiKey,Roots,Weights):
    """ multi-dimensional Gauss quad on multiKey of 
    f_0(x_0)*...*f_dim-1(x_dim-1), Func=(f_0,...,f_dim-1)
    """
    dim=len(multiKey)
    R,W=genRW4mkey(multiKey,Roots,Weights)
    if type(F)==tuple:
        assert(dim==len(FunTup))
        return GaussQuadArr(F,R,W)
    else:
        return GaussQuadFtk(F,R,W)
    
def innerProdMultiIdx(F,G,multiKey,Roots,Weights):
    """ <F,G>, for for multi-index multiKey """
    dim=len(multiKey)
    R,W=genRW4mkey(multiKey,Roots,Weights)
    assert(type(F)==type(G))
    tf=type(F)==tuple
    tg=type(G)==tuple
    if tf and tg:
        flen=len(F)
        assert(flen==len(G))
        assert(flen==dim)
        return innerProdTuples(F,G,R,W)
    else:
        return innerProdFct(F,G,R,W)

def GaussQuadArr(FunTup,Roots,Weights):
    dim=len(FunTup)
    evals=Roots.shape[0]
    Ba=dim==Roots.shape[1]
    Bb=Roots.size==Weights.size
    assert(a and b)
    S=0
    for l in range(evals):
        tmp=1
        for d in range(dim):
            key=(l,d)
            tmp=tmp*FunTup[d](Roots[key])*Weights[key]
        S+=tmp
    return S

def innerProdTuples(F,G,Roots,Weights):
    """ <F,G>, F,G are given by tuples """
    dim=len(F)
    evals=Roots.shape[0]
    a=dim==len(G)
    b=Roots.size == Weights.size
    c=dim==Roots.shape[1]
    assert(a and b and c)
    S=0
    for l in range(evals):
        tmp=1
        for d in range(dim):
            key=(l,d)
            x=Roots[key]
            tmp=tmp*F[d](x)*G[d](x)*Weights[key]
        S+=tmp
    return S

def GaussQuadFct(F,Roots,Weights):
    assert(Roots.shape == Weights.shape)
    A=F(Roots)*Weights
    P=np.prod(A,axis=1)
    return sum(P)

def innerProdFct(F,G,Roots,Weights):
    assert(Roots.shape == Weights.shape)
    A= F(Roots)*G(Roots)*Weights
    P=np.prod(A,axis=1)
    return sum(P)

def innerProdArrFct(Arr,F,Roots,Weights,srcs):
    """ inner product of data in Arr and F(Roots) weighted with Weigths """
    assert(Roots.shape==Weights.shape)
    A=F(Roots[srcs])*Weights[srcs]
    P=np.prod(A,axis=1)
    return np.inner(Arr,P)
     

def genRW4mkey(mKey,Roots,Weights):
    """ generates roots and weights arrays from dict's Roots and Weights for multikey mkey """
    cols=len(mKey)
    ls=[]
    for d in range(cols):
        key=mKey[d]
        #print("k->r",key,Roots[key])
        lk=len(Roots[key])
        ls.append(lk)
    lens=np.array(ls)
    lines=lens.prod()
    I=u.mIdx4quad(lens)
    Rs=np.zeros((lines,cols))
    Ws=np.zeros([lines,cols])
    for c in range(cols):
        key=mKey[c]
        r=Roots[key]
        w=Weights[key]
        idx=I[:,c]
        #print(idx,r)
        rs=r[idx]
        Rs[:,c]=rs
        Ws[:,c]=w[I[:,c]]
    return Rs, Ws

def getRW4Nrs(Nrs,srcs,Roots,Weights):
    """ generates eval. points and weights for an Nr level """
    dim=len(Nrs)
    Nris=np.zeros(dim)
    NriCnt=np.zeros(dim,dtype=int)
    divs=np.zeros(dim)
    R=np.array([])
    W=np.array([])
    mkLstLong=[] #  multi-key in order of apperance
    
    for d in range(dim):
        aNr=Nrs[d]
        NriCnt[d]=2**aNr
        divs[d]=np.prod(NriCnt[0:d])
    tNriCnt=np.prod(NriCnt)
    
    for l in range(tNriCnt):
        for d in range(dim):
            v=(l//divs[d] % NriCnt[d])
            Nris[d]=v
        mkey=u.genMultiKey(Nrs,Nris,srcs)
        r,w=genRW4mkey(mkey,Roots,Weights)
        #r=np.reshape(r,(-1,dim))
        mkLstLong=mkLstLong+[mkey for c in range(len(r))]
        if len(R)==0:
            R=r
            W=w
        else:
            #print(l,":",len(R))
            R=np.concatenate([R,r],axis=0)
            W=np.concatenate([W,w],axis=0)
    return R,W,mkLstLong

def genQuantDict(dataframe,srcs,NrRange,wvt):
    """ 
    generates dictionary of quantiles on roots for each Nri and Nr in NrRange
    accorting to roots stored in already initialized object of wavetools wt
    using data in dataframe for columns in srcs
    """
    Qdict={}
    
    for src in srcs:
        data=dataframe[src]
        for aNr in NrRange:
            for Nri in range(2**aNr):
                quants=wvt.cmpDataOnRoots(data,aNr,Nri)
                key=u.genDictKey(aNr,Nri,src)
                Qdict[key]=quants
    return Qdict

def genDetailDict(Qdict,wvt,dicts=0):
    """ 
    generates dictionary of Details on roots for each set of quantiles
    stored in Qdict, using initalized wavetool wvt
    dits=0: sum(abs(details)), 1: lDetails only, 2 both
    """
    DetDict={}
    lDetDict={}
    s= dicts==0 or dicts==2
    l= dicts>=1
    for key, data in Qdict.items():
        Nr=key[u.ParPos['Nr']]
        lDetails=wvt.cmpDetails(data)
        if l:
            lDetDict[key]=lDetails
        if s:
            DetDict[key]=sum(abs(lDetails))
    if dicts==0:
        return DetDict
    elif dicts==1:
        return lDetDict
    else:
        return DetDict, lDetDict

def markDict4keep(Ddict,thres):
    """ marks the details>= threshold for keep """
    Kdict={}
    for key,data in Ddict.items():
        b=data>=thres
        Kdict[key]=b
    return Kdict

def getTrueKids(Kdict,key):
    """
    checks leafs of the tree bottom ab, leafs only highest "True"-level on True
    """
    ret=0
    if Kdict[key]:
        ret=True
        Nri=key[u.ParPos['Nri']]
        Nr=key[u.ParPos['Nr']]
        src=key[u.ParPos['src']]
        lNri=2*Nri
        rNri=lNri+1
        lkey=u.genDictKey(Nr+1,lNri,src)
        rkey=u.genDictKey(Nr+1,rNri,src)
        lex= lkey in Kdict.keys()
        rex= rkey in Kdict.keys()
        if lex and rex:
            l= getTrueKids(Kdict,lkey)
            r=getTrueKids(Kdict,rkey)
            kids= l or r
            if kids:
                Kdict[key]=False
                if not l:
                    Kdict[lkey]=True
                    ret=ret+1
                if not r:
                    Kdict[rkey]=True
                    ret=ret+1
    return ret
        
def getTopKeys(Kdict,srcs):
    """ returns set with top level (True) keys only (bottom up)"""
    tKeys=Kdict.copy()
    for src in srcs:
        nkey=u.genDictKey(0,0,src)
        if nkey in tKeys.keys():
            cnt=getTrueKids(tKeys,nkey)
            if cnt==0:
                tKeys[nkey]=True # set root node to True if no leafs are selected
    return tKeys

def genMkeyList(Kdict,srcs):
    """ generates array of multi-keys from the dictionary Kdict """
    isrcs=u.invSrcArr(srcs)
    kLst=[[] for s in isrcs]
    srclen=len(isrcs)
    sidx=u.ParPos['src']
    for key,chk in Kdict.items():
        if chk:
            idx=key[sidx]
            kLst[isrcs[idx]].append(key)
    alen=[len(c) for c in kLst]
    I=u.mIdx4quad(alen)
    ilen=I.shape[0]    
    mkLst=[ tuple([kLst[c][I[i,c]] for c in range(srclen)]) for i in range(ilen)]
    # required also for 1-dim case, to generate multikey -> tuple(tuple)
    return mkLst

def genMkeySidRel(samples, mkLst, NRBdict, all_mk=False):
    """
    generates long sample->[multi-key ]
    multi-key -> np.array([sample id]) dictionaries
    """
    sample_cnt, ndim = samples.shape
    sids = np.arange(sample_cnt)
    sid2mk = {}
    mk2sids = {}
    for mkey in mkLst:
        B = cmpMVQuantDomainMK(samples, NRBdict, mkey)
        mk2sids[mkey] = sids[B]
        for sid in mk2sids[mkey]:
            if sid in sid2mk:
                sid2mk[sid] += mkey
            else:
                sid2mk[sid] = [mkey]
    return sid2mk, mk2sids
    
    
def getRW4mKey(mkLst,Roots,Weights):
    """ generates eval. points and weights  np.arrays and 
    (point number)->mkey list   for multi-keys in mkArr list """
    tcnt=len(mkLst)
    R=np.array([])
    W=np.array([])
    mkLstLong=[] #  multi-key in order of apperance
    Points4mk=0
    for mkey in mkLst:
        r,w=genRW4mkey(mkey,Roots,Weights)
        Points4mk=len(r)
        mkLstLong=mkLstLong+[mkey for c in range(Points4mk)]
        if len(R)==0:
            R=r
            W=w
        else:
            R=np.concatenate([R,r],axis=0)
            W=np.concatenate([W,w],axis=0)
    return R,W,mkLstLong

def sample2mKey(sample, mkLst, NRBdict, all=False):
    """ finds first, all multi-key in NR-Bounds dictrionary corresponding to the 
    multi-element containing the sample
    """
    ndim = len(sample)
    smkList = []
    for mk in mkLst:
        chk = True
        for d in range(ndim):
            qlb, qrb = NRBdict[mk[d]]
            chk = chk & cmpQuantDomain(sample[d], qlb, qrb)
        if chk:
            if all:
                smkList += [mk]
            else:
                smkList = [mk]
    return smkList

def cmpRescCfL(aNRList):
    """ 
    computes rescaling cfs c=<phi^Nr_l,0,phi^0_0,0>. 
    Is relevant for computing Exp. / Var. from coefficients only
    for aNRlist [aNr_0,...,aNr_d]
    """
    cf=1
    for aNr in aNRList:
        cf/=2**(aNr)
    return cf

def cmpRescCf(mKey):
    """ 
    computes rescaling coeficients c=<phi^Nr_l,0,phi^0_0,0>. 
    Is relevant for computing Exp. / Var. from coefficients only
    for multi-key mKey
    """
    dim = len(mKey)
    NrPos = u.ParPos['aNr']
    cf = 1
    for d in range(dim):
        key = mKey[d]
        cf/=2**(key[NrPos])
    return cf

def genRCfDict(mkList):
    """
    generates dictionary with rescaling coefficients for ech
    multi-key in mkList [(mk),...]
    """
    rCfdict = {}
    for mKey in mkList:
        rCfdict[mKey] = cmpRescCf(mKey)
    return rCfdict

def genPolOnSamplesArr(samples,nPCdict,Alphas,mk2sid):
    """
    generates np.array with pol. vals for each sample and pol. degree
    samples : samples for evaluation (evtl. samples[srcs] )
    nPCdict: pol. coeff. dictionary
    Alphas: matrix of multiindexes representing pol. degrees
    mk2sid: multi-key -> sample id's (sid lists should be disjoint)
    return: pol_vals[sample id] = [pol_i:p_i(x_0),...p_i(x_end)]
    """
    n,m = samples.shape
    P = Alphas.shape[0]
    pol_vals = np.zeros((P,n))

    for mk in mk2sid:
        sids = mk2sid[mk]
        for p in range(P):
            pCfs = PCfs4eval(nPCdict,mk,Alphas[p])
            pvals = pt.PCeval(pCfs,samples[sids,:])
            pol_vals[p,sids] = np.prod(pvals,axis=1)
    return pol_vals
            
    
def main():
    """ some tests """
    # data location
    url='../data/InputParameters.txt'

    # load data
    dataframe=pd.read_csv(url,header=None,sep='\s+ ',engine='python')
    Nr=2
    NrRange=np.arange(Nr+1)
    No=2
    srcs=np.arange(0,1)
    method=0
    Hdict=genHankel(dataframe,srcs,NrRange,No)
    print(Hdict)
    # further with test004
    r,w=genRootsWeights(Hdict,method)
    print(r,w)

