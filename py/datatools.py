import pandas as pd
import numpy as np
import polytools as pt
import utils as u

def genHankel(dataframe,srcs,NrRange,No):
    """ generates Hankel matrixes for each Nri, writes in Hdict """
    Nr=max(NrRange)
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
                H=pt.Hankel(No,qdata)
                key=u.genDictKey(Nr,aNr,Nri,src)
                Hdict[key]=H
                #print(H)
    return Hdict
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
    b=(data>=qlb) & (data<=qrb)
    return b

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
    return func(roots)@weights

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
        return innerProdArr(F,G,R,W)
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

def innerProdArr(F,G,Roots,Weights):
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

def genRW4mkey(mKey,Roots,Weights):
    """ generates roots and weights arrays from dict's Roots and Weights for multikey mkey """
    cols=len(mKey)
    ls=[]
    for d in range(cols):
        key=mKey[d]
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


    
if  __name__=="__main__":
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
  
