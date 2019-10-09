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
