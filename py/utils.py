import numpy as np
#import math
from scipy.special import comb
import pickle

def genMultiIdx(No,dim):
    """ 
    generates mulit-indices of multi-variate polynomial base
    uses graded lexicographic ordering (p. 156, Sullivan)
    """
    P=comb(No+dim,dim)
    Alphas=np.zeros((int(P),dim),dtype=int)
    tA=np.zeros(dim)
    l=1
    pmax=(No+1)**dim
    for p in range(1,pmax):
        rp=p
        for d in range(dim):
            md=(No+1)**(dim-d-1)
            val=rp//md
            tA[d]=val
            rp=rp-val*md
        if sum(tA)<=No:
            Alphas[l,:]=tA
            l=l+1
    return Alphas

def mIdx4quad(arLens):
    nLens=np.array(arLens)
    cols=len(arLens)
    lines=nLens.prod()
    I=np.zeros((lines,cols),dtype=int)
    divs=np.zeros(cols)
    for c in range(cols):
        divs[c]=nLens[0:c].prod()
    #print(divs)
    for l in range(lines):
        for c in range(cols):
            v=(l//divs[c] % nLens[c])
            I[l,c]=v
    return I

# Data format for roots, weights and details: DictName(Nr,aNr,Nri,src)
ParPos={'Nr':0,'aNr':1,'Nri':2,'src':3}

def genDictKey(Nr,aNr,Nri,src=-1):
    """ 
    generates dictionary key for 
    Nr - max Nr, aNr - actually Nr, Nri , src
    according to ParPos
    """
    chkNr=aNr<=Nr
    chkNri=Nri<2**aNr
    assert(chkNr and chkNri)
    if src==-1:
        return (Nr,aNr,Nri)
    else:
        return (Nr,aNr,Nri,src)
    
def getDictEntry(Dict,Nr,aNr,Nri,src=-1):
    """ returns dictionary entry """
    key=genDictKey(Nr,aNr,Nri,src)
    return Dict[key]

def genMultiKey(Nrs,aNrs,Nris,srcs):
    """
    generates a tuple of tuples that will be used as a key
    """
    dims=len(srcs)
    keyList=[]
    for d in range(dims):
        keyList.append(genDictKey(Nrs[d],aNrs[d],Nris[d],srcs[d]))
    return tuple(keyList)

def getMultiEntry(Dict,Nrs,aNrs,Nris,srcs):
    """ returns dictionary entriy """
    key=genMultiKes(Nrs,aNrs,Nris,srcs)
    return Dict[key]

def storeDataDict(Dict,fname,dir="../data"):
    """ stores dictionary in {dir}/{fname}.p using pickle """
    file=dir +"/" + fname + '.p'
    f=open(file,"wb")
    pickle.dump(Dict,f)
    f.close()

def loadDataDict(fname,dir="../data"):
    """ load picle stored data from {dir}/{fname}.p """
    file=dir +"/" + fname + '.p'
    f=open(file,"rb")
    Dict=pickle.load(f)
    f.close()
    return Dict

if __name__=="__main__":
    print("test utils.py")
    No=3
    dim=2
    Alphas=genMultiIdx(No,dim)
    print(Alphas)
    D={}
    key=genDictKey(3,2,0)
    D[key]=Alphas
    storeDataDict(D,"abc")
    A=loadDataDict("abc")
    print(A)

