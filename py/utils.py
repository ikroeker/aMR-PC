import numpy as np
import math
from scipy.special import comb

def genMultiIdx(No,dim):
    """ 
    generates mulit-indices of multi-variate polynomial base
    uses graded lexicographic ordering (p. 156, Sullivan)
    """
    P=comb(No+dim,dim)
    Alphas=np.zeros((P,dim))
    tA=zeros(dim)
    p=0
    l=2
    pmax=(No+1)**dim
    for p in range(pmax):
        rp=p
        for d in range(dim):
            md=(No+1)**(dim-d)
            val=rp//md
            tA[d]=val
            rp=rp-val*md
        if sum(tA)<=No:
            Alphas[l,:]=tA
            l=l+1
    return Alphas

if __name__=="__main__":
    print("test mvtools.py")
