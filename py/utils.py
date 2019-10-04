import numpy as np
import math
from scipy.special import comb

def genMultiIdx(No,dim):
    """ 
    generates mulit-indices of multi-variate polynomial base
    uses graded lexicographic ordering (p. 156, Sullivan)
    """
    P=comb(No+dim,dim)
    Alphas=np.zeros((int(P),dim))
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

if __name__=="__main__":
    print("test utils.py")
    No=3
    dim=2
    Alphas=genMultiIdx(No,dim)
    print(Alphas)
