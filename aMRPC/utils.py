import numpy as np
from scipy.special import comb
#import math

# Data format for roots, weights and details: DictName(Nr,aNr,Nri,src)
#ParPos={'Nr':0,'aNr':1,'Nri':2,'src':3}
ParPos={'Nr':0,'aNr':0,'Nri':1,'src':2}

def genMultiIdx(No, dim):
    """ 
    generates mulit-indices of multi-variate polynomial base
    uses graded lexicographic ordering (p. 156, Sullivan)
    """
    P = comb(No+dim, dim)
    Alphas = np.zeros((int(P), dim), dtype=int)
    tA = np.zeros(dim)
    l = 1
    pmax = (No+1)**dim
    for p in range(1, pmax):
        rp=p
        for d in range(dim):
            md = (No+1)**(dim-d-1)
            val = rp//md
            tA[d] = val
            rp = rp-val*md
        if sum(tA) <= No:
            Alphas[l,:] = tA
            l = l+1
    return Alphas

def genNriRange(Nrs):
    """ 
    generates an array with Nri-entries
    input:
    Nrs -- list of Nr for each dim, e.g. [Nr, Nr] 

    return:
    Nris -- np.array with Nri-(integer) entries
    NriCnt -- length of the array
    """
    dim = len(Nrs)
    NriCnts = np.zeros(dim);
    divs = np.zeros(dim)
    NriCnt = 1
    for d in range(dim):
        NriCnts[d] = 2**(Nrs[d])
        divs[d] = NriCnts[0:d].prod()
    NriCnt = int(NriCnts.prod())        
    Nris = np.zeros((NriCnt, dim), dtype=int)
    for nri in range(NriCnt):
        for d in range(dim):
            v = (nri//divs[d] % NriCnts[d])
            Nris[nri, d] = v
    return Nris, NriCnt

def genNriRange_mkset(mkey_set, dim):
    """
    generates an np.array with Nri-entries
    input: 
    mkey_set -- set of multi-keys
    dim -- dimension, also number of keys in each mkey
    return:
    Nris -- np.array with Nri-(integer) entries
    NriCnt -- length of the array
    """
    NriCnt = len(mkey_set) # number of multi-keys
    POS = ParPos['Nri']
    assert (NriCnt > 0)
    Nris = np.zeros((NriCnt, dim), dtype=int)
    cnt = 0 # counter
    for mkey in mkey_set:
        for d in range(dim):
            Nris[cnt, d] = mkey[d][POS]
        cnt += 1
    assert(cnt == NriCnt)
    return Nris, NriCnt
    
def mIdx4quad(arLens):
    """ generates indexes for eval. points etc. """
    nLens = np.array(arLens)
    cols = len(arLens)
    lines = nLens.prod()
    I = np.zeros((lines, cols), dtype=int)
    divs = np.zeros(cols)
    for c in range(cols):
        divs[c] = nLens[0:c].prod()
    #print(divs)
    for l in range(lines):
        for c in range(cols):
            v = (l//divs[c] % nLens[c])
            I[l, c] = v
    return I


def genDictKey(aNr, Nri, src=None):
    """ 
    generates dictionary key for 
    aNr - actually Nr, Nri , src
    according to ParPos
    """
    #chkNri = Nri < 2**aNr
    #assert(chkNri)
    if src is None:
        return (aNr, Nri)
    else:
        return (aNr, Nri, src)
    
def getDictEntry(Dict, aNr, Nri, src=-1):
    """ returns dictionary entry """
    key = genDictKey(aNr, Nri, src)
    return Dict[key]

def genMultiKey(aNrs, Nris, srcs):
    """
    generates a tuple of tuples that will be used as a key
    srcs - source numbers
    aNrs - Nr- levels for each entree in srcs
    Nris - Nr indices for each entree in src
    """
    dims = len(srcs)
    keyList = [genDictKey(aNrs[d], Nris[d], srcs[d]) for d in range(dims)]
    #for d in range(dims):
    #    keyList.append(genDictKey(aNrs[d],Nris[d],srcs[d]))
    return tuple(keyList)

def MultiKey2srcs(mk):
    """ generates srcs list from multi-key """
    sPos = ParPos['src']
    return [c[sPos] for c in mk]

def getMultiEntry(Dict, aNrs, Nris, srcs):
    """ returns dictionary entriy """
    key = genMultiKey(aNrs, Nris, srcs)
    return Dict[key]

def chooseCols(A, ccols, zeroCols=-1):
    """ picks the ccols columns from A """
    ret = A[:, ccols]
    if zeroCols != -1:
        ret[:, zeroCols] = 0
    return ret

def invSrcArr(srcs):
    """ generates dictionary with positions of srcs """
    isrc = {}
    i = 0
    for s in srcs:
       isrc[s] = i
       i = i+1
    return isrc

if __name__=="__main__":
    print("test utils.py")
    No=3
    dim=2
    Alphas=genMultiIdx(No,dim)
    print(Alphas)
    D={}
    key=genDictKey(3,2,0)
    D[key]=Alphas

