import numpy as np
#import context
import aMRPC.iodata as iod

def test_txt():
    fname='pts.txt'
    iod.inDir='./tests/data'
    iod.outDir=iod.inDir
    n=100
    m=4
    rpts=np.random.randn(n,m)
    n,m=rpts.shape
    iod.writeEvalPoints(rpts,fname)
    data=iod.loadEvalPoints(fname)
    k,l=data.shape
    assert n==k
    assert m==l

def test_dict():
    dir='./tests/data'
    iod.outDir=dir
    fname="tst.p"
    Dict={'a':34,'b':55}
    iod.storeDataDict(Dict,fname)
    nDict=iod.loadDataDict(fname,dir)
    assert Dict['a']==nDict['a']
