import numpy as np
import context
import aMRPC.iodata as iod

def test_txt():
    fname='pts.txt'
    iod.inDir='./tests/data'
    iod.outDir=iod.inDir
    n=100
    m=4
    rpts=np.random.randn(n,m)
    n,m=rpts.shape
    iod.write_eval_points(rpts,fname)
    data=iod.load_eval_points(fname)
    k,l=data.shape
    assert n==k
    assert m==l

def test_dict():
    dir='./tests/data'
    iod.outDir=dir
    fname="tst.p"
    Dict={'a':34,'b':55}
    iod.store_data_dict(Dict,fname)
    nDict=iod.load_data_dict(fname,dir)
    assert Dict['a']==nDict['a']

def test_npArr():
    dir = './tests/data'
    iod.outDir = dir
    fname =  "tst.npy"
    Arr = np.array([1,2,3,4,42])
    iod.store_np_arr(Arr,fname)
    sArr=iod.load_np_arr(fname,dir)
    assert( sum(abs(Arr-sArr)) == 0)
