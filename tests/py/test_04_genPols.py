import numpy as np
import pandas as pd
import context
import aMRPC.iodata as iod
import aMRPC.datatools as dt
import aMRPC.polytools as pt
import aMRPC.utils as u

iod.inDir='./tests/data'
iod.outDir=iod.inDir
fname='InputParameters.txt'
Nr=1
No=4
srcs=[0, 1, 2, 3]

tol=1e-9

def load():
    return iod.loadEvalPoints(fname)
    

def test_load():
    """ test if data are correct loaded, dataset has 10k lines """
    data=load()
    n,m=data.shape
    assert n==10000


def gen_aPC4method(data,method):
    """ generate and test an aPC basis with method =method"""
    aNr=0;
    for src in srcs:
        H=pt.Hankel(No,data[src])
        r,w=pt.genRW(H,method)
        Cfs=pt.genPCmx(H,method)
        nCfs=pt.genNPCmx(Cfs,r,w)
        for i in range(No):
            for j in range(No):
                if i<=j:
                    pi=pt.PCeval(nCfs[i,:],r)
                    pj=pt.PCeval(nCfs[j,:],r)
                    q=(pi*pj)@w
                    if i==j:
                        assert abs(q-1)<tol
                    else:
                        assert abs(q)<tol

def test_aPC4m0():
    """ method 0: Gautschi"""
    data=load()
    gen_aPC4method(data,0)

def test_aPC4m1():
    """ method 1: aPC- Sergey, r,w -Karniadakis & Kirby () """
    data=load()
    gen_aPC4method(data,1)
