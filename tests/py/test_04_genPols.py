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
Nr=4
No=4
srcs=[0, 1, 2, 3]

tol=1e-5

def load():
    return iod.loadEvalPoints(fname)
    

def test_load():
    """ test if data are correct loaded, dataset has 10k lines """
    data=load()
    n,m=data.shape
    assert n==10000


def gen_aPC4method(data,method,ctol):
    """ generate and test an aPC basis with method =method"""
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
                        assert abs(q-1)<ctol
                    else:
                        assert abs(q)<ctol

def test_aPC4m0():
    """ method 0: Gautschi"""
    data=load()
    gen_aPC4method(data,0,tol)

def test_aPC4m1():
    """ method 1: aPC- Sergey, r,w -Karniadakis & Kirby () """
    data=load()
    gen_aPC4method(data,1,tol)


def test_aMRPC4m0():
    """ tests aMR_PC pols generated by method 0 (Gautschi) """
    method=0
    dataframe=load()
    for aNr in range(0,Nr+1):
        for Nri in range(2**aNr):
            lb,rb=dt.cmpLRB(aNr,Nri)
            qlb=dataframe.quantile(lb)
            qrb=dataframe.quantile(rb)  
            bd=dt.cmpQuantDomain(dataframe,qlb,qrb)
            qdata=dataframe[bd]
            gen_aPC4method(qdata,method,tol)


def test_aMRPC4m0():
    """ tests aMR_PC pols generated by method 0 (Gautschi) """
    method=1
    dataframe=load()
    for aNr in range(0,Nr+1):
        for Nri in range(2**aNr):
            lb,rb=dt.cmpLRB(aNr,Nri)
            qlb=dataframe.quantile(lb)
            qrb=dataframe.quantile(rb)  
            bd=dt.cmpQuantDomain(dataframe,qlb,qrb)
            qdata=dataframe[bd]
            gen_aPC4method(qdata,method,tol)
