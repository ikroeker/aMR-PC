import numpy as np
#import pandas as pd
import context
import aMRPC.iodata as iod
import aMRPC.datatools as dt
import aMRPC.polytools as pt
#import aMRPC.utils as u

iod.inDir = './tests/data'
iod.outDir = iod.inDir
fname = 'InputParameters.txt'
Nr = 2
No = 4
srcs = [0, 1, 2, 3]

tol = 1e-5

def load():
    return iod.load_eval_points(fname)


def test_load():
    """ test if data are correct loaded, dataset has 10k lines """
    data = load()
    n, m = data.shape
    assert n == 10000


def gen_aPC4method(data, method, ctol, norm_method=0):
    """ generate and test an aPC basis with method =method"""
    for src in srcs:
        H = pt.Hankel(No, data[src])
        Cfs = pt.gen_pc_mx(H, method)
        r, w = pt.gen_rw(H, method)
        if norm_method == 1:
            nCfs = pt.gen_npc_mx_mm(Cfs, H)
        else:
            nCfs = pt.gen_npc_mx(Cfs, r, w)


        for i in range(No):
            for j in range(No):
                if i<= j:
                    pi = pt.pc_eval(nCfs[i, :], r)
                    pj = pt.pc_eval(nCfs[j, :], r)
                    q = (pi*pj) @ w
                    if i == j:
                        assert abs(q-1) < ctol
                    else:
                        assert abs(q) < ctol


def test_aPC4m0():
    """ method 0: Gautschi"""
    data = load()
    gen_aPC4method(data, 0, tol)

def test_aPC4m1():
    """ method 1: aPC- Sergey, r,w -Karniadakis & Kirby () """
    data = load()
    gen_aPC4method(data, 1, tol)


def test_aMRPC4m0():
    """ tests aMR_PC pols generated by method 0 (Gautschi) """
    method = 0
    dataframe = load()
    for aNr in range(0, Nr+1):
        for Nri in range(2**aNr):
            l_b, r_b = dt.cmp_lrb(aNr, Nri)
            qlb = dataframe.quantile(l_b)
            qrb = dataframe.quantile(r_b)
            bd = dt.cmp_quant_domain(np.array(dataframe), np.array(qlb),
                                     np.array(qrb))
            qdata = dataframe[bd]
            gen_aPC4method(qdata, method, tol)


def test_aMRPC4m1():
    """ tests aMR_PC pols generated by method 1 - Sergey moment normalization """
    method = 1
    dataframe = load()
    for aNr in range(0, Nr+1):
        for Nri in range(2**aNr):
            l_b, r_b = dt.cmp_lrb(aNr, Nri)
            qlb = dataframe.quantile(l_b)
            qrb = dataframe.quantile(r_b)
            bd = dt.cmp_quant_domain(np.array(dataframe), np.array(qlb),
                                     np.array(qrb))
            qdata = dataframe[bd]
            gen_aPC4method(qdata, method, tol, 1)

def test_aMRPC4m1q():
    """ tests aMR_PC pols generated by method 1 - Sergey quad. normalization """
    method = 1
    dataframe = load()
    for aNr in range(0, Nr+1):
        for Nri in range(2**aNr):
            l_b, r_b = dt.cmp_lrb(aNr, Nri)
            qlb = dataframe.quantile(l_b)
            qrb = dataframe.quantile(r_b)
            bd = dt.cmp_quant_domain(np.array(dataframe), np.array(qlb),
                                     np.array(qrb))
            qdata = dataframe[bd]
            gen_aPC4method(qdata, method, tol)
