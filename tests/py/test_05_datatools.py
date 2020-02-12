"""
pytest-3 test modul for datatools.py
"""
import numpy as np
#import pandas as pd
from scipy.special import comb
import context
import aMRPC.iodata as iod
import aMRPC.datatools as dt
import aMRPC.polytools as pt
import aMRPC.utils as u
import aMRPC.wavetools as wt

iod.inDir = './tests/data'
iod.outDir = iod.inDir
FNAME = 'InputParameters.txt'
Nr = 3
No = 3
srcs = [0, 1, 2, 3]
srcs = [0, 2, 3]
method = 0
tol = 1e-6 * 10**Nr
eps = 0.01 # multi-wavelet threshold
#eps = 0.005 # multi-wavelet threshold

NrRange = np.arange(Nr+1)
DIM = len(srcs)

def load():
    return iod.load_eval_points(FNAME)


def test_load():
    """ test if data are correct loaded, dataset has 10k lines """
    data = load()
    n, _ = data.shape
    assert n == 10000


def test_innerProdFct():
    """ tests inner product for two polynomials as functions"""
    dataframe = load()
    Hdict = dt.genHankel(dataframe, srcs, NrRange, No)
    R, W = dt.gen_roots_weights(Hdict, method)
    PCdict = dt.gen_pcs(Hdict, method)
    nPCdict = dt.gen_npcs(PCdict, R, W)
    Alphas = u.gen_multi_idx(No, DIM)
    P = int(comb(No+DIM, DIM))
    for Nra in NrRange:
        aNrs = Nra*np.ones(DIM)
        NriRange, Lines = u.gen_nri_range(aNrs)
        for Nri in range(Lines):
            Nris = NriRange[Nri, :]
            mk = u.gen_multi_key(aNrs, Nris, srcs)
            aR, aW = dt.gen_rw_4mkey(mk, R, W)
            for p in range(P):
                pCfs = dt.pcfs4eval(nPCdict, mk, Alphas[p])
                F = lambda x: pt.pc_eval(pCfs, x)
                for q in range(P):
                    qCfs = dt.pcfs4eval(nPCdict, mk, Alphas[q])
                    G = lambda x: pt.pc_eval(qCfs, x)
                    q_pq = dt.inner_prod_fct(F, G, aR, aW)
                    q_mk = dt.inner_prod_multi_idx(F, G, mk, R, W)
                    if p != q:
                        assert abs(q_pq) < tol
                        assert abs(q_mk) < tol
                    else:
                        assert abs(1-q_pq) < tol
                        assert abs(1-q_mk) < tol


def test_sample2mkey():
    """ tests the point to multi-key relationship """
    dataframe = load()
    myNrRange = [Nr]
    # Generate Hankel matrices
    Hdict = dt.genHankel(dataframe, srcs, NrRange, No)
    # Roots and Weights
    R, W = dt.gen_roots_weights(Hdict, method)
    # Generates dictionary of MR-elements bounds
    NRBdict = dt.gen_nr_range_bds(dataframe, srcs, myNrRange)
    # get roots and weights for the output
    mkLst = dt.gen_mkey_list(NRBdict, srcs)
    tR, _, mkLstLong = dt.get_rw_4mkey(mkLst, R, W)
    n = len(mkLstLong)
    for i in range(n):
        mk = mkLstLong[i]
        sample = tR[i, :]
        mks = dt.sample2mkey(sample, mkLst, NRBdict, True)
        assert len(mks) == 1
        assert mk == mks[0]

def test_u_nri_ranges():
    """
    tests both utils.genNriRange functions
    """
    dataframe = load()
    myNrRange = [Nr]
    # Generate Hankel matrices
    Hdict = dt.genHankel(dataframe, srcs, NrRange, No)
    # Roots and Weights
    R, W = dt.gen_roots_weights(Hdict, method)
    # Generates dictionary of MR-elements bounds
    NRBdict = dt.gen_nr_range_bds(dataframe, srcs, myNrRange)
    # get roots and weights for the output
    mkLst = dt.gen_mkey_list(NRBdict, srcs)
    mkey_set = set(mkLst)
    assert len(mkey_set) > 0
    aNrs = Nr*np.ones(DIM)
    Nri_arr, Nri_cnt = u.gen_nri_range(aNrs)
    Nri_arr_set, Nri_cnt_set = u.gen_nri_range_4mkset(mkey_set, DIM)
    assert Nri_cnt == Nri_cnt_set
    sum_nri = np.sum(Nri_arr, axis=0)
    sum_nri_set = np.sum(Nri_arr_set, axis=0)
    for d in range(DIM):
        assert sum_nri[d] == sum_nri_set[d]

def test_mkey_sid_rel():
    """ tests the point to multi-key relationship in dictionaries"""
    dataframe = load()
    myNrRange = [Nr]
    # Generate Hankel matrices
    Hdict = dt.genHankel(dataframe, srcs, NrRange, No)
    # Roots and Weights
    R, W = dt.gen_roots_weights(Hdict, method)
    # Generates dictionary of MR-elements bounds
    NRBdict = dt.gen_nr_range_bds(dataframe, srcs, myNrRange)
    # get roots and weights for the output
    mkLst = dt.gen_mkey_list(NRBdict, srcs)
    tR, tW, mkLstLong = dt.get_rw_4mkey(mkLst, R, W)
    sid2mk, mk2sid = dt.gen_mkey_sid_rel(tR, mkLst, NRBdict)
    n = len(mkLstLong)
    for sid in range(n):
        mk = mkLstLong[sid]
        mkl = sid2mk[sid]
        assert mk == mkl[0]
        sids = mk2sid[mk]
        assert sid in sids

def test_pol_on_samples():
    """ tests genPolOnSamplesArr(...)"""
    dataframe = load()
    myNrRange = [Nr]
    Hdict = dt.genHankel(dataframe, srcs, NrRange, No)
    R, W = dt.gen_roots_weights(Hdict, method)
    PCdict = dt.gen_pcs(Hdict, method)
    nPCdict = dt.gen_npcs(PCdict, R, W)
    Alphas = u.gen_multi_idx(No, DIM)
    P = int(comb(No+DIM, DIM))
    # Generates dictionary of MR-elements bounds
    NRBdict = dt.gen_nr_range_bds(dataframe, srcs, myNrRange)
    # get roots and weights for the output
    mkLst = dt.gen_mkey_list(NRBdict, srcs)
    tR, tW, mkLstLong = dt.get_rw_4mkey(mkLst, R, W)
    sid2mk, mk2sid = dt.gen_mkey_sid_rel(tR, mkLst, NRBdict)
    polVals = dt.gen_pol_on_samples_arr(tR, nPCdict, Alphas, mk2sid)
    Gws = np.prod(tW, axis=1)
    n = tR.shape[0]
    for mk in mk2sid:
        sids = mk2sid[mk]
        for p in range(P):
            pV = polVals[p, sids]
            for q in range(P):
                qV = polVals[q, sids]
                p_pq = np.inner(pV*qV, Gws[sids])
                if p == q:
                    assert abs(1-p_pq) < tol
                else:
                    assert abs(p_pq) < tol


def test_cmp_resc_cf():
    """ tests sum cfs =1 """
    dataframe = load()
    myNrRange = [Nr]
    NRBdict = dt.gen_nr_range_bds(dataframe, srcs, myNrRange)
    mkLst = dt.gen_mkey_list(NRBdict, srcs)
    cf_sum = 0
    #DIM = len(srcs)
    assert abs(dt.cmp_resc_cf(mkLst[0])-dt.cmp_resc_cfl([Nr]*DIM)) < tol
    rCdict = dt.gen_rcf_dict(mkLst)
    for mk in mkLst:
        cf = dt.cmp_resc_cf(mk)
        assert cf == rCdict[mk]
        cf_sum += cf
    assert abs(cf_sum-1) < tol

def test_wavelet_adapted():
    """
    generates wavelet adapted multi-element dataset
    test if each sid in one multi-element only
    and all sid's are contained
    """
    dataframe = load()
    minNr = 2
    maxNr = Nr
    minNo = 1
    maxNo = No
    for nr in range(minNr, maxNr+1):
        #nrs = nr*np.ones(DIM)
        myNrRange = np.arange(nr+1)
        NRBdict = dt.gen_nr_range_bds(dataframe, srcs, myNrRange)
        # multi-key list
        #mkLst = dt.gen_mkey_list(NRBdict, srcs)
        # rescaling coefficients for proj-> Nr=0
        #rCdict = dt.genRCfDict(mkLst)
        for no in range(minNo, maxNo+1):
            # generate roots, weights and also polynomials
            # compare with Rooots-n-Weights
            # Generate Hankel matrices
            Hdict = dt.genHankel(dataframe, srcs, myNrRange, no)
            # Roots and Weights
            R, W = dt.gen_roots_weights(Hdict, method)
            # (monic) orthogonal polynomial coefficients
            PCdict = dt.gen_pcs(Hdict, method)
            # normed orthogonal polynomials
            nPCdict = dt.gen_npcs(PCdict, R, W)

            # generate wavelets and compute details
            P = no+1
            wv = wt.WaveTools(P)
            wv.genWVlets()
            # compute quantiles on roots on rescaled (0,1)-Legendre polynomials
            Qdict = dt.gen_quant_dict(dataframe, srcs, myNrRange, wv)
            # details
            Dtdict = dt.gen_detail_dict(Qdict, wv)
            # stoch elements to keep (details >= threshold)
            Kdict = dt.mark_dict4keep(Dtdict, eps)
            # highest level
            topKeys = dt.get_top_keys(Kdict, srcs)
            # multi-keys generated from topKeys
            mkLst = dt.gen_mkey_list(topKeys, srcs)
            #print(topKeys)
            # get roots and weights for the output
            tR, tW, mkLstLong = dt.get_rw_4mkey(mkLst, R, W)
            sid2mk, mk2sid = dt.gen_mkey_sid_rel(tR, mkLst, NRBdict)
            point_set = set()
            len_roots = tR.shape[0]
            for sid, mk in sid2mk.items():
                assert len(mk) == 1
                point_set.add(sid)
            assert len(point_set) == len_roots
