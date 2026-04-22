"""
Extended unit tests for aMRPC/datatools.py

Covers functions not tested in test_05_datatools.py:
- cmp_lrb
- cmp_quant_domain
- genHankel_uniform
- gen_nr_range_bds_4list
- get_nr_bds
- gen_bds_arr
- cmp_mv_quant_domain_mk / cmp_mv_quant_domain_arr
- Gauss_quad, inner_prod
- Gauss_quad_fct, inner_prod_fct, Gauss_quad_arr, inner_prod_tuples
- gen_amrpc_dec_ls (basic LS decomposition)
- gen_amrpc_dec_q (quadrature decomposition)
- cf_2_mean_var, cf_2_mean_var_4s, cf_2_mean_var_4mkey
- add_samples
- gen_phi
"""
import numpy as np
import pandas as pd
from scipy.special import comb
import context
import aMRPC.iodata as iod
import aMRPC.datatools as dt
import aMRPC.polytools as pt
import aMRPC.utils as u

iod.inDir = './tests/data'
iod.outDir = iod.inDir
FNAME = 'InputParameters.txt'
NR = 2
NO = 2
SRCS = [0, 1, 2]
METHOD = 1
TOL = 1e-5
DIM = len(SRCS)
NR_RANGE = np.arange(NR + 1)


def load():
    return iod.load_eval_points(FNAME)


# --- cmp_lrb ---

def test_cmp_lrb_nr0():
    """Nr=0, Nri=0 should give [0, 1]."""
    lb, rb = dt.cmp_lrb(0, 0)
    assert abs(lb - 0.0) < TOL
    assert abs(rb - 1.0) < TOL


def test_cmp_lrb_nr1():
    """Nr=1 should produce two halves."""
    lb0, rb0 = dt.cmp_lrb(1, 0)
    lb1, rb1 = dt.cmp_lrb(1, 1)
    assert abs(lb0) < TOL
    assert abs(rb0 - 0.5) < TOL
    assert abs(lb1 - 0.5) < TOL
    assert abs(rb1 - 1.0) < TOL


def test_cmp_lrb_coverage():
    """Sub-intervals should tile [0,1] without gaps."""
    for nr in range(5):
        prev = 0.0
        for nri in range(2**nr):
            lb, rb = dt.cmp_lrb(nr, nri)
            assert abs(lb - prev) < TOL
            prev = rb
        assert abs(prev - 1.0) < TOL


# --- cmp_quant_domain ---

def test_cmp_quant_domain_all():
    """cmp_quant_domain 'all' mode includes both bounds."""
    data = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    mask = dt.cmp_quant_domain(data, 0.25, 0.75)
    assert np.array_equal(mask, [False, True, True, True, False])


def test_cmp_quant_domain_rb():
    """cmp_quant_domain 'rb' mode excludes left bound."""
    data = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    mask = dt.cmp_quant_domain(data, 0.25, 0.75, bds='rb')
    assert np.array_equal(mask, [False, False, True, True, False])


def test_cmp_quant_domain_lb():
    """cmp_quant_domain 'lb' mode excludes right bound."""
    data = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    mask = dt.cmp_quant_domain(data, 0.25, 0.75, bds='lb')
    assert np.array_equal(mask, [False, True, True, False, False])


# --- genHankel_uniform ---

def test_genHankel_uniform_shape():
    """genHankel_uniform produces correctly shaped Hankel matrices."""
    lb_v = [0.0, 0.0]
    ub_v = [1.0, 1.0]
    srcs = [0, 1]
    nr_range = [0]
    no = 3
    h_dict = dt.genHankel_uniform(lb_v, ub_v, srcs, nr_range, no)
    for key, H in h_dict.items():
        assert H.shape == (no + 2, no + 2)


def test_genHankel_uniform_matches_uniHank():
    """genHankel_uniform at Nr=0 should match pt.uniHank."""
    lb_v = [0.0]
    ub_v = [1.0]
    srcs = [0]
    nr_range = [0]
    no = 4
    h_dict = dt.genHankel_uniform(lb_v, ub_v, srcs, nr_range, no)
    key = u.gen_dict_key(0, 0, 0)
    H_expected = pt.uniHank(no + 1, 0.0, 1.0)
    np.testing.assert_allclose(h_dict[key], H_expected, atol=TOL)


# --- gen_nr_range_bds_4list ---

def test_gen_nr_range_bds_4list():
    """gen_nr_range_bds_4list produces correct boundary dictionary."""
    bds_list = {0: (0.0, 10.0), 1: (-5.0, 5.0)}
    srcs = [0, 1]
    nr_range = [0, 1]
    nrb_dict = dt.gen_nr_range_bds_4list(bds_list, srcs, nr_range)

    # Nr=0 for src=0: lb=0, rb=10
    key = u.gen_dict_key(0, 0, 0)
    lb, rb = nrb_dict[key]
    # Nri=0 at Nr=0 gets eps adjustment on lb
    assert lb < 0.0 + 0.1  # with EPS offset
    assert abs(rb - 10.0) < 0.1

    # Nr=1 for src=1 has two sub-elements
    key0 = u.gen_dict_key(1, 0, 1)
    key1 = u.gen_dict_key(1, 1, 1)
    assert key0 in nrb_dict
    assert key1 in nrb_dict


# --- get_nr_bds ---

def test_get_nr_bds():
    """get_nr_bds extracts boundaries for a specific refinement level."""
    dataframe = load()
    nrb_dict = dt.gen_nr_range_bds(dataframe, SRCS, NR_RANGE)
    nr_bds = dt.get_nr_bds(nrb_dict, SRCS, NR)
    # Should have 2^NR entries per source
    expected_count = len(SRCS) * (2**NR)
    assert len(nr_bds) == expected_count
    for key in nr_bds:
        assert key[0] == NR  # all should have aNr == NR


# --- gen_bds_arr ---

def test_gen_bds_arr():
    """gen_bds_arr produces correct boundary array from multi-key."""
    dataframe = load()
    nrb_dict = dt.gen_nr_range_bds(dataframe, SRCS, NR_RANGE)
    mk_lst = dt.gen_mkey_list(nrb_dict, SRCS)
    # Pick the first multi-key
    mk = mk_lst[0]
    bds_arr = dt.gen_bds_arr(mk, nrb_dict)
    assert bds_arr.shape == (DIM, 2)
    # Lower bound should be less than upper bound
    for d in range(DIM):
        assert bds_arr[d, 0] <= bds_arr[d, 1]


# --- cmp_mv_quant_domain_mk and cmp_mv_quant_domain_arr ---

def test_mv_quant_domain_mk_vs_arr():
    """cmp_mv_quant_domain_mk and cmp_mv_quant_domain_arr should agree."""
    dataframe = load()
    nrb_dict = dt.gen_nr_range_bds(dataframe, SRCS, [NR])
    mk_lst = dt.gen_mkey_list(nrb_dict, SRCS)

    # Create test points
    np.random.seed(42)
    n_pts = 50
    test_pts = np.random.randn(n_pts, DIM)

    mk = mk_lst[0]
    mask_mk = dt.cmp_mv_quant_domain_mk(test_pts, nrb_dict, mk)
    bds_arr = dt.gen_bds_arr(mk, nrb_dict)
    mask_arr = dt.cmp_mv_quant_domain_arr(test_pts, bds_arr)
    np.testing.assert_array_equal(mask_mk, mask_arr)


# --- Gauss_quad, inner_prod (1-D) ---

def test_gauss_quad_polynomial():
    """Gauss quadrature should integrate polynomial exactly."""
    # Integrate x^2 on [0,1] using U(0,1) quadrature
    H = pt.uniHank(5)
    r, w = pt.gen_rw(H, 0)
    # int_0^1 x^2 dx = 1/3
    result = dt.Gauss_quad(lambda x: x**2, r, w)
    assert abs(result - 1.0 / 3) < TOL


def test_inner_prod_orthogonal():
    """Inner product of orthonormal polynomials should be delta_ij."""
    H = pt.uniHank(5)
    cf_mx = pt.gen_pc_mx(H, 0)
    r, w = pt.gen_rw(H, 0)
    ncf = pt.gen_npc_mx(cf_mx, r, w)
    No = cf_mx.shape[0]
    for i in range(No):
        fi = lambda x, ci=ncf[i, :]: pt.pc_eval(ci, x)
        for j in range(No):
            fj = lambda x, cj=ncf[j, :]: pt.pc_eval(cj, x)
            ip = dt.inner_prod(fi, fj, r, w)
            if i == j:
                assert abs(ip - 1.0) < TOL
            else:
                assert abs(ip) < TOL


# --- gen_phi ---

def test_gen_phi_shape():
    """gen_phi produces information matrix of correct shape."""
    dataframe = load()
    Hdict = dt.genHankel(dataframe, SRCS, NR_RANGE, NO)
    R, W = dt.gen_roots_weights(Hdict, METHOD)
    PCdict = dt.gen_pcs(Hdict, METHOD)
    nPCdict = dt.gen_npcs(PCdict, R, W)
    Alphas = u.gen_multi_idx(NO, DIM)
    P = int(comb(NO + DIM, DIM))
    NRBdict = dt.gen_nr_range_bds(dataframe, SRCS, [NR])
    mkLst = dt.gen_mkey_list(NRBdict, SRCS)
    tR, tW, mkLstLong = dt.get_rw_4mkey(mkLst, R, W)
    sid2mk, mk2sid = dt.gen_mkey_sid_rel(tR, mkLst, NRBdict)
    polVals = dt.gen_pol_on_samples_arr(tR, nPCdict, Alphas, mk2sid)

    for mk, sids in mk2sid.items():
        phi = dt.gen_phi(mk, polVals, mk2sid)
        assert phi.shape[0] == len(sids)
        assert phi.shape[1] == P


# --- gen_amrpc_dec_ls (LS decomposition) ---

def test_gen_amrpc_dec_ls_constant():
    """LS decomposition of constant function should give c0 = const, rest ~ 0."""
    dataframe = load()
    Hdict = dt.genHankel(dataframe, SRCS, NR_RANGE, NO)
    R, W = dt.gen_roots_weights(Hdict, METHOD)
    PCdict = dt.gen_pcs(Hdict, METHOD)
    nPCdict = dt.gen_npcs(PCdict, R, W)
    Alphas = u.gen_multi_idx(NO, DIM)
    P = int(comb(NO + DIM, DIM))
    NRBdict = dt.gen_nr_range_bds(dataframe, SRCS, [NR])
    mkLst = dt.gen_mkey_list(NRBdict, SRCS)
    tR, tW, mkLstLong = dt.get_rw_4mkey(mkLst, R, W)
    sid2mk, mk2sid = dt.gen_mkey_sid_rel(tR, mkLst, NRBdict)
    polVals = dt.gen_pol_on_samples_arr(tR, nPCdict, Alphas, mk2sid)

    # Constant function f(x) = 5.0 for all samples, 1 x-point
    n_s = tR.shape[0]
    data = 5.0 * np.ones((n_s, 1))

    cfs = dt.gen_amrpc_dec_ls(data, polVals, mk2sid)
    # c0 should be ~5.0 for each multi-key, higher coeffs ~0
    for mk, sids in mk2sid.items():
        c0 = cfs[sids[0], 0, 0]
        assert abs(c0 - 5.0) < 0.1
        for p_idx in range(1, P):
            assert abs(cfs[sids[0], p_idx, 0]) < 0.1


# --- gen_amrpc_dec_q (quadrature decomposition) ---

def test_gen_amrpc_dec_q_constant():
    """Quadrature decomposition of constant function should give c0 = const."""
    dataframe = load()
    Hdict = dt.genHankel(dataframe, SRCS, NR_RANGE, NO)
    R, W = dt.gen_roots_weights(Hdict, METHOD)
    PCdict = dt.gen_pcs(Hdict, METHOD)
    nPCdict = dt.gen_npcs(PCdict, R, W)
    Alphas = u.gen_multi_idx(NO, DIM)
    P = int(comb(NO + DIM, DIM))
    NRBdict = dt.gen_nr_range_bds(dataframe, SRCS, [NR])
    mkLst = dt.gen_mkey_list(NRBdict, SRCS)
    tR, tW, mkLstLong = dt.get_rw_4mkey(mkLst, R, W)
    sid2mk, mk2sid = dt.gen_mkey_sid_rel(tR, mkLst, NRBdict)
    polVals = dt.gen_pol_on_samples_arr(tR, nPCdict, Alphas, mk2sid)

    n_s = tR.shape[0]
    data = 5.0 * np.ones((n_s, 1))

    cfs = dt.gen_amrpc_dec_q(data, polVals, mk2sid, tW)
    # c0 should be ~5.0, higher coeffs ~0
    for mk, sids in mk2sid.items():
        c0 = cfs[sids[0], 0, 0]
        assert abs(c0 - 5.0) < 0.5
        for p_idx in range(1, P):
            assert abs(cfs[sids[0], p_idx, 0]) < 0.5


# --- cf_2_mean_var (sample-based and mkey-based) ---

def test_cf_2_mean_var_4s_constant():
    """Mean of constant function should be the constant, variance ~ 0."""
    dataframe = load()
    Hdict = dt.genHankel(dataframe, SRCS, NR_RANGE, NO)
    R, W = dt.gen_roots_weights(Hdict, METHOD)
    PCdict = dt.gen_pcs(Hdict, METHOD)
    nPCdict = dt.gen_npcs(PCdict, R, W)
    Alphas = u.gen_multi_idx(NO, DIM)
    P = int(comb(NO + DIM, DIM))
    NRBdict = dt.gen_nr_range_bds(dataframe, SRCS, [NR])
    mkLst = dt.gen_mkey_list(NRBdict, SRCS)
    tR, tW, mkLstLong = dt.get_rw_4mkey(mkLst, R, W)
    sid2mk, mk2sid = dt.gen_mkey_sid_rel(tR, mkLst, NRBdict)
    polVals = dt.gen_pol_on_samples_arr(tR, nPCdict, Alphas, mk2sid)

    n_s = tR.shape[0]
    data = 5.0 * np.ones((n_s, 1))

    cfs = dt.gen_amrpc_dec_ls(data, polVals, mk2sid)
    rCdict = dt.gen_rcf_dict(mkLst)
    mean, var = dt.cf_2_mean_var_4s(cfs, rCdict, mk2sid)

    assert abs(mean[0] - 5.0) < 0.5
    assert abs(var[0]) < 0.5


def test_cf_2_mean_var_dispatch():
    """cf_2_mean_var correctly dispatches based on input type."""
    dataframe = load()
    Hdict = dt.genHankel(dataframe, SRCS, NR_RANGE, NO)
    R, W = dt.gen_roots_weights(Hdict, METHOD)
    PCdict = dt.gen_pcs(Hdict, METHOD)
    nPCdict = dt.gen_npcs(PCdict, R, W)
    Alphas = u.gen_multi_idx(NO, DIM)
    P = int(comb(NO + DIM, DIM))
    NRBdict = dt.gen_nr_range_bds(dataframe, SRCS, [NR])
    mkLst = dt.gen_mkey_list(NRBdict, SRCS)
    tR, tW, mkLstLong = dt.get_rw_4mkey(mkLst, R, W)
    sid2mk, mk2sid = dt.gen_mkey_sid_rel(tR, mkLst, NRBdict)
    polVals = dt.gen_pol_on_samples_arr(tR, nPCdict, Alphas, mk2sid)

    n_s = tR.shape[0]
    data = 5.0 * np.ones((n_s, 1))

    # Sample-based
    cfs_s = dt.gen_amrpc_dec_ls(data, polVals, mk2sid)
    rCdict = dt.gen_rcf_dict(mkLst)
    mean_s, var_s = dt.cf_2_mean_var(cfs_s, rCdict, mk2sid)

    # mkey-based dict
    cfs_mk = {}
    for mk, sids in mk2sid.items():
        cfs_mk[mk] = cfs_s[sids[0], :, :]
    mean_mk, var_mk = dt.cf_2_mean_var(cfs_mk, rCdict, mk2sid)

    np.testing.assert_allclose(mean_s, mean_mk, atol=TOL)
    np.testing.assert_allclose(var_s, var_mk, atol=TOL)


# --- add_samples ---

def test_add_samples():
    """add_samples concatenates two sample arrays."""
    s1 = np.array([[1, 2], [3, 4]])
    s2 = np.array([[5, 6]])
    result = dt.add_samples(s1, s2)
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result[0], [1, 2])
    np.testing.assert_array_equal(result[2], [5, 6])


def test_add_samples_preserves_cols():
    """add_samples preserves column count."""
    s1 = np.random.randn(10, 4)
    s2 = np.random.randn(5, 4)
    result = dt.add_samples(s1, s2)
    assert result.shape == (15, 4)


# --- Gauss_quad_fct and inner_prod_fct (multivariate) ---

def test_gauss_quad_fct():
    """Gauss_quad_fct integrates f(x0, x1) = 1 correctly."""
    dataframe = load()
    srcs2 = [0, 1]
    dim2 = len(srcs2)
    Hdict = dt.genHankel(dataframe, srcs2, [0], NO)
    R, W = dt.gen_roots_weights(Hdict, METHOD)
    mk = u.gen_multi_key([0] * dim2, [0] * dim2, srcs2)
    r_mk, w_mk = dt.gen_rw_4mkey(mk, R, W)
    result = dt.Gauss_quad_fct(lambda x: x * 0 + 1, r_mk, w_mk)
    # Should be ~1 (integral of 1 over the probability measure)
    assert abs(result - 1.0) < 0.1


# --- get_rw_4nrs ---

def test_get_rw_4nrs():
    """get_rw_4nrs generates eval points for a specific Nr level."""
    dataframe = load()
    Hdict = dt.genHankel(dataframe, SRCS, NR_RANGE, NO)
    R, W = dt.gen_roots_weights(Hdict, METHOD)
    nrs = np.array([NR] * DIM, dtype=float)
    tR, tW, mkLstLong = dt.get_rw_4nrs(nrs, SRCS, R, W)
    assert tR.shape[0] > 0
    assert tR.shape[1] == DIM
    assert tR.shape == tW.shape
    assert len(mkLstLong) == tR.shape[0]
