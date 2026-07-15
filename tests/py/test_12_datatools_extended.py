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
- gen_amrpc_dec_ls (basic LS decomposition, incl. reg_n/reg_t)
- gen_amrpc_dec_ls_mask (masked LS decomposition, incl. reg_t / num_aux)
- gen_amrpc_dec_mk_ls (mkey-based LS decomposition, incl. reg_n/reg_t)
- gen_amrpc_dec_q (quadrature decomposition)
- cf_2_mean_var, cf_2_mean_var_4s, cf_2_mean_var_4mkey
- add_samples
- gen_phi
- gen_cov_mx_4lh, gen_cov_mx_4lh_noex (Woodbury-based covariance inverse)
"""
import numpy as np
import pandas as pd
import pytest
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


def _setup_amrpc_pipeline():
    """Shared pipeline setup for gen_amrpc_dec_* tests.

    Returns (tR, mk2sid, polVals, p_max, data) with a non-constant
    data column (data = sin(tR[:, 0])) so regularized regression
    coefficients are non-trivial.
    """
    dataframe = load()
    Hdict = dt.genHankel(dataframe, SRCS, NR_RANGE, NO)
    R, W = dt.gen_roots_weights(Hdict, METHOD)
    PCdict = dt.gen_pcs(Hdict, METHOD)
    nPCdict = dt.gen_npcs(PCdict, R, W)
    Alphas = u.gen_multi_idx(NO, DIM)
    p_max = int(comb(NO + DIM, DIM))
    NRBdict = dt.gen_nr_range_bds(dataframe, SRCS, [NR])
    mkLst = dt.gen_mkey_list(NRBdict, SRCS)
    tR, tW, mkLstLong = dt.get_rw_4mkey(mkLst, R, W)
    sid2mk, mk2sid = dt.gen_mkey_sid_rel(tR, mkLst, NRBdict)
    polVals = dt.gen_pol_on_samples_arr(tR, nPCdict, Alphas, mk2sid)

    data = np.sin(tR[:, 0]).reshape(-1, 1)
    return tR, mk2sid, polVals, p_max, data


def _ref_reg_t(phi, data_col, sigma_n, sigma_p):
    """Reference reg_t coefficients/P_inv via a direct np.linalg.pinv(P)."""
    sigma_n_sq = sigma_n ** 2
    sigma_p_sq = sigma_p ** 2
    P = (phi.T / sigma_n_sq) @ phi + np.eye(phi.shape[1]) / sigma_p_sq
    P_inv = np.linalg.pinv(P)
    v_ls = P_inv @ phi.T / sigma_n_sq @ data_col
    return v_ls, P_inv


def _ref_reg_n(phi, data_col, sigma_n):
    """Reference reg_n coefficients/P_inv via a direct np.linalg.pinv(P)."""
    sigma_n_sq = sigma_n ** 2
    P = phi.T @ phi / sigma_n_sq
    P_inv = np.linalg.pinv(P)
    v_ls = P_inv @ phi.T / sigma_n_sq @ data_col
    return v_ls, P_inv


def test_gen_amrpc_dec_ls_reg_t_matches_pinv_reference():
    """method='reg_t' (Cholesky-primary) matches a direct pinv(P) reference."""
    tR, mk2sid, polVals, p_max, data = _setup_amrpc_pipeline()
    sigma_n, sigma_p = 0.1, 2.0

    cfs = dt.gen_amrpc_dec_ls(data, polVals, mk2sid, method='reg_t',
                              sigma_n=sigma_n, sigma_p=sigma_p)

    for mk, sids in mk2sid.items():
        phi = polVals[:, sids].T
        v_ref, _ = _ref_reg_t(phi, data[sids, 0], sigma_n, sigma_p)
        np.testing.assert_allclose(cfs[sids[0], :, 0], v_ref, atol=1e-8)


def test_gen_amrpc_dec_ls_reg_n_matches_pinv_reference():
    """method='reg_n' (Cholesky-primary) matches a direct pinv(P) reference."""
    tR, mk2sid, polVals, p_max, data = _setup_amrpc_pipeline()
    sigma_n = 0.1

    cfs = dt.gen_amrpc_dec_ls(data, polVals, mk2sid, method='reg_n',
                              sigma_n=sigma_n)

    for mk, sids in mk2sid.items():
        phi = polVals[:, sids].T
        v_ref, _ = _ref_reg_n(phi, data[sids, 0], sigma_n)
        np.testing.assert_allclose(cfs[sids[0], :, 0], v_ref, atol=1e-8)


def test_gen_amrpc_dec_ls_reg_t_return_cov():
    """reg_t return_cov option returns P_inv matching the pinv reference."""
    tR, mk2sid, polVals, p_max, data = _setup_amrpc_pipeline()
    sigma_n, sigma_p = 0.1, 2.0

    cfs, covs = dt.gen_amrpc_dec_ls(data, polVals, mk2sid, method='reg_t',
                                    sigma_n=sigma_n, sigma_p=sigma_p,
                                    return_cov=True)

    for mk, sids in mk2sid.items():
        phi = polVals[:, sids].T
        _, P_inv_ref = _ref_reg_t(phi, data[sids, 0], sigma_n, sigma_p)
        np.testing.assert_allclose(covs[sids[0], :, :, 0], P_inv_ref, atol=1e-8)


# --- gen_amrpc_dec_ls_mask (masked LS decomposition) ---

def test_gen_amrpc_dec_ls_mask_reg_t_matches_pinv_reference():
    """gen_amrpc_dec_ls_mask reg_t (num_aux=False) matches pinv(P) reference."""
    tR, mk2sid, polVals, p_max, data = _setup_amrpc_pipeline()
    sigma_n, sigma_p = 0.1, 2.0
    mask_dict = {mk: np.ones(p_max, dtype=np.bool_) for mk in mk2sid}

    cfs = dt.gen_amrpc_dec_ls_mask(data, polVals, mk2sid, mask_dict,
                                   method='reg_t', sigma_n=sigma_n,
                                   sigma_p=sigma_p, num_aux=False)

    for mk, sids in mk2sid.items():
        phi = polVals[:, sids].T
        v_ref, _ = _ref_reg_t(phi, data[sids, 0], sigma_n, sigma_p)
        np.testing.assert_allclose(cfs[sids[0], :, 0], v_ref, atol=1e-8)


def test_gen_amrpc_dec_ls_mask_reg_t_numba_aux_matches_reference():
    """gen_amrpc_dec_ls_mask reg_t (num_aux=True, numba path) matches
    pinv(P) reference and agrees with the non-numba path."""
    tR, mk2sid, polVals, p_max, data = _setup_amrpc_pipeline()
    sigma_n, sigma_p = 0.1, 2.0
    mask_dict = {mk: np.ones(p_max, dtype=np.bool_) for mk in mk2sid}

    cfs_aux = dt.gen_amrpc_dec_ls_mask(data, polVals, mk2sid, mask_dict,
                                       method='reg_t', sigma_n=sigma_n,
                                       sigma_p=sigma_p, num_aux=True)
    cfs_noaux = dt.gen_amrpc_dec_ls_mask(data, polVals, mk2sid, mask_dict,
                                         method='reg_t', sigma_n=sigma_n,
                                         sigma_p=sigma_p, num_aux=False)

    np.testing.assert_allclose(cfs_aux, cfs_noaux, atol=1e-8)
    for mk, sids in mk2sid.items():
        phi = polVals[:, sids].T
        v_ref, _ = _ref_reg_t(phi, data[sids, 0], sigma_n, sigma_p)
        np.testing.assert_allclose(cfs_aux[sids[0], :, 0], v_ref, atol=1e-8)


# --- gen_amrpc_dec_mk_ls (mkey-based LS decomposition) ---

def test_gen_amrpc_dec_mk_ls_reg_t_matches_pinv_reference():
    """gen_amrpc_dec_mk_ls reg_t matches a direct pinv(P) reference."""
    tR, mk2sid, polVals, p_max, data = _setup_amrpc_pipeline()
    sigma_n, sigma_p = 0.1, 2.0

    cfs = dt.gen_amrpc_dec_mk_ls(data, polVals, mk2sid, method='reg_t',
                                 sigma_n=sigma_n, sigma_p=sigma_p)

    for mk, sids in mk2sid.items():
        phi = polVals[:, sids].T
        v_ref, _ = _ref_reg_t(phi, data[sids, 0], sigma_n, sigma_p)
        np.testing.assert_allclose(cfs[mk][:, 0], v_ref, atol=1e-8)


def test_gen_amrpc_dec_mk_ls_reg_n_matches_pinv_reference():
    """gen_amrpc_dec_mk_ls reg_n matches a direct pinv(P) reference."""
    tR, mk2sid, polVals, p_max, data = _setup_amrpc_pipeline()
    sigma_n = 0.1

    cfs = dt.gen_amrpc_dec_mk_ls(data, polVals, mk2sid, method='reg_n',
                                 sigma_n=sigma_n)

    for mk, sids in mk2sid.items():
        phi = polVals[:, sids].T
        v_ref, _ = _ref_reg_n(phi, data[sids, 0], sigma_n)
        np.testing.assert_allclose(cfs[mk][:, 0], v_ref, atol=1e-8)


# --- multi-column (x_len > 1) equivalence tests -----------------------
#
# gen_amrpc_dec_ls / gen_amrpc_dec_mk_ls / gen_amrpc_dec_ls_mask solve
# a per-mkey coefficient vector for each of x_len space/time columns.
# The operator (P_inv / pinv(phi) / mx_inv / ...) is constant across
# columns, so the implementations solve all x_len columns in a single
# batched matrix product instead of looping per column. The tests
# below build genuinely multi-column data (x_len=5) and cross-check
# the batched result against a straightforward per-column loop that
# calls np.linalg.pinv/lstsq/etc. one column at a time -- i.e. exactly
# the computation the original per-idx_x Python loop performed.

def _setup_amrpc_pipeline_multicol(x_len=5):
    """Like _setup_amrpc_pipeline but with x_len>1 non-constant columns."""
    dataframe = load()
    Hdict = dt.genHankel(dataframe, SRCS, NR_RANGE, NO)
    R, W = dt.gen_roots_weights(Hdict, METHOD)
    PCdict = dt.gen_pcs(Hdict, METHOD)
    nPCdict = dt.gen_npcs(PCdict, R, W)
    Alphas = u.gen_multi_idx(NO, DIM)
    p_max = int(comb(NO + DIM, DIM))
    NRBdict = dt.gen_nr_range_bds(dataframe, SRCS, [NR])
    mkLst = dt.gen_mkey_list(NRBdict, SRCS)
    tR, tW, mkLstLong = dt.get_rw_4mkey(mkLst, R, W)
    sid2mk, mk2sid = dt.gen_mkey_sid_rel(tR, mkLst, NRBdict)
    polVals = dt.gen_pol_on_samples_arr(tR, nPCdict, Alphas, mk2sid)

    # x_len different, non-constant columns (distinct frequencies so
    # per-column solutions genuinely differ from each other).
    cols = [np.sin((k + 1) * tR[:, 0]) + 0.1 * k * tR[:, 1]
            for k in range(x_len)]
    data = np.stack(cols, axis=1)
    return tR, mk2sid, polVals, p_max, data


def _loop_reference(phi, data_block, method, sigma_n_sq=None, sigma_p_sq=None):
    """Per-column reference: solves each column of data_block
    (n_s, x_len) independently via a Python loop, mirroring the
    pre-vectorization implementation column-by-column."""
    n_p = phi.shape[1]
    x_len = data_block.shape[1]
    out = np.zeros((n_p, x_len))
    for idx_x in range(x_len):
        col = data_block[:, idx_x]
        if method == 'pinv':
            out[:, idx_x] = np.linalg.pinv(phi) @ col
        elif method == 'unbias':
            out[:, idx_x] = np.linalg.pinv(phi.T @ phi) @ phi.T @ col
        elif method == 'reg_n':
            P = phi.T @ phi / sigma_n_sq
            P_inv = np.linalg.pinv(P)
            out[:, idx_x] = (P_inv @ phi.T / sigma_n_sq) @ col
        elif method == 'reg_t':
            P = (phi.T / sigma_n_sq) @ phi + np.eye(n_p) / sigma_p_sq
            P_inv = np.linalg.pinv(P)
            out[:, idx_x] = (P_inv @ phi.T / sigma_n_sq) @ col
        elif method == 'lstsq':
            v_ls, _, _, _ = np.linalg.lstsq(phi, col, rcond=None)
            out[:, idx_x] = v_ls
        else:
            raise ValueError(method)
    return out


@pytest.mark.parametrize("method", ['pinv', 'unbias', 'reg_n', 'reg_t', 'lstsq'])
def test_gen_amrpc_dec_ls_multicol_matches_loop_reference(method):
    """gen_amrpc_dec_ls batched (x_len>1) result matches a per-column loop."""
    tR, mk2sid, polVals, p_max, data = _setup_amrpc_pipeline_multicol()
    sigma_n, sigma_p = 0.2, 1.5
    kwargs = {'method': method}
    if method in ('reg_n', 'reg_t'):
        kwargs.update(sigma_n=sigma_n, sigma_p=sigma_p)

    cfs = dt.gen_amrpc_dec_ls(data, polVals, mk2sid, **kwargs)

    for mk, sids in mk2sid.items():
        phi = polVals[:, sids].T
        data_block = data[sids, :]
        ref = _loop_reference(phi, data_block, method,
                              sigma_n_sq=sigma_n**2, sigma_p_sq=sigma_p**2)
        np.testing.assert_allclose(cfs[sids[0], :, :], ref, atol=1e-8)


@pytest.mark.parametrize("method", ['pinv', 'unbias', 'reg_n', 'reg_t', 'lstsq'])
def test_gen_amrpc_dec_mk_ls_multicol_matches_loop_reference(method):
    """gen_amrpc_dec_mk_ls batched (x_len>1) result matches a per-column loop."""
    tR, mk2sid, polVals, p_max, data = _setup_amrpc_pipeline_multicol()
    sigma_n, sigma_p = 0.2, 1.5
    kwargs = {'method': method}
    if method in ('reg_n', 'reg_t'):
        kwargs.update(sigma_n=sigma_n, sigma_p=sigma_p)

    cfs = dt.gen_amrpc_dec_mk_ls(data, polVals, mk2sid, **kwargs)

    for mk, sids in mk2sid.items():
        phi = polVals[:, sids].T
        data_block = data[sids, :]
        ref = _loop_reference(phi, data_block, method,
                              sigma_n_sq=sigma_n**2, sigma_p_sq=sigma_p**2)
        np.testing.assert_allclose(cfs[mk], ref, atol=1e-8)


@pytest.mark.parametrize("method", ['pinv', 'unbias', 'reg_n', 'reg_t', 'lstsq'])
def test_gen_amrpc_dec_ls_mask_multicol_matches_loop_reference(method):
    """gen_amrpc_dec_ls_mask (num_aux=False) batched (x_len>1) result
    matches a per-column loop, for a full (all-True) mask."""
    tR, mk2sid, polVals, p_max, data = _setup_amrpc_pipeline_multicol()
    sigma_n, sigma_p = 0.2, 1.5
    mask_dict = {mk: np.ones(p_max, dtype=np.bool_) for mk in mk2sid}
    kwargs = {'method': method, 'num_aux': False}
    if method in ('reg_n', 'reg_t'):
        kwargs.update(sigma_n=sigma_n, sigma_p=sigma_p)

    cfs = dt.gen_amrpc_dec_ls_mask(data, polVals, mk2sid, mask_dict, **kwargs)

    for mk, sids in mk2sid.items():
        phi = polVals[:, sids].T
        data_block = data[sids, :]
        ref = _loop_reference(phi, data_block, method,
                              sigma_n_sq=sigma_n**2, sigma_p_sq=sigma_p**2)
        np.testing.assert_allclose(cfs[sids[0], :, :], ref, atol=1e-8)


def test_gen_amrpc_dec_ls_multicol_reg_t_return_cov_broadcast():
    """return_cov P_inv is constant across x_len columns for reg_t."""
    tR, mk2sid, polVals, p_max, data = _setup_amrpc_pipeline_multicol()
    sigma_n, sigma_p = 0.2, 1.5

    cfs, covs = dt.gen_amrpc_dec_ls(data, polVals, mk2sid, method='reg_t',
                                    sigma_n=sigma_n, sigma_p=sigma_p,
                                    return_cov=True)

    for mk, sids in mk2sid.items():
        phi = polVals[:, sids].T
        _, P_inv_ref = _ref_reg_t(phi, data[sids, 0], sigma_n, sigma_p)
        for idx_x in range(data.shape[1]):
            np.testing.assert_allclose(covs[sids[0], :, :, idx_x],
                                       P_inv_ref, atol=1e-8)


def test_gen_amrpc_dec_ls_mask_multicol_reg_t_return_std_broadcast():
    """return_std sqrt(diag(P_inv)) is constant across x_len columns."""
    tR, mk2sid, polVals, p_max, data = _setup_amrpc_pipeline_multicol()
    sigma_n, sigma_p = 0.2, 1.5
    mask_dict = {mk: np.ones(p_max, dtype=np.bool_) for mk in mk2sid}

    cfs, stds = dt.gen_amrpc_dec_ls_mask(data, polVals, mk2sid, mask_dict,
                                         method='reg_t', sigma_n=sigma_n,
                                         sigma_p=sigma_p, num_aux=False,
                                         return_std=True)

    for mk, sids in mk2sid.items():
        phi = polVals[:, sids].T
        _, P_inv_ref = _ref_reg_t(phi, data[sids, 0], sigma_n, sigma_p)
        std_ref = np.sqrt(np.diag(P_inv_ref))
        for idx_x in range(data.shape[1]):
            np.testing.assert_allclose(stds[sids[0], :, idx_x], std_ref,
                                       atol=1e-8)


# --- numba-jitted aux helpers (Phase 2): gen_amrpc_dec_ls_mask_aux,
#     gen_reg_t_aux, cmp_var_4_mk -- direct multi-column equivalence
#     tests against a per-column Python-loop reference mirroring the
#     pre-vectorization implementation. ---

def _loop_reference_ls_mask_aux(data, sids, pol_vals, alpha_mask, cov_mask,
                                sigma_n_mk, sigma_p_mk, meth_mode, cov_mode,
                                p_max, x_start, x_len):
    """Per-column reference for gen_amrpc_dec_ls_mask_aux."""
    n_s = sids.shape[0]
    phi = np.ascontiguousarray(dt.gen_phi_fast(pol_vals, sids, alpha_mask))
    if meth_mode == 1:
        P_inv = np.linalg.pinv(phi)
    elif meth_mode in (2, 5):
        P = (phi.T / sigma_n_mk) @ phi + np.eye(phi.shape[1]) / sigma_p_mk
        P_inv = np.linalg.pinv(P)
        mx_inv = P_inv @ phi.T / sigma_n_mk
    elif meth_mode == 3:
        P_inv = np.linalg.pinv(phi.T @ phi)
    elif meth_mode == 4:
        cov_op = np.linalg.pinv(1 / sigma_n_mk * phi.T @ phi)

    if meth_mode == 5:
        ret_std_cov_4s = np.zeros((1, p_max, p_max, x_len))
        ret_cf_ls_4s = np.zeros((1, p_max, x_len))
    else:
        ret_std_cov_4s = np.zeros((n_s, p_max, p_max, x_len))
        ret_cf_ls_4s = np.zeros((n_s, p_max, x_len))

    for idx_x in range(x_len):
        dt_idx_x = x_start + idx_x
        rs_data = data[sids, dt_idx_x]
        if meth_mode == 1:
            v_ls = P_inv @ rs_data
        elif meth_mode == 2:
            v_ls = mx_inv @ rs_data
            if cov_mode == 1:
                ret_std_cov_4s[:, :, 0, idx_x] = np.sqrt(np.diag(P_inv))
            elif cov_mode == 2:
                p_mx = np.zeros((p_max * p_max))
                p_mx[cov_mask.flatten()] = P_inv.flatten()
                p_mx = p_mx.reshape((p_max, -1))
                for s_idx in range(n_s):
                    ret_std_cov_4s[s_idx, :, :, idx_x] = p_mx
        elif meth_mode == 5:
            v_ls = mx_inv @ rs_data
            if cov_mode == 1:
                ret_std_cov_4s[0, :, 0, idx_x] = np.sqrt(np.diag(P_inv))
            elif cov_mode == 2:
                p_mx = np.zeros((p_max * p_max))
                p_mx[cov_mask.flatten()] = P_inv.flatten()
                ret_std_cov_4s[0, :, :, idx_x] = p_mx.reshape((p_max, -1))
        elif meth_mode == 3:
            v_ls = P_inv @ phi.T @ rs_data
        elif meth_mode == 4:
            v_ls = (cov_op @ phi.T / sigma_n_mk) @ rs_data
            if cov_mode == 1:
                ret_std_cov_4s[:, :, 0, idx_x] = np.sqrt(np.diag(cov_op))
            elif cov_mode == 2:
                ret_std_cov_4s[:, :, :, idx_x] = cov_op
        else:
            v_ls, _, _, _ = np.linalg.lstsq(phi, rs_data, rcond=-1)
        if meth_mode in (5,):
            ret_cf_ls_4s[0, alpha_mask, idx_x] = v_ls
        else:
            tmp = np.zeros(p_max)
            tmp[alpha_mask] = v_ls
            ret_cf_ls_4s[:, :, idx_x] = tmp
    return ret_cf_ls_4s, ret_std_cov_4s


@pytest.mark.parametrize("meth_mode,cov_mode",
                         [(0, 0), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2),
                          (3, 0), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2)])
def test_gen_amrpc_dec_ls_mask_aux_multicol_matches_loop_reference(meth_mode, cov_mode):
    """gen_amrpc_dec_ls_mask_aux (numba, batched over idx_x) matches a
    per-column Python-loop reference, for a full (all-True) mask, across
    all meth_mode/cov_mode combinations."""
    np.random.seed(11)
    n_s, p_max, x_len, x_start = 20, 6, 9, 2
    sids = np.arange(n_s, dtype=np.int64)
    pol_vals = np.ascontiguousarray(np.random.randn(p_max, n_s + 3))
    data = np.random.randn(n_s + 3, x_len + x_start + 2)
    alpha_mask = np.ones(p_max, dtype=np.bool_)
    cov_mask = np.multiply.outer(alpha_mask, alpha_mask)
    sigma_n_mk, sigma_p_mk = 0.05, 3.0

    cf, cov = dt.gen_amrpc_dec_ls_mask_aux(data, sids, pol_vals, alpha_mask,
                                           cov_mask, sigma_n_mk, sigma_p_mk,
                                           meth_mode, cov_mode, p_max,
                                           x_start, x_len)
    cf_ref, cov_ref = _loop_reference_ls_mask_aux(
        data, sids, pol_vals, alpha_mask, cov_mask, sigma_n_mk, sigma_p_mk,
        meth_mode, cov_mode, p_max, x_start, x_len)

    np.testing.assert_allclose(cf, cf_ref, atol=1e-8)
    np.testing.assert_allclose(cov, cov_ref, atol=1e-8)


def test_gen_amrpc_dec_ls_mask_aux_n_s_le_1_edge_case():
    """gen_amrpc_dec_ls_mask_aux degenerate n_s<=1 branch (kept as a
    plain per-column loop) still matches a direct reference."""
    np.random.seed(12)
    p_max, x_len, x_start = 4, 3, 1
    sids = np.array([0], dtype=np.int64)
    pol_vals = np.ascontiguousarray(np.random.randn(p_max, 5))
    alpha_mask = np.ones(p_max, dtype=np.bool_)
    cov_mask = np.multiply.outer(alpha_mask, alpha_mask)
    data = np.random.randn(5, x_len + x_start + 1)

    cf, _ = dt.gen_amrpc_dec_ls_mask_aux(data, sids, pol_vals, alpha_mask,
                                         cov_mask, 0.1, 1.0, 0, 0, p_max,
                                         x_start, x_len)
    phi = dt.gen_phi_fast(pol_vals, sids, alpha_mask)
    ref = np.zeros((p_max, x_len))
    for idx_x in range(x_len):
        ref[:, idx_x] = np.ravel(data[0, x_start + idx_x] / phi)
    np.testing.assert_allclose(cf[0], ref, atol=1e-8)


def test_gen_reg_t_aux_multicol_matches_loop_reference():
    """gen_reg_t_aux (numba, batched over idx_x) matches a per-column
    Python-loop reference."""
    np.random.seed(13)
    n_s, p_max, x_len, x_start = 20, 5, 7, 1
    sids = np.arange(n_s, dtype=np.int64)
    phi = np.ascontiguousarray(np.random.randn(n_s, p_max))
    data = np.random.randn(n_s, x_len + x_start + 3)
    alpha_mask = np.ones(p_max, dtype=np.bool_)
    cov_mask = np.ones((p_max, p_max), dtype=np.bool_)
    sigma_n_sq, sigma_p_sq = 0.1, 2.0

    cf, P_inv = dt.gen_reg_t_aux(data, sids, phi, alpha_mask, cov_mask,
                                 sigma_n_sq, sigma_p_sq, x_start, x_len)

    P = (phi.T / sigma_n_sq) @ phi + np.eye(phi.shape[1]) / sigma_p_sq
    P_inv_ref = np.linalg.pinv(P)
    ref = np.zeros((p_max, x_len))
    for idx_x in range(x_len):
        dt_idx_x = x_start + idx_x
        v_ls = (P_inv_ref @ phi.T / sigma_n_sq) @ data[sids, dt_idx_x]
        ref[alpha_mask, idx_x] = v_ls

    np.testing.assert_allclose(cf, ref, atol=1e-8)
    np.testing.assert_allclose(P_inv, P_inv_ref, atol=1e-8)


def test_cmp_var_4_mk_matches_loop_reference():
    """cmp_var_4_mk (numba, batched quadratic form over idx_x) matches
    a per-sample/per-(k,r) Python-loop reference."""
    np.random.seed(14)
    p_max, n_s, n_x = 7, 6, 12
    p_vals = np.ascontiguousarray(np.random.randn(p_max, n_s))
    f_cov = np.ascontiguousarray(np.random.randn(p_max, p_max, n_x))
    idxs_pm = np.array([0, 2, 3, 5, 6], dtype=np.int64)
    vec_x = np.arange(n_x, dtype=np.int64)

    out = dt.cmp_var_4_mk(vec_x, p_vals, idxs_pm, n_s, f_cov)

    ref = np.zeros((n_s, n_x))
    for idx_x in vec_x:
        for s_i in range(n_s):
            s_var = 0.0
            for k in idxs_pm:
                for r in idxs_pm:
                    s_var += p_vals[k, s_i] * p_vals[r, s_i] * f_cov[k, r, idx_x]
            ref[s_i, idx_x] = s_var

    np.testing.assert_allclose(out, ref, atol=1e-10)


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


# --- gen_cov_mx_4lh / gen_cov_mx_4lh_noex (Woodbury identity) ---
#
# cov_mx = Q + phi @ R @ phi.T, Q = s_sigma_n * I_n, R = s_sigma_p * I_p.
# The inverse is computed via the Sherman-Morrison-Woodbury identity
# (eq. A.9 in Rasmussen & Williams) when p < n, inverting the (much
# smaller) p x p matrix P = phi.T @ Q_inv @ phi + R_inv instead of the
# n x n matrix cov_mx directly. When p >= n, Woodbury gives no benefit,
# so cov_mx is inverted directly via Cholesky. np.linalg.pinv of
# cov_mx is used only as a last-resort exception fallback.

def _ref_cov_mx_inv(phi, s_sigma_n, s_sigma_p):
    """Reference cov_mx / cov_mx_inv via direct construction + pinv."""
    n_s = phi.shape[0]
    Q = np.eye(n_s) * s_sigma_n
    R = np.eye(phi.shape[1]) * s_sigma_p
    cov_mx = phi @ R @ phi.T + Q
    cov_mx_inv = np.linalg.pinv(cov_mx)
    return cov_mx, cov_mx_inv


def test_gen_cov_mx_4lh_matches_direct_inverse():
    """gen_cov_mx_4lh (Woodbury) matches a direct pinv reference."""
    np.random.seed(0)
    for n_s, n_p in [(20, 4), (200, 6), (50, 10)]:
        phi = np.random.randn(n_s, n_p)
        s_sigma_n, s_sigma_p = 0.7, 1.3
        cov_mx, cov_mx_inv = dt.gen_cov_mx_4lh(phi, s_sigma_n, s_sigma_p)
        ref_cov_mx, ref_inv = _ref_cov_mx_inv(phi, s_sigma_n, s_sigma_p)
        np.testing.assert_allclose(cov_mx, ref_cov_mx, atol=TOL)
        np.testing.assert_allclose(cov_mx_inv, ref_inv, atol=1e-6)


def test_gen_cov_mx_4lh_identity_property():
    """cov_mx @ cov_mx_inv should be (approximately) the identity."""
    np.random.seed(1)
    phi = np.random.randn(100, 5)
    s_sigma_n, s_sigma_p = 0.5, 2.0
    cov_mx, cov_mx_inv = dt.gen_cov_mx_4lh(phi, s_sigma_n, s_sigma_p)
    n_s = phi.shape[0]
    np.testing.assert_allclose(cov_mx @ cov_mx_inv, np.eye(n_s), atol=1e-6)


def test_gen_cov_mx_4lh_noex_matches_direct_inverse():
    """gen_cov_mx_4lh_noex (Woodbury, numba) matches a direct pinv reference."""
    np.random.seed(2)
    for n_s, n_p in [(20, 4), (200, 6), (50, 10)]:
        phi = np.random.randn(n_s, n_p)
        s_sigma_n, s_sigma_p = 0.7, 1.3
        cov_mx, cov_mx_inv = dt.gen_cov_mx_4lh_noex(phi, s_sigma_n, s_sigma_p)
        ref_cov_mx, ref_inv = _ref_cov_mx_inv(phi, s_sigma_n, s_sigma_p)
        np.testing.assert_allclose(cov_mx, ref_cov_mx, atol=TOL)
        np.testing.assert_allclose(cov_mx_inv, ref_inv, atol=1e-6)


def test_gen_cov_mx_4lh_noex_identity_property():
    """cov_mx @ cov_mx_inv should be (approximately) the identity."""
    np.random.seed(3)
    phi = np.random.randn(100, 5)
    s_sigma_n, s_sigma_p = 0.5, 2.0
    cov_mx, cov_mx_inv = dt.gen_cov_mx_4lh_noex(phi, s_sigma_n, s_sigma_p)
    n_s = phi.shape[0]
    np.testing.assert_allclose(cov_mx @ cov_mx_inv, np.eye(n_s), atol=1e-6)


def test_gen_cov_mx_4lh_variants_agree():
    """gen_cov_mx_4lh and gen_cov_mx_4lh_noex should agree with each other."""
    np.random.seed(4)
    phi = np.random.randn(60, 6)
    s_sigma_n, s_sigma_p = 0.3, 1.7
    cov_a, inv_a = dt.gen_cov_mx_4lh(phi, s_sigma_n, s_sigma_p)
    cov_b, inv_b = dt.gen_cov_mx_4lh_noex(phi, s_sigma_n, s_sigma_p)
    np.testing.assert_allclose(cov_a, cov_b, atol=TOL)
    np.testing.assert_allclose(inv_a, inv_b, atol=1e-6)


def test_gen_cov_mx_4lh_p_greater_than_n():
    """Direct-Cholesky path (p >= n) should still be correct (few samples)."""
    np.random.seed(5)
    phi = np.random.randn(3, 10)
    s_sigma_n, s_sigma_p = 0.5, 2.0
    cov_mx, cov_mx_inv = dt.gen_cov_mx_4lh(phi, s_sigma_n, s_sigma_p)
    ref_cov_mx, ref_inv = _ref_cov_mx_inv(phi, s_sigma_n, s_sigma_p)
    np.testing.assert_allclose(cov_mx, ref_cov_mx, atol=TOL)
    np.testing.assert_allclose(cov_mx_inv, ref_inv, atol=1e-6)


def test_gen_cov_mx_4lh_noex_p_greater_than_n():
    """Direct-Cholesky path (p >= n) should be correct for gen_cov_mx_4lh_noex."""
    np.random.seed(6)
    phi = np.random.randn(3, 10)
    s_sigma_n, s_sigma_p = 0.5, 2.0
    cov_mx, cov_mx_inv = dt.gen_cov_mx_4lh_noex(phi, s_sigma_n, s_sigma_p)
    ref_cov_mx, ref_inv = _ref_cov_mx_inv(phi, s_sigma_n, s_sigma_p)
    np.testing.assert_allclose(cov_mx, ref_cov_mx, atol=TOL)
    np.testing.assert_allclose(cov_mx_inv, ref_inv, atol=1e-6)


def test_gen_cov_mx_4lh_p_equals_n_boundary():
    """p == n boundary (routed to the direct-Cholesky path) is correct
    for both gen_cov_mx_4lh and gen_cov_mx_4lh_noex."""
    np.random.seed(7)
    phi = np.random.randn(8, 8)
    s_sigma_n, s_sigma_p = 0.4, 1.1
    ref_cov_mx, ref_inv = _ref_cov_mx_inv(phi, s_sigma_n, s_sigma_p)

    cov_a, inv_a = dt.gen_cov_mx_4lh(phi, s_sigma_n, s_sigma_p)
    np.testing.assert_allclose(cov_a, ref_cov_mx, atol=TOL)
    np.testing.assert_allclose(inv_a, ref_inv, atol=1e-6)

    cov_b, inv_b = dt.gen_cov_mx_4lh_noex(phi, s_sigma_n, s_sigma_p)
    np.testing.assert_allclose(cov_b, ref_cov_mx, atol=TOL)
    np.testing.assert_allclose(inv_b, ref_inv, atol=1e-6)
