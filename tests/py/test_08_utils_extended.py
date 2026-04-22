"""
Extended unit tests for aMRPC/utils.py

Covers functions not tested in test_01_utils.py:
- gen_multi_idx_old, gen_multi_idx_old1 (legacy multi-index generators)
- gen_midx_mask, gen_midx_mask_part, gen_midx_mask_hyp
- midx4quad
- gen_dict_key, get_dict_entry
- compare_multi_key_for_idx
- gen_corr_rcf
- get_multi_entry
- choose_cols
- inv_src_arr
"""
import numpy as np
from scipy.special import comb
import context
import aMRPC.utils as u

NO = 3
DIM = 3
NR = 2


# --- Multi-index generation (legacy versions) ---

def test_gen_multi_idx_old_shape():
    """gen_multi_idx_old produces the correct number of multi-indices."""
    alphas = u.gen_multi_idx_old(NO, DIM)
    expected = int(comb(NO + DIM, DIM, exact=True))
    assert alphas.shape == (expected, DIM)


def test_gen_multi_idx_old1_shape():
    """gen_multi_idx_old1 produces the correct number of multi-indices."""
    alphas = u.gen_multi_idx_old1(NO, DIM)
    expected = int(comb(NO + DIM, DIM, exact=True))
    assert alphas.shape == (expected, DIM)


def test_gen_multi_idx_old_matches_current():
    """Legacy gen_multi_idx_old matches current gen_multi_idx (same row set)."""
    alphas_new = u.gen_multi_idx(NO, DIM)
    alphas_old = u.gen_multi_idx_old(NO, DIM)
    # Both should have the same shape
    assert alphas_new.shape == alphas_old.shape
    # Both should contain the same set of multi-indices (order may differ)
    set_new = set(tuple(row) for row in alphas_new)
    set_old = set(tuple(row) for row in alphas_old)
    assert set_new == set_old


def test_gen_multi_idx_old1_matches_current():
    """Legacy gen_multi_idx_old1 matches current gen_multi_idx (same row set)."""
    alphas_new = u.gen_multi_idx(NO, DIM)
    alphas_old1 = u.gen_multi_idx_old1(NO, DIM)
    assert alphas_new.shape == alphas_old1.shape
    set_new = set(tuple(row) for row in alphas_new)
    set_old1 = set(tuple(row) for row in alphas_old1)
    assert set_new == set_old1


def test_gen_multi_idx_dim1():
    """1-D multi-index should be [0, 1, ..., NO]."""
    alphas = u.gen_multi_idx(NO, 1)
    assert alphas.shape == (NO + 1, 1)
    for i in range(NO + 1):
        assert alphas[i, 0] == i


# --- Multi-index masks ---

def test_gen_midx_mask_full():
    """Mask with no_max == NO should keep all indices."""
    alphas = u.gen_multi_idx(NO, DIM)
    mask = u.gen_midx_mask(alphas, NO)
    assert mask.all()


def test_gen_midx_mask_zero():
    """Mask with no_max == 0 should keep only the zero index."""
    alphas = u.gen_multi_idx(NO, DIM)
    mask = u.gen_midx_mask(alphas, 0)
    assert mask.sum() == 1
    assert alphas[mask][0].sum() == 0


def test_gen_midx_mask_partial():
    """Mask with no_max=1 should only include indices with sum <= 1."""
    alphas = u.gen_multi_idx(NO, DIM)
    mask = u.gen_midx_mask(alphas, 1)
    for i in range(len(mask)):
        if mask[i]:
            assert alphas[i].sum() <= 1
        else:
            assert alphas[i].sum() > 1


def test_gen_midx_mask_part():
    """Partial mask should filter correctly on subset of dimensions."""
    alphas = u.gen_multi_idx(NO, DIM)
    no_min = 0
    no_max = NO
    idx_set = np.array([0], dtype=np.int64)
    mask = u.gen_midx_mask_part(alphas, no_min, no_max, idx_set)
    for i in range(len(mask)):
        if mask[i]:
            assert alphas[i].sum() <= no_max
            # Dimensions not in idx_set should have degree <= no_min
            for d in range(DIM):
                if d not in idx_set:
                    assert alphas[i, d] <= no_min


def test_gen_midx_mask_hyp():
    """Hyperbolic truncation mask with p_norm=1 should match total degree."""
    alphas = u.gen_multi_idx(NO, DIM)
    mask_hyp = u.gen_midx_mask_hyp(alphas, NO, 1.0)
    mask_std = u.gen_midx_mask(alphas, NO)
    # L1-norm mask should match standard total degree mask
    np.testing.assert_array_equal(mask_hyp, mask_std)


def test_gen_midx_mask_hyp_low_pnorm():
    """Hyperbolic truncation with p_norm < 1 should be more restrictive."""
    alphas = u.gen_multi_idx(NO, DIM)
    mask_full = u.gen_midx_mask(alphas, NO)
    mask_hyp = u.gen_midx_mask_hyp(alphas, NO, 0.5)
    # Hyperbolic should be a subset of total degree
    assert mask_hyp.sum() <= mask_full.sum()
    for i in range(len(mask_hyp)):
        if mask_hyp[i]:
            assert mask_full[i]


# --- midx4quad ---

def test_midx4quad_shape():
    """midx4quad produces tensor product index matrix of correct shape."""
    lens = np.array([3, 4, 2], dtype=np.int64)
    idx_mx = u.midx4quad(lens)
    assert idx_mx.shape == (24, 3)  # 3*4*2=24


def test_midx4quad_values():
    """midx4quad indices stay within bounds."""
    lens = np.array([3, 4], dtype=np.int64)
    idx_mx = u.midx4quad(lens)
    for row in idx_mx:
        assert 0 <= row[0] < 3
        assert 0 <= row[1] < 4


def test_midx4quad_unique():
    """midx4quad generates unique index combinations."""
    lens = np.array([2, 3, 2], dtype=np.int64)
    idx_mx = u.midx4quad(lens)
    rows = set(tuple(row) for row in idx_mx)
    assert len(rows) == idx_mx.shape[0]


# --- Dictionary key functions ---

def test_gen_dict_key_with_src():
    """gen_dict_key with src produces a 3-tuple."""
    key = u.gen_dict_key(2, 1, 0)
    assert key == (2, 1, 0)
    assert len(key) == 3


def test_gen_dict_key_without_src():
    """gen_dict_key without src produces a 2-tuple."""
    key = u.gen_dict_key(2, 1)
    assert key == (2, 1)
    assert len(key) == 2


def test_get_dict_entry():
    """get_dict_entry retrieves correct dictionary entry."""
    d = {}
    key = u.gen_dict_key(2, 1, 0)
    d[key] = 42
    assert u.get_dict_entry(d, 2, 1, 0) == 42


def test_get_multi_entry():
    """get_multi_entry retrieves correct dictionary entry from multi-key."""
    srcs = [0, 1]
    anrs = [NR, NR]
    nris = [0, 0]
    mk = u.gen_multi_key(anrs, nris, srcs)
    d = {mk: "test_value"}
    assert u.get_multi_entry(d, anrs, nris, srcs) == "test_value"


# --- compare_multi_key_for_idx ---

def test_compare_multi_key_for_idx_equal():
    """Identical multi-keys should compare equal on all indices."""
    srcs = [0, 1, 2]
    anrs = [NR] * 3
    nris = [0, 0, 0]
    mk = u.gen_multi_key(anrs, nris, srcs)
    assert u.compare_multi_key_for_idx(mk, mk, [0, 1, 2])


def test_compare_multi_key_for_idx_partial():
    """Multi-keys differing in one dimension should compare equal on others."""
    srcs = [0, 1, 2]
    anrs = [NR] * 3
    nris_a = [0, 0, 0]
    nris_b = [0, 1, 0]
    mk_a = u.gen_multi_key(anrs, nris_a, srcs)
    mk_b = u.gen_multi_key(anrs, nris_b, srcs)
    # Dimensions 0 and 2 are the same
    assert u.compare_multi_key_for_idx(mk_a, mk_b, [0, 2])
    # Dimension 1 differs
    assert not u.compare_multi_key_for_idx(mk_a, mk_b, [1])


# --- gen_corr_rcf ---

def test_gen_corr_rcf_nr0():
    """At Nr=0 for all sources, correction coefficient should be 1."""
    srcs = [0, 1]
    anrs = [0, 0]
    nris = [0, 0]
    mk = u.gen_multi_key(anrs, nris, srcs)
    cf = u.gen_corr_rcf(mk, [0, 1])
    assert abs(cf - 1.0) < 1e-15


def test_gen_corr_rcf_partial():
    """gen_corr_rcf only considers specified sources."""
    srcs = [0, 1]
    anrs = [NR, NR]
    nris = [0, 0]
    mk = u.gen_multi_key(anrs, nris, srcs)
    cf_all = u.gen_corr_rcf(mk, [0, 1])
    cf_one = u.gen_corr_rcf(mk, [0])
    # cf_all = 2^(-NR) * 2^(-NR), cf_one = 2^(-NR)
    assert abs(cf_all - 2.0**(-2 * NR)) < 1e-15
    assert abs(cf_one - 2.0**(-NR)) < 1e-15


# --- choose_cols ---

def test_choose_cols_basic():
    """choose_cols selects specified columns."""
    arr = np.arange(12).reshape(3, 4)
    result = u.choose_cols(arr, [0, 2])
    expected = arr[:, [0, 2]]
    np.testing.assert_array_equal(result, expected)


def test_choose_cols_with_zero():
    """choose_cols zeros out specified columns."""
    arr = np.arange(12).reshape(3, 4).astype(float)
    result = u.choose_cols(arr, [0, 1, 2], zero_cols=[1])
    assert result[:, 1].sum() == 0
    np.testing.assert_array_equal(result[:, 0], arr[:, 0])


# --- inv_src_arr ---

def test_inv_src_arr():
    """inv_src_arr builds correct position dictionary."""
    srcs = [2, 5, 7]
    isrc = u.inv_src_arr(srcs)
    assert isrc[2] == 0
    assert isrc[5] == 1
    assert isrc[7] == 2
    assert len(isrc) == 3


def test_inv_src_arr_consecutive():
    """inv_src_arr for consecutive sources."""
    srcs = [0, 1, 2, 3]
    isrc = u.inv_src_arr(srcs)
    for i, s in enumerate(srcs):
        assert isrc[s] == i


# --- gen_nri_range_4mkset ---

def test_gen_nri_range_4mkset():
    """gen_nri_range_4mkset reconstructs Nri array from multi-key set."""
    srcs = [0, 1]
    dim = len(srcs)
    anrs = [NR] * dim
    nris_arr, nri_cnt = u.gen_nri_range(anrs)
    # Build multi-key set
    mkey_set = set()
    for i in range(nri_cnt):
        mk = u.gen_multi_key(anrs, nris_arr[i], srcs)
        mkey_set.add(mk)

    nris_from_set, cnt_from_set = u.gen_nri_range_4mkset(mkey_set, dim)
    assert cnt_from_set == nri_cnt
    # Same column sums
    sum_orig = np.sort(nris_arr.sum(axis=0))
    sum_set = np.sort(nris_from_set.sum(axis=0))
    np.testing.assert_array_equal(sum_orig, sum_set)
