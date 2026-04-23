"""
Extended unit tests for aMRPC/wavetools.py

Covers functions not tested in test_03_genWVlets.py:
- WaveTools constructor parameters
- step_one_ms (alternative step_one)
- sfr, fr (scalar/vector wavelet evaluation)
- bd_chk (boundary checking)
- set_new_lrb (domain change)
- cmp_lrbi (sub-interval boundaries)
- resc_cf, rq_cf (rescaling coefficients)
- cmp_resc_details
- cmp_data_on_roots
"""
import numpy as np
import pandas as pd
import context
import aMRPC.wavetools as wt

P = 3
TOL = 1e-9


def gen_wv(deg=P, qdeg=-1):
    """Helper to generate initialized WaveTools."""
    wv = wt.WaveTools(deg, qdeg)
    wv.genWVlets()
    return wv


# --- Constructor ---

def test_constructor_defaults():
    """WaveTools constructor sets correct defaults."""
    wv = wt.WaveTools(P)
    assert wv.P == P
    assert wv.n == 2 * (P + 1)  # default qdeg
    assert wv.lb == 0
    assert wv.rb == 1
    assert wv.len == 1


def test_constructor_custom_qdeg():
    """WaveTools with custom quadrature degree."""
    wv = wt.WaveTools(P, qdeg=20)
    assert wv.n == 20


# --- step_one_ms (Markus Schmidgall variant) ---

def test_genWVlets_ms_orthonormal():
    """Multi-wavelets generated with MS variant should be orthonormal."""
    wv = wt.WaveTools(P)
    wv.genWVlets(ms_type=True)
    for i in range(P):
        for j in range(P):
            q = np.dot((wv.fpsi(i, wv.roots) * wv.fpsi(j, wv.roots)),
                       wv.weights)
            if i == j:
                assert abs(q - 1) < TOL
            else:
                assert abs(q) < TOL


def test_genWVlets_ms_polynomial_orthogonality():
    """MS-generated wavelets should satisfy <x^i, psi_j> = 0."""
    wv = wt.WaveTools(P)
    wv.genWVlets(ms_type=True)
    x = wv.roots
    for i in range(P):
        for j in range(P):
            q = (x**i) * wv.fpsi(j, x) @ wv.weights
            assert abs(q) < TOL


# --- sfr / fr (scalar/vector evaluation) ---

def test_sfr_scalar():
    """sfr evaluates wavelet at a single scalar point."""
    wv = gen_wv()
    x = 0.3
    for i in range(P):
        val = wv.sfr(i, x)
        assert isinstance(val, (float, np.floating))


def test_fr_scalar_matches_sfr():
    """fr with scalar input should match sfr."""
    wv = gen_wv()
    x = 0.7
    for i in range(P):
        assert abs(wv.fr(i, x) - wv.sfr(i, x)) < TOL


def test_fr_vector():
    """fr with vector input returns array of correct length."""
    wv = gen_wv()
    x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    for i in range(P):
        result = wv.fr(i, x)
        assert len(result) == len(x)


def test_fr_vector_matches_scalar():
    """fr vector output should match element-wise scalar calls."""
    wv = gen_wv()
    x = np.array([0.2, 0.5, 0.8])
    for i in range(P):
        vec_result = wv.fr(i, x)
        for j, xj in enumerate(x):
            scalar_result = wv.sfr(i, xj)
            assert abs(vec_result[j] - scalar_result) < TOL


# --- bd_chk (boundary checking) ---

def test_bd_chk_scalar_inside():
    """bd_chk returns True for point inside sub-interval."""
    wv = gen_wv()
    Nr, Nri = 2, 1
    lbi, rbi, _ = wv.cmp_lrbi(Nr, Nri)
    mid = (lbi + rbi) / 2
    assert wv.bd_chk(mid, Nr, Nri)


def test_bd_chk_scalar_outside():
    """bd_chk returns False for point outside sub-interval."""
    wv = gen_wv()
    Nr, Nri = 2, 0
    _, rbi, _ = wv.cmp_lrbi(Nr, Nri)
    assert not wv.bd_chk(rbi + 0.1, Nr, Nri)


def test_bd_chk_vector():
    """bd_chk returns boolean array for vector input."""
    wv = gen_wv()
    Nr, Nri = 1, 0  # [0, 0.5]
    x = np.array([0.1, 0.3, 0.6, 0.9])
    result = wv.bd_chk(x, Nr, Nri)
    assert result[0] == True
    assert result[1] == True
    assert result[2] == False
    assert result[3] == False


# --- cmp_lrbi (sub-interval boundaries) ---

def test_cmp_lrbi_nr0():
    """At Nr=0, Nri=0, interval should be [0,1]."""
    wv = gen_wv()
    lbi, rbi, scf = wv.cmp_lrbi(0, 0)
    assert abs(lbi - 0.0) < TOL
    assert abs(rbi - 1.0) < TOL
    assert abs(scf - 1.0) < TOL


def test_cmp_lrbi_nr1():
    """At Nr=1, two sub-intervals should cover [0,1]."""
    wv = gen_wv()
    lbi0, rbi0, scf0 = wv.cmp_lrbi(1, 0)
    lbi1, rbi1, scf1 = wv.cmp_lrbi(1, 1)
    assert abs(lbi0 - 0.0) < TOL
    assert abs(rbi0 - 0.5) < TOL
    assert abs(lbi1 - 0.5) < TOL
    assert abs(rbi1 - 1.0) < TOL
    assert abs(scf0 - 0.5) < TOL


def test_cmp_lrbi_coverage():
    """All sub-intervals at given Nr should cover [0,1] without gaps."""
    wv = gen_wv()
    Nr = 3
    prev_rb = 0.0
    for nri in range(2**Nr):
        lbi, rbi, _ = wv.cmp_lrbi(Nr, nri)
        assert abs(lbi - prev_rb) < TOL
        prev_rb = rbi
    assert abs(prev_rb - 1.0) < TOL


# --- set_new_lrb ---

def test_set_new_lrb():
    """set_new_lrb changes domain bounds correctly."""
    wv = gen_wv()
    wv.set_new_lrb(-1, 2)
    assert wv.lb == -1
    assert wv.rb == 2
    assert abs(wv.len - 3.0) < TOL


# --- resc_cf and rq_cf ---

def test_resc_cf_nr0():
    """At Nr=0, wavelet rescaling coefficient should be 1/sqrt(len)."""
    wv = gen_wv()
    assert abs(wv.resc_cf(0) - 1.0) < TOL  # len=1, so 1/sqrt(1) = 1


def test_resc_cf_scaling():
    """resc_cf should scale as 2^(Nr/2)."""
    wv = gen_wv()
    for Nr in range(5):
        expected = 2**(Nr / 2)
        assert abs(wv.resc_cf(Nr) - expected) < TOL


def test_rq_cf_nr0():
    """At Nr=0, quadrature coefficient should be len = 1."""
    wv = gen_wv()
    assert abs(wv.rq_cf(0) - 1.0) < TOL


def test_rq_cf_scaling():
    """rq_cf should scale as 1/2^Nr."""
    wv = gen_wv()
    for Nr in range(5):
        expected = 1.0 / (2**Nr)
        assert abs(wv.rq_cf(Nr) - expected) < TOL


# --- cmp_details / cmp_resc_details ---

def test_cmp_details_single():
    """cmp_details with specific No returns scalar."""
    wv = gen_wv()
    data = np.sin(2 * np.pi * wv.roots)
    for i in range(P):
        d = wv.cmp_details(data, i)
        assert isinstance(d, (float, np.floating))


def test_cmp_details_all():
    """cmp_details with No=-1 returns all P details."""
    wv = gen_wv()
    data = np.sin(2 * np.pi * wv.roots)
    details = wv.cmp_details(data)
    assert len(details) == P


def test_cmp_resc_details():
    """cmp_resc_details is rq_cf(Nr) * cmp_details."""
    wv = gen_wv()
    data = np.sin(2 * np.pi * wv.roots)
    Nr = 2
    resc_details = wv.cmp_resc_details(data, Nr)
    details = wv.cmp_details(data)
    expected = wv.rq_cf(Nr) * details
    np.testing.assert_allclose(resc_details, expected, atol=TOL)


# --- cmp_data_on_roots ---

def test_cmp_data_on_roots():
    """cmp_data_on_roots computes quantiles at wavelet roots."""
    wv = gen_wv()
    # Create a simple dataset
    np.random.seed(42)
    data = pd.Series(np.random.randn(1000))
    qor = wv.cmp_data_on_roots(data, 0, 0)
    assert len(qor) == wv.n
    # Quantiles should be monotonically increasing for sorted roots
    sorted_roots = np.sort(wv.roots)
    sorted_quantiles = data.quantile(sorted_roots)
    for i in range(1, len(sorted_quantiles)):
        assert sorted_quantiles.iloc[i] >= sorted_quantiles.iloc[i - 1]


# --- rfpsi (rescaled multi-wavelet) ---

def test_rfpsi_outside_zero():
    """rfpsi should return 0 for points outside the sub-interval."""
    wv = gen_wv()
    Nr, Nri = 2, 0  # [0, 0.25]
    x_outside = 0.5
    for i in range(P):
        assert abs(wv.rfpsi(x_outside, i, Nr, Nri)) < TOL


def test_rfpsi_vector_outside_zero():
    """rfpsi vector should be 0 for points outside the sub-interval."""
    wv = gen_wv()
    Nr, Nri = 2, 0  # [0, 0.25]
    x = np.array([0.5, 0.75, 1.0])
    for i in range(P):
        result = wv.rfpsi(x, i, Nr, Nri)
        np.testing.assert_allclose(result, 0.0, atol=TOL)
