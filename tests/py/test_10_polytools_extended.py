"""
Extended unit tests for aMRPC/polytools.py

Covers functions not directly tested in test_04_genPols.py:
- moment
- apc_cfs / pc_cfs directly
- cmp_alpha_beta
- Jacobi_mx
- cmp_Grw
- gen_Gw
- cmp_norm_cf / cmp_norm_cf_moments
- uniHank
- norm_hank
- pc_eval (boundary cases)
"""
import numpy as np
from scipy.stats import norm as scipy_norm
import context
import aMRPC.polytools as pt

TOL = 1e-7


# --- moment ---

def test_moment_zeroth():
    """0th raw moment of any dataset is 1."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(pt.moment(0, data) - 1.0) < TOL


def test_moment_first():
    """1st raw moment is the mean."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    expected = data.mean()
    assert abs(pt.moment(1, data) - expected) < TOL


def test_moment_second():
    """2nd raw moment is mean of squares."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.mean(data**2)
    assert abs(pt.moment(2, data) - expected) < TOL


# --- Hankel matrix ---

def test_hankel_symmetry():
    """Hankel matrix should be symmetric."""
    data = np.random.randn(500)
    H = pt.Hankel(4, data)
    np.testing.assert_allclose(H, H.T, atol=TOL)


def test_hankel_shape():
    """Hankel matrix should be (m+1) x (m+1)."""
    m = 5
    data = np.random.randn(500)
    H = pt.Hankel(m, data)
    assert H.shape == (m + 1, m + 1)


def test_hankel_h00_is_one():
    """H[0,0] = E[x^0] = 1."""
    data = np.random.randn(1000)
    H = pt.Hankel(3, data)
    assert abs(H[0, 0] - 1.0) < TOL


# --- uniHank (analytical Hankel for uniform distribution) ---

def test_uniHank_standard():
    """uniHank for U(0,1) should match data-based Hankel for large sample."""
    n = 4
    np.random.seed(42)
    data = np.random.uniform(0, 1, 100000)
    H_data = pt.Hankel(n, data)
    H_uni = pt.uniHank(n, 0.0, 1.0)
    np.testing.assert_allclose(H_data, H_uni, atol=1e-2)


def test_uniHank_symmetry():
    """Analytical uniform Hankel matrix should be symmetric."""
    H = pt.uniHank(5, 0.0, 1.0)
    np.testing.assert_allclose(H, H.T, atol=TOL)


def test_uniHank_shape():
    """uniHank should return (n+1) x (n+1) matrix."""
    n = 6
    H = pt.uniHank(n)
    assert H.shape == (n + 1, n + 1)


# --- norm_hank (analytical Hankel for normal distribution) ---

def test_norm_hank_symmetry():
    """Analytical normal Hankel matrix should be symmetric."""
    H = pt.norm_hank(5, 0.0, 1.0)
    np.testing.assert_allclose(H, H.T, atol=TOL)


def test_norm_hank_h00_is_one():
    """H[0,0] for standard normal should be 1."""
    H = pt.norm_hank(4, 0.0, 1.0)
    assert abs(H[0, 0] - 1.0) < TOL


def test_norm_hank_matches_data():
    """norm_hank for N(0,1) should match data-based Hankel for large sample."""
    n = 3
    np.random.seed(42)
    data = np.random.randn(100000)
    H_data = pt.Hankel(n, data)
    H_norm = pt.norm_hank(n, 0.0, 1.0)
    np.testing.assert_allclose(H_data, H_norm, atol=5e-2)


# --- pc_cfs and apc_cfs ---

def test_pc_cfs_degree_zero():
    """Degree-0 polynomial should be constant [1, 0, ...]."""
    H = pt.uniHank(4)
    cfs = pt.pc_cfs(H, 0, 5)
    assert abs(cfs[0] - 1.0) < TOL
    for i in range(1, len(cfs)):
        assert abs(cfs[i]) < TOL


def test_apc_cfs_degree_zero():
    """aPC degree-0 polynomial should be constant [1, 0, ...]."""
    H = pt.uniHank(4)
    cfs = pt.apc_cfs(H, 0, 5)
    assert abs(cfs[0] - 1.0) < TOL
    for i in range(1, len(cfs)):
        assert abs(cfs[i]) < TOL


def test_pc_cfs_monic():
    """Leading coefficient of pc_cfs for degree k should be 1."""
    H = pt.uniHank(5)
    for k in range(1, 5):
        cfs = pt.pc_cfs(H, k, 5)
        assert abs(cfs[k] - 1.0) < TOL


# --- cmp_alpha_beta and Jacobi matrix ---

def test_cmp_alpha_beta_shape():
    """Alpha and beta arrays should have length k."""
    H = pt.uniHank(5)
    alpha, beta = pt.cmp_alpha_beta(H, 4)
    assert alpha.shape == (4,)
    assert beta.shape == (4,)


def test_jacobi_mx_symmetry():
    """Jacobi matrix should be symmetric."""
    H = pt.uniHank(5)
    alpha, beta = pt.cmp_alpha_beta(H, 4)
    J = pt.Jacobi_mx(alpha, beta)
    np.testing.assert_allclose(J, J.T, atol=TOL)


# --- cmp_Grw (Gauss quadrature roots and weights) ---

def test_cmp_Grw_positive_weights():
    """Gaussian quadrature weights should be positive."""
    H = pt.uniHank(5)
    r, w = pt.cmp_Grw(H)
    assert all(w > 0)


def test_cmp_Grw_weights_sum():
    """For U(0,1), sum of weights should equal 1 (zeroth moment)."""
    H = pt.uniHank(5)
    r, w = pt.cmp_Grw(H)
    assert abs(sum(w) - 1.0) < TOL


def test_cmp_Grw_roots_in_range():
    """For U(0,1), roots should lie in [0,1]."""
    H = pt.uniHank(5)
    r, w = pt.cmp_Grw(H)
    assert all(r >= -TOL)
    assert all(r <= 1.0 + TOL)


# --- gen_Gw ---

def test_gen_Gw_matches_Grw():
    """gen_Gw weights should match cmp_Grw weights."""
    H = pt.uniHank(5)
    r, w = pt.cmp_Grw(H)
    w2 = pt.gen_Gw(H[0, :], r)
    np.testing.assert_allclose(np.sort(w), np.sort(w2), atol=TOL)


# --- gen_rw (combined roots and weights generation) ---

def test_gen_rw_method0():
    """gen_rw method 0 produces roots and weights with correct count."""
    H = pt.uniHank(5)
    r, w = pt.gen_rw(H, method=0)
    assert len(r) == len(w)
    assert len(r) > 0


def test_gen_rw_method1():
    """gen_rw method 1 produces roots and weights with correct count."""
    H = pt.uniHank(5)
    r, w = pt.gen_rw(H, method=1)
    assert len(r) == len(w)
    assert len(r) > 0


# --- Normalization coefficients ---

def test_cmp_norm_cf_positive():
    """Normalization coefficient should be positive."""
    H = pt.uniHank(4)
    cf_mx = pt.gen_pc_mx(H, 0)
    r, w = pt.gen_rw(H, 0)
    for k in range(cf_mx.shape[0]):
        nc = pt.cmp_norm_cf(cf_mx[k, :], r, w)
        assert nc > 0


def test_cmp_norm_cf_moments_matches_quad():
    """Normalization via moments should match normalization via quadrature."""
    H = pt.uniHank(4)
    cf_mx = pt.gen_pc_mx(H, 1)  # aPC method
    r, w = pt.gen_rw(H, 1)
    for k in range(cf_mx.shape[0]):
        nc_quad = pt.cmp_norm_cf(cf_mx[k, :], r, w)
        nc_mm = pt.cmp_norm_cf_moments(cf_mx[k, :], H)
        assert abs(nc_quad - nc_mm) < 1e-4


# --- pc_eval ---

def test_pc_eval_constant():
    """Evaluating constant polynomial should return constant."""
    cfs = np.array([3.0, 0.0, 0.0])
    x = np.array([0.1, 0.5, 0.9])
    result = pt.pc_eval(cfs, x)
    np.testing.assert_allclose(result, 3.0, atol=TOL)


def test_pc_eval_linear():
    """Evaluating linear polynomial c0 + c1*x."""
    cfs = np.array([2.0, 3.0])
    x = np.array([0.0, 1.0, 2.0])
    result = pt.pc_eval(cfs, x)
    expected = 2.0 + 3.0 * x
    np.testing.assert_allclose(result, expected, atol=TOL)


def test_pc_eval_quadratic():
    """Evaluating quadratic polynomial c0 + c1*x + c2*x^2."""
    cfs = np.array([1.0, -2.0, 3.0])
    x = np.array([0.0, 1.0, -1.0, 2.0])
    result = pt.pc_eval(cfs, x)
    expected = 1.0 - 2.0 * x + 3.0 * x**2
    np.testing.assert_allclose(result, expected, atol=TOL)


# --- Orthonormality check for uniform Hankel ---

def test_orthonormality_uniHank():
    """Normalized polynomials from uniHank should be orthonormal."""
    No = 4
    H = pt.uniHank(No + 1)
    cf_mx = pt.gen_pc_mx(H, 0, No)
    r, w = pt.gen_rw(H, 0, No)
    ncf = pt.gen_npc_mx(cf_mx, r, w, No)
    for i in range(No):
        for j in range(No):
            pi = pt.pc_eval(ncf[i, :], r)
            pj = pt.pc_eval(ncf[j, :], r)
            q = (pi * pj) @ w
            if i == j:
                assert abs(q - 1) < 1e-5
            else:
                assert abs(q) < 1e-5
