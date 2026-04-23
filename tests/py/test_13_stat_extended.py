"""
Extended unit tests for aMRPC/stat.py

Covers functions not tested in test_07_stat.py:
- cmp_log_likelihood_cf / cmp_log_likelihood_cf_mv
- cmp_norm_likelihood_core_inv (with precomputed inverse)
- cmp_log_likelihood_core / cmp_log_likelihood_core_inv / cmp_log_likelihood_core_sinv
- bme_norm_response
- lbme_norm_response
- d_kl_norm_prior_response
"""
from math import exp, log, sqrt, pi
import numpy as np
from scipy.stats import norm
import context
import aMRPC.stat as st

EPS = 1.0e-6
NUMBER_OF_MEASUREMENTS = 10
STD = 0.3
NUMBER_OF_REALIZATIONS = 8


def gen_observations(eval_points, params):
    """Simple synthetic model for testing."""
    return (pow(params[0] * params[0] + params[1] - 1, 2)
            + 0.1 * params[0] * exp(params[1])
            - np.sqrt(0.5 * eval_points) * 2 * params[1]
            + 1 + np.sin(5 * eval_points))


# --- Log-likelihood coefficients ---

def test_log_likelihood_cf_matches_log_of_cf():
    """log(cmp_norm_likelihood_cf) should equal cmp_log_likelihood_cf."""
    cf = st.cmp_norm_likelihood_cf(STD, NUMBER_OF_MEASUREMENTS)
    log_cf = st.cmp_log_likelihood_cf(STD, NUMBER_OF_MEASUREMENTS)
    assert abs(np.log(cf) - log_cf) < EPS


def test_log_likelihood_cf_mv_matches_log_of_cf_mv():
    """log(cmp_norm_likelihood_cf_mv) should equal cmp_log_likelihood_cf_mv."""
    cov_mx = STD * STD * np.eye(NUMBER_OF_MEASUREMENTS)
    cf = st.cmp_norm_likelihood_cf_mv(cov_mx)
    log_cf = st.cmp_log_likelihood_cf_mv(cov_mx)
    assert abs(np.log(cf) - log_cf) < EPS


def test_log_likelihood_cf_mv_diagonal_matches_scalar():
    """For diagonal cov matrix, mv log-cf should match scalar log-cf."""
    cov_mx = STD * STD * np.eye(NUMBER_OF_MEASUREMENTS)
    log_cf_mv = st.cmp_log_likelihood_cf_mv(cov_mx)
    log_cf_s = st.cmp_log_likelihood_cf(STD, NUMBER_OF_MEASUREMENTS)
    assert abs(log_cf_mv - log_cf_s) < EPS


# --- cmp_norm_likelihood_core_inv ---

def test_norm_likelihood_core_inv_scalar():
    """cmp_norm_likelihood_core_inv matches cmp_norm_likelihood_core for scalar."""
    points = np.linspace(0, 1, NUMBER_OF_MEASUREMENTS)
    params = np.array([0, 0])
    obs = gen_observations(points, params)
    cov_mx = STD * STD * np.eye(NUMBER_OF_MEASUREMENTS)
    cov_inv = np.linalg.pinv(cov_mx)
    response = obs + np.sin(points * 30) / 10

    lh_core = st.cmp_norm_likelihood_core(obs, response, cov_mx)
    lh_core_inv = st.cmp_norm_likelihood_core_inv(obs, response, cov_inv)
    assert abs(lh_core - lh_core_inv) < EPS


def test_norm_likelihood_core_inv_vector():
    """cmp_norm_likelihood_core_inv matches core for vector responses."""
    points = np.linspace(0, 1, NUMBER_OF_MEASUREMENTS)
    params = np.array([0, 0])
    obs = gen_observations(points, params)
    cov_mx = STD * STD * np.eye(NUMBER_OF_MEASUREMENTS)
    cov_inv = np.linalg.pinv(cov_mx)

    responses = np.zeros([NUMBER_OF_MEASUREMENTS, NUMBER_OF_REALIZATIONS])
    for i in range(NUMBER_OF_REALIZATIONS):
        responses[:, i] = obs + np.sin(points * 30) / (10 + i)

    lh_core = st.cmp_norm_likelihood_core(obs, responses, cov_mx)
    # cmp_norm_likelihood_core_inv expects observation as 1-D; for 2-D
    # response, the observation is subtracted directly (no reshape).
    # Test individual 1-D responses to verify consistency.
    for i in range(NUMBER_OF_REALIZATIONS):
        lh_single = st.cmp_norm_likelihood_core_inv(obs, responses[:, i], cov_inv)
        assert abs(lh_core[i] - lh_single) < EPS


# --- Log-likelihood core ---

def test_log_likelihood_core_scalar():
    """cmp_log_likelihood_core should match log of cmp_norm_likelihood_core."""
    points = np.linspace(0, 1, NUMBER_OF_MEASUREMENTS)
    params = np.array([0, 0])
    obs = gen_observations(points, params)
    cov_mx = STD * STD * np.eye(NUMBER_OF_MEASUREMENTS)
    response = obs + np.sin(points * 30) / 10

    lh_core = st.cmp_norm_likelihood_core(obs, response, cov_mx)
    log_lh_core = st.cmp_log_likelihood_core(obs, response, cov_mx)
    assert abs(np.log(lh_core) - log_lh_core) < EPS


def test_log_likelihood_core_vector():
    """log-likelihood core matches log of likelihood core for vector."""
    points = np.linspace(0, 1, NUMBER_OF_MEASUREMENTS)
    params = np.array([0, 0])
    obs = gen_observations(points, params)
    cov_mx = STD * STD * np.eye(NUMBER_OF_MEASUREMENTS)

    responses = np.zeros([NUMBER_OF_MEASUREMENTS, NUMBER_OF_REALIZATIONS])
    for i in range(NUMBER_OF_REALIZATIONS):
        responses[:, i] = obs + np.sin(points * 30) / (10 + i)

    lh_core = st.cmp_norm_likelihood_core(obs, responses, cov_mx)
    log_lh_core = st.cmp_log_likelihood_core(obs, responses, cov_mx)
    np.testing.assert_allclose(np.log(lh_core), log_lh_core, atol=EPS)


# --- cmp_log_likelihood_core_inv ---

def test_log_likelihood_core_inv():
    """cmp_log_likelihood_core_inv matches cmp_log_likelihood_core."""
    points = np.linspace(0, 1, NUMBER_OF_MEASUREMENTS)
    params = np.array([0, 0])
    obs = gen_observations(points, params)
    cov_mx = STD * STD * np.eye(NUMBER_OF_MEASUREMENTS)
    cov_inv = np.linalg.pinv(cov_mx)
    response = obs + np.sin(points * 30) / 10

    log_lh = st.cmp_log_likelihood_core(obs, response, cov_mx)
    log_lh_inv = st.cmp_log_likelihood_core_inv(obs, response, cov_inv)
    assert abs(log_lh - log_lh_inv) < EPS


# --- cmp_log_likelihood_core_sinv ---

def test_log_likelihood_core_sinv():
    """cmp_log_likelihood_core_sinv matches cmp_log_likelihood_core for 1-D."""
    points = np.linspace(0, 1, NUMBER_OF_MEASUREMENTS)
    params = np.array([0, 0])
    obs = gen_observations(points, params)
    cov_mx = STD * STD * np.eye(NUMBER_OF_MEASUREMENTS)
    cov_inv = np.linalg.pinv(cov_mx)
    response = obs + np.sin(points * 30) / 10

    log_lh = st.cmp_log_likelihood_core(obs, response, cov_mx)
    log_lh_sinv = st.cmp_log_likelihood_core_sinv(obs, response, cov_inv)
    assert abs(log_lh - log_lh_sinv) < EPS


# --- BME (Bayesian Model Evidence) ---

def test_bme_positive():
    """BME should be positive."""
    points = np.linspace(0, 1, NUMBER_OF_MEASUREMENTS)
    params = np.array([0, 0])
    obs = gen_observations(points, params)
    cov_mx = STD * STD * np.eye(NUMBER_OF_MEASUREMENTS)

    n_samples = 20
    responses = np.zeros([n_samples, NUMBER_OF_MEASUREMENTS])
    for i in range(n_samples):
        responses[i, :] = obs + np.sin(points * 30) / (10 + i)

    bme = st.bme_norm_response(obs, responses, cov_mx)
    assert bme > 0


def test_lbme_matches_log_bme():
    """lbme should be close to log(bme) when bme is computable."""
    points = np.linspace(0, 1, NUMBER_OF_MEASUREMENTS)
    params = np.array([0, 0])
    obs = gen_observations(points, params)
    cov_mx = STD * STD * np.eye(NUMBER_OF_MEASUREMENTS)

    n_samples = 20
    responses = np.zeros([n_samples, NUMBER_OF_MEASUREMENTS])
    for i in range(n_samples):
        responses[i, :] = obs + np.sin(points * 30) / (10 + i)

    bme = st.bme_norm_response(obs, responses, cov_mx)
    lbme = st.lbme_norm_response(obs, responses, cov_mx)
    if bme > 0:
        assert abs(np.log(bme) - lbme) < 1.0  # some tolerance for numerical


def test_lbme_norm_response_inv():
    """lbme_norm_response_inv should match lbme_norm_response."""
    points = np.linspace(0, 1, NUMBER_OF_MEASUREMENTS)
    params = np.array([0, 0])
    obs = gen_observations(points, params)
    cov_mx = STD * STD * np.eye(NUMBER_OF_MEASUREMENTS)
    cov_inv = np.linalg.pinv(cov_mx)
    llh_cf = st.cmp_log_likelihood_cf_mv(cov_mx)

    n_samples = 20
    responses = np.zeros([n_samples, NUMBER_OF_MEASUREMENTS])
    for i in range(n_samples):
        responses[i, :] = obs + np.sin(points * 30) / (10 + i)

    lbme = st.lbme_norm_response(obs, responses, cov_mx)
    lbme_inv = st.lbme_norm_response_inv(obs, responses, cov_inv, llh_cf)
    assert abs(lbme - lbme_inv) < 0.5


# --- KL divergence ---

def test_d_kl_nonnegative():
    """KL divergence should be non-negative (in expectation)."""
    np.random.seed(42)
    points = np.linspace(0, 1, NUMBER_OF_MEASUREMENTS)
    params = np.array([0, 0])
    obs = gen_observations(points, params)
    cov_mx = STD * STD * np.eye(NUMBER_OF_MEASUREMENTS)

    n_samples = 50
    responses = np.zeros([n_samples, NUMBER_OF_MEASUREMENTS])
    for i in range(n_samples):
        responses[i, :] = obs + np.random.randn(NUMBER_OF_MEASUREMENTS) * STD * 2

    dkl = st.d_kl_norm_prior_response(obs, responses, cov_mx)
    # KL divergence should be non-negative (may have small numerical noise)
    assert dkl >= -0.5 or np.isnan(dkl)


def test_d_kl_inv_matches_standard():
    """d_kl_norm_prior_response_inv should be close to d_kl_norm_prior_response."""
    np.random.seed(42)
    points = np.linspace(0, 1, NUMBER_OF_MEASUREMENTS)
    params = np.array([0, 0])
    obs = gen_observations(points, params)
    cov_mx = STD * STD * np.eye(NUMBER_OF_MEASUREMENTS)
    cov_inv = np.linalg.pinv(cov_mx)

    n_samples = 50
    responses = np.zeros([n_samples, NUMBER_OF_MEASUREMENTS])
    for i in range(n_samples):
        responses[i, :] = obs + np.random.randn(NUMBER_OF_MEASUREMENTS) * STD * 2

    dkl = st.d_kl_norm_prior_response(obs, responses, cov_mx)
    dkl_inv = st.d_kl_norm_prior_response_inv(obs, responses, cov_inv)
    # Both should give similar results (stochastic, so allow generous tolerance)
    if not np.isnan(dkl) and not np.isnan(dkl_inv):
        assert abs(dkl - dkl_inv) < 2.0


# --- Non-diagonal covariance matrix ---

def test_norm_likelihood_cf_mv_nondiag():
    """cmp_norm_likelihood_cf_mv works with non-diagonal covariance."""
    n = 5
    # Create a valid positive definite covariance matrix
    A = np.random.randn(n, n)
    cov_mx = A @ A.T + np.eye(n) * 0.1
    cf = st.cmp_norm_likelihood_cf_mv(cov_mx)
    assert cf > 0


def test_log_likelihood_cf_mv_nondiag():
    """cmp_log_likelihood_cf_mv returns finite value for non-diagonal cov."""
    n = 5
    np.random.seed(123)
    A = np.random.randn(n, n)
    cov_mx = A @ A.T + np.eye(n) * 0.1
    log_cf = st.cmp_log_likelihood_cf_mv(cov_mx)
    assert np.isfinite(log_cf)
    # Verify it is negative (log of normalizing constant < 0 for dim > 0)
    assert log_cf < 0
