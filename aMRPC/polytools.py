"""
polytools.py - generates aPC basis and related Gaussian quadrature

@author:: kroeker
https://orcid.org/0000-0003-0360-5307

"""
# import sys
import math
from scipy.stats import norm
import numpy as np
# from numpy.polynomial import polynomial as P
try:
    from numba import njit, prange # int64, jit, float64, int32 # , jit_module
    NJM = True
except ImportError:
    NJM = False
    pass


@njit(nogil=True, cache=True)
def moment(mm_, data):
    """
    computes mm-th raw moment

    Parameters
    ----------
    mm : int
        degree.
    data : np.array
        dataset containing samples.

    Returns
    -------
    float
        mm-th raw moment.

    """
    # print(mm)
    # return np.power(data, mm, dtype=np.float64).mean()
    return np.mean(data ** mm_)


def Hankel(m_mx, data):
    return Hankel_np(m_mx, np.array(data, dtype=np.float64))


@njit(nogil=True, parallel=True, cache=True)
def Hankel_np(m_mx, data):
    """
      Generates Hankel matrix for max. order m_mx for dateset given in data

    Parameters
    ----------
    m_mx : int
        max. order.
    data : np.array
        dataset.

    Returns
    -------
    H : np.array
        Hankel Matrix.

    """

    H = np.empty((m_mx+1, m_mx+1), dtype=np.float64)
    for i in prange(m_mx+1):
        for j in range(i, m_mx+1):
            H[i, j] = moment(i+j, data)
            H[j, i] = H[i, j]
    return H


def apc_cfs(H, k=-1, alen=-1):
    """
    generates monomial coefficients in increasing degree
    c_0 + c_1*x + c_2*x^2 + ...
    to obtain aPC orthogonal polynomial basis according to

    S. Oladyshkin, W. Nowak,
    Data-driven uncertainty quantification using the arbitrary polynomial chaos
    expansion,
    Reliability Engineering & System Safety,
    Volume 106,  2012, Pages 179-190, ISSN 0951-8320,
    https://doi.org/10.1016/j.ress.2012.05.002.

    Parameters
    ----------
    H : np.array
        Hankel matrix.
    k : int, optional
        highest pol. degree The default is -1.
    alen : int, optional
        length of the output array. The default is -1.

    Returns
    -------
    cfs : np.array
        polynomial coefficients.

    """

    l = H.shape[0]
    assert l >= alen
    if alen == -1:
        alen = l
    cfs = np.zeros(alen, dtype=np.float64)
    if k < 0:
        k = l-1
    elif k == 0:
        cfs[0] = 1
        return cfs
    assert k < l
    rH = np.copy(H[0:k+1, 0:k+1])
    rs = np.zeros(k+1, dtype=np.float64)
    rs[-1] = 1
    for j in range(k+1):
        rH[k, j] = 0
    rH[k, k] = 1
    # cfs = np.linalg.solve(rH, rs)
    cfs = np.linalg.lstsq(rH, rs, rcond=None)[0]
    # if not np.allclose(np.dot(rH, cfs), rs):
    #    cfs, _, _, _ = np.linalg.lstsq(rH, rs, rcond=-1)

    # assert np.allclose(np.dot(rH, cfs), rs), "problems with aPC coeffs"

    cfs.resize(alen, refcheck=False)  # np.ndarray.resize() -> not np.resize(a)

    return cfs


def pc_cfs(H, k=-1, alen=-1):
    """
    polynomial coefficientes in increasing order, computed via moment
    determinants.
    Gautschi, Walter
    Orthogonal polynomials: computation and approximation.
    Numerical Mathematics and Scientific Computation. Oxford Science
    Publications. Oxford University Press, New York, 2004. x+301 pp.
    ISBN: 0-19-850672-4 (p. 53)

    Parameters
    ----------
        H : np.array
        Hankel matrix.
    k : int, optional
        highest pol. degree The default is -1.
    alen : int, optional
        length of the output array. The default is -1.

    Returns
    -------
    cfs : np.array
        polynomial coefficients.
    """
    l = H.shape[0]
    assert alen <= l
    if alen == -1:
        alen = l
    cfs = np.zeros(alen)
    if k < 0:
        k = l-1
    elif k == 0:
        cfs[0] = 1
        return cfs
    assert k < l
    red_H = H[0:k, 0:k]
    idx = np.ones(l, dtype=bool)
    idx[k+1:l] = False
    delta = np.linalg.det(red_H)
    for i in range(0, k):
        idx[i] = False
        Hk = H[0:k, idx]
        dHk = np.linalg.det(Hk)
        cfs[i] = (-1)**(k+i) * dHk / delta
        idx[i] = True
    cfs[k] = 1
    return cfs


def cmp_alpha_beta(H, k=-1):
    """
    Generates vectors of recursion coefficientes alpha and beta

    Parameters
    ----------
    H : np.arary
        Hankel matrix.
    k : int, optional
        max. used degree The default is -1.

    Returns
    -------
    alpha : np.array
        rec. cf. alpha.
    beta : np.array
        rec. cf. beta.

    """

    n = H.shape[0]
    if k == -1:
        k = n-1
    assert k < n
    alpha = np.zeros(k)
    beta = np.zeros(k)
    delta = np.zeros(k+1)
    deltak = np.zeros(k+1)
    idx = np.zeros(n, dtype=bool)
    for l in range(1, k+1):
        # print(l)
        delta[l] = np.linalg.det(H[0:l, 0:l])
        if l > 1:
            idx[0:l-1] = True
            idx[l] = True
            deltak[l] = np.linalg.det(H[0:l, idx])
            idx[0:n] = False
    delta[0] = 1
    deltak[0] = 0
    deltak[1] = H[0, 1]
    for l in range(k):
        if l > 0:
            alpha[l] = deltak[l+1] / delta[l+1] - deltak[l] / delta[l]
            beta[l] = delta[l+1]*delta[l-1] / (delta[l]**2)
        else:
            alpha[l] = deltak[l+1] / delta[l+1]
            beta[l] = H[0, 0]
    return alpha, beta


def Jacobi_mx(alpha, beta):
    """
    Generates Jacobi matrix (for moments / polynomials)

    Parameters
    ----------
    alpha : np.array
        rec. cf. alpha
    beta : np.array
        rec. cf. beta.

    Returns
    -------
    J : np.array
        Jacobi matrix.

    """

    n = alpha.shape[0]
    m = beta.shape[0]
    assert m == n
    J = np.diag(alpha)
    for l in range(n-1):
        J[l, l+1] = math.sqrt(beta[l+1])
        J[l+1, l] = J[l, l+1]
    return J


def cmp_Grw(H, k=-1):
    """
    Computes roots and weigth of the Gauss quadrature using Hankel Matrix,
    Gautschi p. 153
    Gautschi, Walter
    Orthogonal polynomials: computation and approximation.
    Numerical Mathematics and Scientific Computation. Oxford Science
    Publications.
    Oxford University Press, New York, 2004. x+301 pp. ISBN: 0-19-850672-4

    Parameters
    ----------
    H : np.array
        Hankel matrix.
    k : int, optional
        max. pol. order. The default is -1.

    Returns
    -------
    roots : np.array
        roots of the related polynomials.
    weights : np.array
        related quadrature weights.

    """
    n = H.shape[0]
    assert k < n
    alpha, beta = cmp_alpha_beta(H, k)
    J = Jacobi_mx(alpha, beta)
    # print(J)
    tau, V = np.linalg.eig(J)
    roots = tau
    weights = beta[0] * (V[0, :]**2)
    # v=np.zeros(roots.shape[0])
    # for i in range(roots.shape[0]):
    #    v[i]=V[0,i]**2
    # weights=beta[0]*v
    return roots, weights


def gen_Gw(moments, roots):
    """
    Computes Gaussian weights using moments and roots, compare with
    Karniadakis & Kirby p.236

    Parameters
    ----------
    moments : np.array
        raw moments in increasing order.
    roots : np.array
        roots.

    Returns
    -------
    np.array
        Gaussian quadrature weights.

    """

    m = moments.shape[0]
    r = roots.shape[0]
    assert m >= r
    rs = moments[0:r]
    M = np.zeros([r, r])
    for i in range(r):
        for j in range(r):
            M[i, j] = roots[j]**i
    return np.linalg.solve(M, rs)


def cmp_norm_cf(cfs, roots, weights, eps=0):
    """
    Computes the normalizing factor of the polynomial w.r.t. Gauss quadrature

    Parameters
    ----------
    cfs : np.array
        monomial coeficients, defining polynomial.
    roots : np.array
        Gaussian quadrature roots.
    weights : TYPE
        Gaussian quadrature weights.
    eps : float, optional
        min value for the normalizing coefs. The default is 0.

    Returns
    -------
    float
        normalizing coefficient.

    """

    r = roots.shape[0]
    w = weights.shape[0]
    assert r == w
    c = np.flip(cfs, 0)
    p = np.poly1d(c)
    nc = 0
    for i in range(r):
        nc += (p(roots[i])**2)*weights[i]
    nc = max(nc, eps)  # ugly workarround, should be improved
    return math.sqrt(nc)


def cmp_norm_cf_moments(cfs, H_mx, eps=0):
    """
    Computes the normalizing factor of the polynomial w.r.t. Gauss quadrature
    using Hankel matrix only

    Parameters
    ----------
    cfs : np.array
        onomial coeficients, defining polynomial.
    H_mx : np.array
        Hankel matrix.
    eps : float, optional
        min value for the normalizing coefs. The default is 0.
    Returns
    -------
    float
        normalizing coefficient.

    """
    m = H_mx.shape[0]
    n = cfs.shape[0]
    assert n <= m
    ltwo_norm = 0
    for i in range(n):
        mx_line = cfs * H_mx[i, 0:n]
        ltwo_norm += cfs[i] * np.add.reduce(mx_line)

    ltwo_norm = max(ltwo_norm, eps)  # ugly workarround, should be improved
    return math.sqrt(ltwo_norm)


@njit(nogil=True, parallel=True, cache=True)
def uniHank(n, a=0.0, b=1.0):
    """
    Generates Hankel Matrix H_n for U(a,b),
    uses m_n=1/n+1 sum_k=0^n a^k b^(n-k)

    Parameters
    ----------
    n : int
        max order.
    a : float, optional
        lower bound. The default is 0.
    b : float, optional
        upper bound. The default is 1.

    Returns
    -------
    H : np.array
        Hankel matrix.

    """

    H = np.zeros((n+1, n+1), dtype=np.float64)
    lva = a*np.ones(2*n+1, dtype=np.float64)
    lvb = b*np.ones(2*n+1, dtype=np.float64)
    for i in range(2*n+1):
        lva[i] = lva[i]**i
        lvb[i] = lvb[i]**(2*n-i)
    for k in prange(n+1):
        for l in range(n+1):
            m = k+l
            va = lva[0:m+1]
            vb = lvb[2*n-m:]
            H[k, l] = np.dot(va, vb) / (m+1)
    return H


def norm_hank(n_mx, mu=0.0, sigma=1.0):
    """
    Generates Hankel Matrix H_n for N(mu, sigma),
    uses m_n=norm.moment(n, mu, sigma)
    
    Parameters
    ----------
    n_mx : int
        max order.
    mu : float, optional
        mean. The default is 0.
    sigma : float, optional
        standard deviation. The default is 1.
    
    Returns
    -------
    H : np.array
        Hankel matrix.
    """
    H = np.zeros((n_mx + 1, n_mx + 1), dtype=np.float64)
    for k_ in range(n_mx + 1):
        for l_ in range(k_, n_mx + 1):
            H[k_, l_] = norm.moment(k_ + l_, loc=mu, scale=sigma)
            H[l_, k_] = H[k_, l_]
    return H

def gen_pc_mx(H, method=0, No=-1):
    """
    Generates a matrix with polynomial coefficients up to degree No

    Parameters
    ----------
    H : np.array
        Hankel matrix.
    method : int, optional
        Method: 0 - Gautschi - style, 1- aPC style. The default is 0.
    No : int, optional
        max. pol. degree. The default is -1.

    Returns
    -------
    cf : np.array
        matrix with polynomial coeficients.

    """

    n = H.shape[0]
    assert No <= n
    if No < 0:
        No = n-1
    cf = np.zeros((No, No), dtype=np.float64)
    for k in range(No):
        if method == 0:
            cf[k, :] = pc_cfs(H, k, No)
        else:
            cf[k, :] = apc_cfs(H, k, No)
    return cf


def gen_rw(H, method=0, No=-1):
    """
    Generates roots and weights from Hankel matrix

    Parameters
    ----------
    H : np.array
        Hankel matrix.
    method : int, optional
        Method 0: Gautschi method, 1: aPC + moment. The default is 0.
    No : int, optional
        polynomial degree. The default is -1.

    Returns
    -------
    r : np.array
        roots.
    w : np.array
        weights.

    """

    n = H.shape[0]
    assert No < n
    if No < 0:
        No = n-1
    if method == 0:
        r, w = cmp_Grw(H, No)
    else:
        pcf = apc_cfs(H, No)
        p = np.poly1d(np.flip(pcf, 0))
        r = np.real(p.r)
        w = gen_Gw(H[0, :], r)
    return r, w


def gen_npc_mx(cf, r, w, No=-1):
    """
    Generates normalized polynomial coefficients using quadrature

    Parameters
    ----------
    cf : np.array
        polynomial coefficients.
    r : np.array
        Quadrature roots.
    w : np.array
        Quadrature weights.
    No : int, optional
        max polynomial degree. The default is -1.

    Returns
    -------
    ncf : np.array
        normalized polynomial coefficients.

    """

    n = cf.shape[0]
    assert No <= n
    if No < 0:
        No = n
    ncf = np.zeros([No, No])
    for k in range(No):
        nc = cmp_norm_cf(cf[k, :], r, w)
        if nc > 0:
            ncf[k, :] = cf[k, :]/nc
        else:
            ncf[k, :] = 0
    return ncf


def gen_npc_mx_mm(cf, H_mx, No=-1):
    """
    Generates normalized polynomial coefficients from Hankel matrix
    using aPC

    Parameters
    ----------
    cf : np.array
        polynomial coefficients.
    H_mx : np.array
        Hankel matrix.
    No : int, optional
        polynomial degree. The default is -1.

    Returns
    -------
    ncf : np.array
        normalized polynomial coefficients.

    """

    n = cf.shape[0]
    assert No <= n
    if No < 0:
        No = n
    ncf = np.zeros([No, No])
    for k in range(No):
        nc = cmp_norm_cf_moments(cf[k, :], H_mx)
        if nc > 0:
            ncf[k, :] = cf[k, :]/nc
        else:
            ncf[k, :] = 0
    return ncf


# @njit(float64[:](float64[:], float64[:]), nogil=True)
@njit(nogil=True, cache=True)
def pc_eval(cfs, X):
    """
    Applies polyval with polyonomial p defined by  Cfs on X [p(X)]

    Parameters
    ----------
    cfs : np.array
        polynomial coefficients.
    X : np.array
        x-values to evaluate the polynomial on

    Returns
    -------
    np.array
        p(X).

    """
    c = cfs.T
    # return P.polyval(X, c, tensor=False)
    # copied from the numpy code to make for numba available
    # if isinstance(X, (tuple, list)):
    #     X = np.asarray(X)
    c0 = c[-1] + X*0
    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0*X
    return c0
