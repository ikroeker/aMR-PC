"""
polytools.py - generates aPC basis and related Gaussian quadrature

@author:: kroeker
"""
import math
import numpy as np
from numpy.polynomial import polynomial as P

def moment(mm, data):
    """computes mm-th raw moment"""
    #print(mm)
    return np.mean(data**mm)

def Hankel(mmx, data):
    """generates Hankel matrix"""
    H = np.zeros([mmx+1, mmx+1])
    for i in range(mmx+1):
        for j in range(i, mmx+1):
            H[i, j] = moment(i+j, data)
            H[j, i] = H[i, j]
    return H

def apc_cfs(H, k=-1, alen=-1):
    """ polynomial coefficients in increasing degree
    c_0 + c_1*x + c_2*x^2 + ...
    Sergey style """
    l = H.shape[0]
    assert l >= alen
    if alen == -1:
        alen = l
    cfs = np.zeros(alen)
    if k < 0:
        k = l-1
    elif k == 0:
        cfs[0] = 1
        return cfs
    assert k < l
    rH = np.copy(H[0:k+1, 0:k+1])
    rs = np.zeros(k+1)
    rs[-1] = 1
    for j in range(k+1):
        rH[k, j] = 0
    rH[k, k] = 1
    cfs = np.linalg.solve(rH, rs)
    cfs.resize(alen, refcheck=False) # np.ndarray.resize() -> not np.resize(a)

    return cfs

def pc_cfs(H, k=-1, alen=-1):
    """
    polynomial coefficientes in increasing order,
    Gautschi style via moment determinants (p. 53)
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
    """generates vectors of recursion coefficientes alpha and beta"""
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
        #print(l)
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
    """generates Jacobi Matrix"""
    n = alpha.shape[0]
    m = beta.shape[0]
    assert m == n
    J = np.diag(alpha)
    for l in range(n-1):
        J[l, l+1] = math.sqrt(beta[l+1])
        J[l+1, l] = J[l, l+1]
    return J

def cmp_Grw(H, k=-1):
    """computes roots and weigth of the Gauss quadrature using Hankel Matrix, Gautschi p. 153"""
    n = H.shape[0]
    assert k < n
    alpha, beta = cmp_alpha_beta(H, k)
    J = Jacobi_mx(alpha, beta)
    #print(J)
    tau, V = np.linalg.eig(J)
    roots = tau
    weights = beta[0]* (V[0, :]**2)
    #v=np.zeros(roots.shape[0])
    #for i in range(roots.shape[0]):
    #    v[i]=V[0,i]**2
    #weights=beta[0]*v
    return roots, weights

def gen_Gw(moments, roots):
    """computes Gaussian weights using moments and roots, compare with Karniadakis & Kirby p.236"""
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
    """computes the norming factor of the polynomial w.r.t. Gauss quadrature"""
    r = roots.shape[0]
    w = weights.shape[0]
    assert r == w
    c = np.flip(cfs, 0)
    p = np.poly1d(c)
    nc = 0
    for i in range(r):
        nc += (p(roots[i])**2)*weights[i]
    if nc < eps:
        #print(nc)
        nc = eps # ugly workarround, should be improved
    return math.sqrt(nc)

def cmp_norm_cf_moments(cfs, moments, eps=0):
    m = moments.shape[0]
    n = cfs.shape[0]
    assert m == n
    pol_moments = cfs * moments
    ltwo_norm = np.add.reduce(np.multiply.outer(pol_moments, pol_moments),
                              axis=None)
    if ltwo_norm < eps:
        print(ltwo_norm)
        ltwo_norm = eps # ugly workarround
    return math.sqrt(ltwo_norm)

def uniHank(n, a=0, b=1):
    """Generates Hankel Matrix H_n for U(a,b), uses m_n=1/n+1 sum_k=0^n a^k b^(n-k)"""
    H = np.zeros([n, n])
    lva = a*np.ones(2*n+1)
    lvb = b*np.ones(2*n+1)
    for i in range(2*n+1):
        lva[i] = lva[i]**i
        lvb[i] = lvb[i]**(2*n-i)
    for k in range(n):
        for l in range(n):
            m = k+l
            va = lva[0:m+1]
            vb = lvb[2*n-m:]
            H[k, l] = np.dot(va, vb) / (m+1)
    return H


def gen_pc_mx(H, method=0, No=-1):
    """
    generates a matrix with polynomial coefficients up to degree No
    H - Hankel Matrix,
    method: aPC method: 0 - Gautschi - style, 1- Sergey style
    """
    n = H.shape[0]
    assert No <= n
    if No < 0:
        No = n-1
    cf = np.zeros([No, No])
    for k in range(No):
        if method == 0:
            cf[k, :] = pc_cfs(H, k, No)
        else:
            cf[k, :] = apc_cfs(H, k, No)
    return cf

def gen_rw(H, method=0, No=-1):
    """ generates roots and weights """
    n = H.shape[0]
    assert No < n
    if No < 0:
        No = n-1
    if method == 0:
        r, w = cmp_Grw(H, No)
    else:
        pcf = apc_cfs(H, No)
        p = np.poly1d(np.flip(pcf, 0))
        r = p.r
        w = gen_Gw(H[0, :], r)
    return r, w

def gen_npc_mx(cf, r, w, No=-1):
    """ generates normed polynomial coefficients """
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

def pc_eval(cfs, X):
    """ application of polyval with Cfs on X """
    C = cfs.T
    #print(C)
    R = P.polyval(X, C, tensor=False)
    return R
