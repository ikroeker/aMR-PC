import numpy as np
import math

def moment(mm, data):
    """computes mm-th raw moment"""
    return np.mean(data**mm)
    
def Hankel(mmx,data):
    """generates Hankel matrix"""
    H=np.zeros((mmx+1,mmx+1))
    for i in range(mmx+1):
        for j in range(i,mmx+1):
            H[i,j]=moment(i+j,data)
            H[j,i]=H[i,j]
    return H

def aPCcfs(H,k=-1,len=-1):
    """polynomial coefficients in increasing order, Sergey style"""
    l=H.shape[0]
    assert l>=len
    if len==-1:
        len=l
    cfs=np.zeros(len)
    if k<0:
        k=l-1
    elif k==0:
        cfs[0]=1
        return cfs
    assert k<l
    rH=H[0:k+1,0:k+1]
    rs=np.zeros(k+1)
    rs[-1]=1
    for j in range(k+1):
        rH[k,j]=0
    rH[k,k]=1
    cfs=np.linalg.solve(rH,rs)
    cfs.resize(len)
    return cfs
 
def PCcfs(H,k=-1,len=-1):
    """polynomial coefficientes in increasing order, Gautschi style via moment determinants (p. 53)"""
    l=H.shape[0]
    assert len<=l
    if len==-1:
        len=l
    cfs=np.zeros(len)
    if k<0:
        k=l-1
    elif k==0:
        cfs[0]=1
        return cfs
    assert k<l
    rH=H[0:k,0:k]
    idx=np.ones(l,dtype=bool)
    idx[k+1:l]=False
    delta=np.linalg.det(rH)
    for i in range(0,k):
        idx[i]=False
        Hk=H[0:k,idx]
        dHk=np.linalg.det(Hk)
        cfs[i]=(-1)**(k+i) * dHk/delta
        idx[i]=True
    cfs[k]=1
    return cfs

def cmpAlphaBeta(H,k=-1):
    """generates vectors of recursion coefficientes alpha and beta"""
    n=H.shape[0]
    if k==-1:
        k=n-1
    assert k<n  
    alpha=np.zeros(k)
    beta=np.zeros(k)
    delta=np.zeros(k+1)
    deltak=np.zeros(k+1)
    idx=np.zeros(n,dtype=bool)
    for l in range(1,k+1):
        #print(l)
        delta[l]=np.linalg.det(H[0:l,0:l])
        if l>1:
            idx[0:l-1]=True
            idx[l]=True
            deltak[l]=np.linalg.det(H[0:l,idx])
            idx[0:n]=False
    delta[0]=1
    deltak[0]=0
    deltak[1]=H[0,1]
    for l in range(k):
        if l>0:
            alpha[l]=deltak[l+1]/delta[l+1] - deltak[l]/delta[l]
            beta[l]=delta[l+1]*delta[l-1]/(delta[l]**2)
        else:
            alpha[l]=deltak[l+1]/delta[l+1]
            beta[l]=H[0,0]
    return alpha, beta

def JacobiMx(alpha,beta):
    """generates Jacobi Matrix"""
    n=alpha.shape[0]
    m=beta.shape[0]
    assert m==n
    J=np.diag(alpha)
    for l in range(n-1):
        J[l,l+1]=math.sqrt(beta[l+1])
        J[l+1,l]=J[l,l+1]
    return J

def cmpGrw(H,k=-1):
    """computes roots and weigth of the Gauss quadrature using Hankel Matrix, Gautschi p. 153"""
    n=H.shape[0]
    assert k<n
    alpha,beta=cmpAlphaBeta(H,k)
    J=JacobiMx(alpha,beta)
    print(J)
    tau, V=np.linalg.eig(J)
    roots=tau
    weights=beta[0]* (V[0,:]**2)
    #v=np.zeros(roots.shape[0])
    #for i in range(roots.shape[0]):
    #    v[i]=V[0,i]**2
    #weights=beta[0]*v  
    return roots,weights

def genGW(moments,roots):
    """computes Gaussian weights using moments and roots, compare with Karniadakis & Kirby p.236"""
    m=moments.shape[0]
    r=roots.shape[0]
    assert m>=r
    rs=moments[0:r]
    M=np.zeros([r,r])
    for i in range(r):
        for j in range(r):
            M[i,j]=roots[j]**i
    return np.linalg.solve(M,rs)
    
def cmpNormCf(cf,roots,weights):
    """computes the norming factor of the polynomial w.r.t. Gauss quadrature"""
    r=roots.shape[0]
    w=weights.shape[0]
    assert r==w
    c=np.flip(cfs,0)
    p=np.poly1d(c)
    nc=0;
    for i in range(r):
        nc+=(p(roots[i])**2)*weights[i]
    return math.sqrt(nc)
