"""
wavetools.py - class WaveTolls provides construction of multi-wavelets on (0,1)
@author: kroeker
"""
import math
import numpy as np

#import pandas as pd

class WaveTools:
    """ generates wavelet functions according to
        Le Maître OP, Najm HN, Ghanem RG, Knio OM. Multi-resolution analysis
        of Wiener-type uncertainty propagation schemes. J Comput Phys
        2004;197(2):502–31.
    """
    def __init__(self, deg, qdeg=-1, decOn=True, lb=0, rb=1):
        self.P = deg # polynomial degree
        if qdeg > 0:
            self.n = qdeg # quadrature degreee / number of quad points
        else:
            self.n = 2*(self.P+1) # number of roots should be even
        self.decOn = decOn
        self.lb = lb
        self.rb = rb
        self.len = self.rb - self.lb
        self.sqlen = math.sqrt(self.len)
        self.half = 0.5
        #self.fs=lambda x: -1*(x<self.half)+ 1*(x>=self.half)
        self.fs = lambda x: np.sign(x-self.half)
        self.vfs = np.vectorize(self.fs)
        self.roots = np.zeros(self.n)
        self.weights = np.zeros(self.n)
        # initializes coefficient arrays
        self.p = np.zeros([self.P, self.n])
        self.qt = np.zeros([self.P, self.n])
        self.q = np.zeros([self.P, self.n])
        self.r = np.zeros([self.P, self.n])
        self.psi = np.zeros([self.P, self.n])
        self.alpha = np.zeros([self.P, self.P])
        self.beta = np.zeros([self.P, self.P])

        self.psncf = np.ones(self.P)    # normalization coeff.

    def init_quad(self,):
        """
        initialises quadrature, qdeg - quadrature degree should be even
        uses two (roots,weights) tuples for lhs and rhs if decOn==True
        """
        if self.decOn:
            # roots and weights on [-1,1]
            tmp_roots, tmp_weights = np.polynomial.legendre.leggauss(self.n//2)
            # transform roots and weights to [0, 1]
            #r=self.half*(roots+1)
            w = self.half*tmp_weights
            #lr=self.rescY(r,1,0)
            #rr=self.rescY(r,1,1)
            lr = (tmp_roots + 1)/4
            rr = (tmp_roots+1)/4 + self.half
            self.roots = np.concatenate((lr, rr))
            wh = w/2
            self.weights = np.concatenate((wh, wh))
        else:
            tmp_roots, tmp_weights = np.polynomial.legendre.leggauss(self. n)
            self.roots = self.half*(tmp_roots+1)
            self.weights = self.half*tmp_weights
        s = self.fs(self.roots)
        for i in range(self.P):
            self.p[i, :] = self.roots**i
            self.qt[i, :] = s*self.p[i, :]
        #print("r",r,lr,rr)
        #print(self.roots,self.weights)



    def step_one(self):
        """ Step 1, according to Le Maitre et al """
        rH = np.zeros([self.P, self.P])
        for i in range(self.P):
            for j in range(self.P):
                rH[i, j] = (self.p[i, :]*self.p[j, :]) @ self.weights
        for j in range(self.P):
            v = -(self.p * self.qt[j, :]) @ self.weights
            self.alpha[:, j] = np.linalg.solve(rH, v)
            #print(v,self.alpha)
            self.q[j, :] = self.qt[j, :] + self.alpha[:, j].T @ self.p

    def step_one_ms(self):
        """ Step 1, inspired by Markus Schmidgall, p. 24, if a=0, b=1 """
        rH = np.zeros([self.P, self.P])
        v = np.zeros(self.P)
        for i in range(self.P):
            for j in range(self.P):
                rH[i, j] = 1/(i+j+1)
        for j in range(self.P):
            for i in range(self.P):
                v[i] = (2**(-(i+j)) -1)/(i+j+1)
            self.alpha[:, j] = np.linalg.solve(rH, v)
            self.q[j, :] = self.qt[j, :] + self.alpha[:, j].T @ self.p


    def step_two(self):
        """ Step 2, acc. Le Maitre """
        self.r[self.P-1, :] = self.q[self.P-1, :]
        for j in range(self.P-2, -1, -1):
            for l in range(j+1, self.P):
                #den=((self.r[l,:]**2) @ self.weights)
                #print("2.",self.r[l,:])
                self.beta[j, l] = -((self.q[j, :]*self.r[l, :])
                                    @ self.weights)/((self.r[l, :]**2)
                                                     @ self.weights)
            self.r[j, :] = self.q[j, :] + self.beta[j, j+1:self.P] @ self.r[j+1:self.P, :]
            #print(self.r)

    def step_three(self):
        """ Step 3, normalization """

        for j in range(self.P):
            self.psncf[j] = math.sqrt((self.r[j, :]**2) @ self.weights)
            #self.psncf[j]=(self.r[j,:]**2) @ self.weights
            self.psi[j, :] = self.r[j, :]/self.psncf[j]

    def genWVlets(self, ms_type=False):
        """
        performs the three steps to generate the Multi-Wavelet basis
        qdeg (quadrature degree) should be even, if seted
        MS - set MS style step one
        decOn - switch the decoupling in two quadratures on
        """
        self.init_quad()
        if ms_type:
            self.step_one_ms()
        else:
            self.step_one()

        self.step_two()
        self.step_three()

    def sfr(self, i, x):
        """ r_i(x) evaluated on an arbitrariy point x (scalar version) """
        pws = np.arange(self.P)
        ps = x**pws
        qs = self.fs(x)*ps + self.alpha.T @ ps
        rs = np.zeros(self.P)
        rs[self.P-1] = qs[self.P-1]
        for j in range(self.P-2, i-1, -1):
            rs[j] = qs[j]+self.beta[j, j+1:self.P] @ rs[j+1:self.P]
        return rs[i]

    def fr(self, i, x):
        """ r_i(x) evaluated on an arbitrariy point x (scalar and vector) """
        if isinstance(x, (float, int)):
            y = self.sfr(i, x)
        else:
            xl = len(x)
            y = np.zeros(xl)
            for j in range(xl):
                y[j] = self.sfr(i, x[j])
        return y

    def fpsi(self, i, x):
        """ function psi_j(x) evaluated on an arbitraty point x """
        return self.fr(i, x)/self.psncf[i]

    def set_new_lrb(self, lb, rb):
        """ replaces standard (0,1)-BD by (lb,rb) """
        self.lb = lb
        self.rb = rb
        self.len = rb-lb
        self.sqlen = math.sqrt(self.len)

    def cmp_lrbi(self, Nr, Nri):
        """ computes interval boundaries for Nr and Nri """
        scf = self.len / 2**Nr
        lbi = self.lb+ Nri * scf
        rbi = self.lb+(Nri+1) *scf
        return lbi, rbi, scf

    def bd_chk(self, x, Nr, Nri):
        """ Boundary check, returns true if x in [lb_i,rb_i] """
        lbi, rbi, _ = self.cmp_lrbi(Nr, Nri)
        if isinstance(x, (float, int)):
            left = x >= lbi
            right = x <= rbi
            bool_ret = left and right
        else:
            xl = len(x)
            bool_ret = np.zeros(xl, dtype=bool)
            bool_ret[x >= lbi] = True
            bool_ret[x > rbi] = False
            #print(x[b])
        return bool_ret

    def resc_x(self, x, Nr, Nri):
        """ transforms x in[lb_i,rb_i] to y in [0,1] """
        lbi, __, scf = self.cmp_lrbi(Nr, Nri)
        #y=self.lb+(x-lbi)/scf
        y = (x-lbi)/scf
        return y

    def resc_y(self, y, Nr, Nri):
        """ transforms y in [0,1] to x in [lbi,rbi] """
        lbi, __, scf = self.cmp_lrbi(Nr, Nri)
        x = lbi+(y)*scf
        return x

    def resc_cf(self, Nr):
        """ resc. coefficients for multi-wavelets """
        return  2**(Nr/2)/self.sqlen
    def rq_cf(self, Nr):
        """ resc. coefficients for quadrature """
        return self.len/(2**Nr)

    def rfpsi(self, x, i, Nr, Nri):
        """ rescaled multi-wavelets """
        in_elem = self.bd_chk(x, Nr, Nri)
        if isinstance(x, (int, float)):
            if in_elem:
                rx = self.resc_x(x, Nr, Nri)
                vy = self.resc_cf(Nr) * self.fpsi(i, rx)
            else:
                vy = 0
        else:
            vx = self.resc_x(x, Nr, Nri)
            vy = np.zeros(len(x))
            vy[in_elem] = self.resc_cf(Nr) * self.fpsi(i, vx[in_elem])
        return vy


    def cmp_details(self, data_on_roots, No=-1):
        """ computes MW coefficients for data on roots """
        assert No <= self.P
        if No >= 0:
            ret = (data_on_roots * self.fpsi(No, self.roots)) @ self.weights
        else:
            ret = np.zeros(self.P)
            for i in range(self.P):
                ret[i] = (data_on_roots * self.fpsi(i, self.roots)) @ self.weights
        return ret

    def cmp_resc_details(self, data_on_roots, Nr, No=-1):
        """
        computes MW coefficients for data on roots rescalled for Nr
        """
        return self.rq_cf(Nr)*self.cmp_details(data_on_roots, No)

    def cmp_data_on_roots(self, data, Nr, Nri):
        """
        computes quantiles of data for the Multi-Element Nri and resolution level NR
        """
        if Nr > 0:
            tmp_roots = self.resc_y(self.roots, Nr, Nri)
        else:
            tmp_roots = np.copy(self.roots)
        quantile_on_roots = data.quantile(tmp_roots)
        return quantile_on_roots

if __name__ == "__main__":
    P = 1
    print("p =", P)
    WV = WaveTools(P)
    WV.genWVlets()
    I = 0
    X = .7
    print("i =", I, "  x =", X)
    print(f"fr({I}, {X})={WV.fr(I, X)}")
    print(f"fpsi({I},{X})={WV.fpsi(I, X)}")
    print(f"rfpsi({X},{I},1,1)={WV.rfpsi(X, I, 1, 1)}")

    print("well done!")
