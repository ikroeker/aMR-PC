import numpy as np
import math
#import polytools as pt

class wavetools:
    """ generates wavelet functions according to Le Maitre et al """
    def __init__(self,deg,lb=0,rb=1):
        self.P=deg
        self.n=2*self.P+1
        self.lb=lb
        self.rb=rb
        self.len=rb-lb
        # roots and weights on [-1,1]
        roots, weights=np.polynomial.legendre.leggauss(self.n)
        # transform roots and weights to [lb, rb]
        self.half=self.len/2
        self.roots=self.half*(roots+1)
        self.weights=self.half*weights
        self.p=np.zeros([self.P,self.n])
        self.qt=np.zeros([self.P,self.n])
        self.q=np.zeros([self.P,self.n])
        self.r=np.zeros([self.P,self.n])
        self.psi=np.zeros([self.P,self.n])
        self.alpha=np.zeros([self.P,self.P])
        self.beta=np.zeros([self.P,self.P])
        #b=self.roots<self.half
        #s=np.ones(self.n)
        #s[b]=-1*s[b]        
        self.fs=lambda x: -1*(x<self.half)+ 1*(x>=self.half)
        self.vfs=np.vectorize(self.fs)
        s=self.fs(self.roots)
        #s=np.sign(self.roots-self.half)        
        for i in range(self.P):
            self.p[i,:]=self.roots**i
            self.qt[i,:]=s*self.p[i,:]
            
    def stepOne(self):
        """ Step 1, according to Le Maitre et al """
        rH=np.zeros([self.P,self.P])
        for i in range(self.P):
            for j in range(self.P):
                rH[i,j]=(self.p[i,:]*self.p[j,:])@ self.weights
        for j in range(self.P):
            v=-(self.p * self.qt[j,:])@ self.weights
            self.alpha[:,j]=np.linalg.solve(rH,v)
            self.q[j,:]=self.qt[j,:]+ self.alpha[:,j].T @ self.p  

    def stepTwo(self):
        """ Step 2, acc. Le Maitre """
        self.r[self.P-1,:]=self.q[self.P-1,:]
        for j in range(self.P-2,-1,-1):
            for l in range(j+1,self.P):
                #den=((self.r[l,:]**2) @ self.weights)
                #print("2.",self.r[l,:])
                self.beta[j,l]=-((self.q[j,:]*self.r[l,:]) @ self.weights)/((self.r[l,:]**2) @ self.weights)
            self.r[j,:]=self.q[j,:]+self.beta[j,j+1:self.P] @ self.r[j+1:self.P,:]
                
    def stepThree(self):
        """ Step 3, normalization """
        self.psncf=np.ones(self.P)
        for j in range(self.P):
            self.psncf[j]=math.sqrt((self.r[j,:]**2) @ self.weights)
            self.psi[j,:]=self.r[j,:]/self.psncf[j]

    def genWVlets(self):
        """ performs the three steps to generate the Multi-Wavelet basis """
        self.stepOne()
        self.stepTwo()
        self.stepThree()
        
    def sfr(self,i,x):
        """ r_i(x) evaluated on an arbitrariy point x (scalar version) """
        pws=np.arange(self.P)
        ps=x**pws
        qs=self.fs(x)*ps+self.alpha.T @ ps
        rs=np.zeros(self.P)
        rs[self.P-1]=qs[self.P-1]
        for j in range(self.P-2,i-1,-1):
            rs[j]=qs[j]+self.beta[j,j+1:self.P] @ rs[j+1:self.P]
        return rs[i]
    
    def fr(self,i,x):
        """ r_i(x) evaluated on an arbitrariy point x (scalar and vector) """
        if type(x) is float or type(x) is int:
            return self.sfr(i,x)
        else:
            xl=len(x)
            y=np.zeros(xl)
            for j in range(xl):
                y[j]=self.sfr(i,x[j])
            return y

    def fpsi(self,i,x):
        """ function psi_j(x) evaluated on an arbitraty point x """
        return self.fr(i,x)/self.psncf[i]

    def bdChk(self,x,Nr,Nri):
        """ Boundary check, returns true if x in [lb_i,rb_i] """
        scf=self.len * 2**(-Nr)
        lbi=self.lb+ Nri * scf
        rbi=self.lb+(Nri+1) *scf
        if type(x)==float or type(x)==int:
            if x >= lbi and x<=rbi:
                return True
            else:
                return False
        else:
            xl=len(x)
            b=np.zeros(xl,dtype=bool)
            b[x>=lbi]=True
            b[x>rbi]=False
            #print(x[b])
            return b
        
    def rescX(self,x,Nr,Nri):
        """ transforms x\in[lb_i,rb_i] to y in [lb,rb] """
        scf=self.len * 2**(-Nr)
        lbi=self.lb+ Nri * scf
        rbi=self.lb+(Nri+1) *scf
        y=self.lb+(x-lbi)/scf
        return y

    def rescCf(self,Nr):
        """ resc. coefficients for multi-wavelets """
        return 2**(Nr/2)

    def rfpsi(self,x,i,Nr,Nri):
        """ rescaled multi-wavelets """
        b=self.bdChk(x,Nr,Nri)
        if type(x)==int or type(x)==float:
            if b:
                rx=self.rescX(x,Nr,Nri)
                return self.rescCf(Nr) * self.fpsi(i,rx)
            else:
                return 0
        else:
            vx=self.rescX(x,Nr,Nri)
            vy=np.zeros(len(x))
            vy[b]=self.rescCf(Nr)*self.fpsi(i,vx[b])
            return vy

    def cmpDetails(self,dataOnRoots,delta,No=-1):
        """ computes MW coefficients for data on roots """
        assert(No<=self.P)
        if No>=0:
            return dataOnRoots @ self.fpsi(No,self.roots)
        else:
            ret=np.zeros(self.P)
            for i in range(self.P):
                ret[i]= dataOnRoots @ self.fpsi(i,self.roots)
            return ret
    
if __name__=="__main__":
    wv=wavetools(3)
    #wv.stepOne()
    #print("a",wv.alpha)
    #print("p",wv.p)
    #print("q",wv.q)
    #wv.stepTwo()
    #wv.stepThree()
    wv.genWVlets()
    i=0
    x=.7
    print(wv.fr(i,x))
    print(wv.fpsi(i,x))
    print(wv.rfpsi(x,1,1,1))
    print("well done!")
