import numpy as np
import math
#import polytools as pt

class wavetools:
    """ generates wavelet functions according to Le Maitre et al """
    def __init__(self,deg,qdeg=-1,decOn=True,lb=0,rb=1):
        self.P=deg # polynomial degree
        if qdeg>0:
            self.n=qdeg # quadrature degreee / number of quad points
        else:
            self.n=2*(self.P+1) # number of roots should be even
        self.decOn=decOn
        self.lb=lb
        self.rb=rb
        self.len=rb-lb
        self.half=lb+self.len/2
        #self.fs=lambda x: -1*(x<self.half)+ 1*(x>=self.half)
        self.fs=lambda x: np.sign(x-self.half)
        self.vfs=np.vectorize(self.fs)

    def initQuad(self,):
        """ 
        initialises quadrature, qdeg - quadrature degree should be even
        uses two (roots,weights) tuples for lhs and rhs if decOn==True
        """
        if self.decOn:
            # roots and weights on [-1,1]
            roots, weights=np.polynomial.legendre.leggauss(self.n//2)
            # transform roots and weights to [lb, rb]
            r=self.half*(roots+1)
            w=self.half*weights            
            lr=self.rescY(r,1,0)
            rr=self.rescY(r,1,1)
            self.roots=np.concatenate((lr,rr))
            wh=w/2
            self.weights=np.concatenate((wh,wh))
        else:
            roots, weights=np.polynomial.legendre.leggauss(self.n)
            self.roots=self.half*(roots+1)
            self.weights=self.half*weights
        #print("r",r,lr,rr)
        #print(self.roots,self.weights)
        
    def initCfs(self):
        """ initializes coefficient arrays """
        self.p=np.zeros([self.P,self.n])
        self.qt=np.zeros([self.P,self.n])
        self.q=np.zeros([self.P,self.n])
        self.r=np.zeros([self.P,self.n])
        self.psi=np.zeros([self.P,self.n])
        self.alpha=np.zeros([self.P,self.P])
        self.beta=np.zeros([self.P,self.P])
        s=self.fs(self.roots)
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
            #print(v,self.alpha)
            self.q[j,:]=self.qt[j,:]+ self.alpha[:,j].T @ self.p  

    def stepOneMS(self):
        """ Step 1, inspired by Markus Schmidgall, p. 24, if a=0, b=1 """
        rH=np.zeros([self.P,self.P])
        v=np.zeros(self.P)
        for i in range(self.P):
            for j in range(self.P):
                rH[i,j]=1/(i+j+1)
        for j in range(self.P):
            for i in range(self.P):
                v[i]= (2**(-(i+j)) -1)/(i+j+1)
            self.alpha[:,j]=np.linalg.solve(rH,v)
            self.q[j,:]=self.qt[j,:]+self.alpha[:,j].T @ self.p
                    
        
    def stepTwo(self):
        """ Step 2, acc. Le Maitre """
        self.r[self.P-1,:]=self.q[self.P-1,:]
        for j in range(self.P-2,-1,-1):
            for l in range(j+1,self.P):
                #den=((self.r[l,:]**2) @ self.weights)
                #print("2.",self.r[l,:])
                self.beta[j,l]=-((self.q[j,:]*self.r[l,:]) @ self.weights)/((self.r[l,:]**2) @ self.weights)
            self.r[j,:]=self.q[j,:]+self.beta[j,j+1:self.P] @ self.r[j+1:self.P,:]
            #print(self.r)
                
    def stepThree(self):
        """ Step 3, normalization """
        self.psncf=np.ones(self.P)
        for j in range(self.P):
            self.psncf[j]=math.sqrt((self.r[j,:]**2) @ self.weights)
            #self.psncf[j]=(self.r[j,:]**2) @ self.weights
            self.psi[j,:]=self.r[j,:]/self.psncf[j]
        
    def genWVlets(self,MS=False):
        """ 
        performs the three steps to generate the Multi-Wavelet basis 
        qdeg (quadrature degree) should be even, if seted
        MS - set MS style step one
        decOn - switch the decoupling in two quadratures on
        """
        self.initQuad()
        self.initCfs()
        bchk= self.lb==0 and self.rb==1
        if MS and bchk:
            self.stepOneMS()
        else:
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
        lbi, rbi, scf=self.cmpLRBi(Nr,Nri)
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
    def cmpLRBi(self,Nr,Nri):
         scf=self.len / 2**Nr
         lbi=self.lb+ Nri * scf
         rbi=self.lb+(Nri+1) *scf
         return lbi, rbi, scf
     
    def rescX(self,x,Nr,Nri):
        """ transforms x\in[lb_i,rb_i] to y in [lb,rb] """
        lbi, __, scf=self.cmpLRBi(Nr,Nri)
        y=self.lb+(x-lbi)/scf
        return y
    
    def rescY(self,y,Nr,Nri):
        """ transforms y in [lb,rb] to x in [lbi,rbi] """
        lbi, __, scf=self.cmpLRBi(Nr,Nri)
        x=lbi+(y-self.lb)*scf
        return x
    
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

    def cmpDetails(self,dataOnRoots,No=-1):
        """ computes MW coefficients for data on roots """
        assert(No<=self.P)
        if No>=0:
            return (dataOnRoots * self.fpsi(No,self.roots)) @ self.weights
        else:
            ret=np.zeros(self.P)
            for i in range(self.P):
                ret[i]= (dataOnRoots * self.fpsi(i,self.roots)) @ self.weights
        return ret
    
if __name__=="__main__":
    p=1
    print("p =",p)
    wv=wavetools(p)
    wv.genWVlets()
    i=0
    x=.7
    print("i =",i,"  x =",x)
    print("fr(%d,%f)=%f"%(i,x,wv.fr(i,x)))
    print("fpsi(%d,%f)=%f"%(i,x,wv.fpsi(i,x)))
    print("rfpsi(%f,%d,1,1)=%f"%(x,i,wv.rfpsi(x,i,1,1)))

    print("well done!")
