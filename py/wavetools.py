import numpy as np
import math
#import polytools as pt

class wavetools:

    def __init__(self,deg,lb=0,rb=1):
        self.P=deg
        self.n=2*self.P+1
        self.lb=lb
        self.rb=rb
        # roots and weights on [-1,1]
        roots, weights=np.polynomial.legendre.leggauss(self.n)
        # transform roots and weights to [lb, rb]
        self.half=(self.rb-self.lb)/2
        self.roots=self.half*roots
        self.weights=self.half*weights
        self.p=np.zeros([self.P,self.n])
        self.qt=np.zeros([self.P,self.n])
        self.q=np.zeros([self.P,self.n])
        self.r=np.zeros([self.P,self.n])
        self.psi=np.zeros([self.P,self.n])
        self.alpha=np.zeros([self.P,self.P])
        self.beta=np.zeros([self.P,self.P])
        s=np.sign(self.roots-self.half)        
        for i in range(self.P):
            self.p[i,:]=self.roots**i
            self.qt[i,:]=s*self.p[i,:]
            
    def stepOne(self):
        """ Step 1, according to Le Maitre at al """
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
        for j in range(self.P-1,-1,-1):
            for l in range(j+1,self.P):
                den=((self.r[l,:]**2) @ self.weights)                
                self.beta[j,l]=-((self.q[j,:]*self.r[l,:]) @ self.weights)/den
            self.r[j,:]=self.q[j,:]+self.beta[j,j+1:self.P] @ self.r[j+1:self.P,:]
                
    def stepThree(self):
        """ Step 3, normalization """
        psncf=np.ones(self.P)
        for j in range(self.P):
            psncf[j]=math.sqrt((self.r[j,:]**2) @ self.weights)
            print(psncf[j])
            self.psi[j,:]=self.r[j,:]/psncf[j]
                                  
if __name__=="__main__":
    wv=wavetools(3)
    wv.stepOne()
    wv.stepTwo()
    wv.stepThree()
    print("done!")
