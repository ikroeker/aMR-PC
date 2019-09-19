# load pandas
import pandas as pd
import polytools as pt
import numpy as np
import matplotlib.pyplot as plt

# data location
url='../data/InputParameters.txt'

# load data
dataframe=pd.read_csv(url,header=None,sep='\s+ ',engine='python')

# dataframe.head(2)
# dataframe[0].head(2)

n=dataframe.shape[0]
n

src=0
data=np.array(dataframe[src])
#H
mm=5
#out=moment(mm,v)
#out
k=-1
H=pt.Hankel(mm,data)   
#print("H=",H[0:6,0:6])
#print(np.linalg.det(H))
acfs=pt.aPCcfs(H,k)
print(acfs)
cfs=pt.PCcfs(H,k)
print(cfs)
#np.linalg.cond(H)
alpha, beta=pt.cmpAlphaBeta(H)
print("alpha:", alpha)
print("beta:",beta)
#J=JacobiMx(alpha,beta)
roots,weights=pt.cmpGrw(H)
print("Roots:", roots)
print("Weights:", weights)

c=np.flip(cfs,0)
p=np.poly1d(c)
rts=p.r
mms=H[0,:]
ws=pt.genGW(mms,rts)
print("p Roots:",rts)
print("p Weights:",ws)
eins=pt.np.ones(roots.shape[0])
np.dot(eins,weights)
np.dot(eins,ws)
ncf=pt.cmpNormCf(cfs,roots,weights)
#pt.uniHank(mm,0,1)
#pt.uniHank.__doc__
x=np.linspace(0,1,100)
plt.plot(x,p(x)/ncf)
plt.show()
