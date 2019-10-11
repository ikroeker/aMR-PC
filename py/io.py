import pandas as pd
import numpy as np
import pickle

def writeEvalPoints(Points,fname,**kwargs):
    """ 
    writes evals points in asci file
    additionals args: dir and template
    """
    if 'dir' in kwargs.keys():
        dir=kwargs['dir']
    else:
        dir="../data"
    if 'tmplt' in kwargs.keys():
        template=kwargs['tmplt']
    else:
        template=' {: 8.7e}  {: 8.7e}  {: 8.7e}  {: 8.7e}\n'
    file=dir+"/" + fname
    lines,cols=Points.shape
    f=open(file,'w')
    for l in range(lines):
        dataLine=Points[l]
        #print(dataLine)
        s=template.format(*dataLine)
        f.write(s)
    f.close()

def storeDataDict(Dict,fname,dir="../data"):
    """ stores dictionary in {dir}/{fname}.p using pickle """
    file=dir +"/" + fname + '.p'
    f=open(file,"wb")
    pickle.dump(Dict,f)
    f.close()

def loadDataDict(fname,dir="../data"):
    """ load picle stored data from {dir}/{fname}.p """
    file=dir +"/" + fname + '.p'
    f=open(file,"rb")
    Dict=pickle.load(f)
    f.close()
    return Dict

    
if  __name__=="__main__":
  rpts=np.random.randn(100,4)
  print(rpts.shape)
  writeEvalPoints(rpts,'pts.txt')
