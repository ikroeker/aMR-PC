import pandas as pd
import numpy as np
import pickle

dataDir="../data" # data directory
inDir=dataDir # directory for input data
outDir=dataDir # directory for output data
def writeEvalPoints(Points,fname,**kwargs):
    """ 
    writes evals points in asci file
    additionals args: dir and template
    """
    if 'dir' in kwargs.keys():
        dir=kwargs['dir']
    else:
        dir=outDir
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

def loadEvalPoints(fname,dir=inDir):
    """
    loads eval points from an ascii file
    """
    url=dir + "/" + fname
    dataframe=pd.read_csv(url,header=None,sep='\s+ ',engine='python')
    return dataframe
        
def storeDataDict(Dict,fname,dir=outDir):
    """ stores dictionary in {dir}/{fname}.p using pickle """
    file=dir +"/" + fname
    try:
        f=open(file,"wb")
        pickle.dump(Dict,f)
        f.close()
    except (IOError, ValueError):
        print("An I/O error or a ValueError occurred")
    except:
        print("An unexpected error occurred")
        raise

def loadDataDict(fname,dir=inDir):
    """ load picle stored data from {dir}/{fname}.p """
    file=dir +"/" + fname
    f=open(file,"rb")
    Dict=pickle.load(f)
    f.close()
    return Dict

FilePfx={
        1: "roots",
        2: "weights",
        3: "Hdict",
        4: "rootDict",
        5: "weightsDict",
        6: "PCdict",
        7: "nPCdict"
        }
FileSfx={
        1: ".txt",
        2: ".txt",
        3: ".p",
        4: ".p",
        5: ".p",
        6: ".p",
        7: ".p"
        }
def genFname(Fkt,**kwargs):
    """ generates filename"""
    PfxChk= Fkt in FilePfx.keys()
    SfxChk= Fkt in FileSfx.keys()
    assert(PfxChk and SfxChk)
    fname=FilePfx.get(Fkt)

    if 'txt' in kwargs.keys():
        fname+=kwargs['txt']
    if 'Nr' in kwargs.keys():
        fname+='_Nr'+kwargs['Nr']
    if 'No' in kwargs.keys():
        fname+='_No' +kwargs['No']
    if 'ths' in kwargs.keys():
        fname+='_ths'+kwargs['ths']

    fname+=FileSfx.get(Fkt) # add sfx to the filename
    return fname
    
def main():
    rpts=np.random.randn(100,4)
    print(rpts.shape)
    writeEvalPoints(rpts,'pts.txt')
    data=loadEvalPoints('pts.txt')
    print(data.describe())

if  __name__=="__main__":
    main()
