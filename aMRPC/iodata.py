import os.path as op
import numpy as np
import pandas as pd
import pickle

dataDir = "../data" # data directory
inDir = dataDir # directory for input data
outDir = dataDir # directory for output data
def writeEvalPoints(Points, fname, **kwargs):
    """
    writes evals points in asci file
    additionals args: dir and template
    """
    if 'dir' in kwargs.keys():
        mydir = kwargs['dir']
    else:
        mydir = outDir
    if 'tmplt' in kwargs.keys():
        template=kwargs['tmplt']
    else:
        template = ' {: 8.7e}  {: 8.7e}  {: 8.7e}  {: 8.7e}\n'
    file = mydir + "/" + fname
    lines, _ = Points.shape
    f = open(file, 'w')
    for l in range(lines):
        dataLine = Points[l]
        #print(dataLine)
        s = template.format(*dataLine)
        f.write(s)
    f.close()

def loadEvalPoints(fname, mydir=None):
    """
    loads eval points from an ascii file
    """
    if mydir is None:
        mydir = inDir
    url = mydir + "/" + fname
    if op.exists(url):
        dataframe=pd.read_csv(url, header=None, sep='\s+ ', engine='python')
        #dataframe=np.loadtxt(url)
    else:
        print(url," does not exist!")
    return dataframe

def storeDataDict(Dict, fname, mydir=None):
    """ stores dictionary in {dir}/{fname}.p using pickle """
    if mydir is None:
        mydir = outDir
    file = mydir +"/" + fname
    try:
        f = open(file, "wb")
        pickle.dump(Dict, f)
        f.close()
    except (IOError, ValueError):
        print("An I/O error or a ValueError occurred")
    except:
        print("An unexpected error occurred")
        raise

def loadDataDict(fname, mydir=None):
    """ load picle stored data from {dir}/{fname}.p """
    if mydir is None:
        mydir = outDir
    file = mydir + "/" + fname
    f = open(file, "rb")
    try:
        Dict = pickle.load(f)
        f.close()
    except (IOError, ValueError):
        print("An I/O error or a ValueError occurred")
    except:
        print("An unexpected error occurred")
        raise
    return Dict

def storeNPArr(npArray, fname, mydir=None):
    """ stores numpy.array npArray in {dir}/{fname} """
    if mydir is None:
        mydir = outDir
    file = mydir +"/" + fname
    try:
        np.save(file, npArray)
    except (IOError, ValueError):
        print("An I/O error or a ValueError occurred")
    except:
        print("An unexpected error occurred")
        raise

def loadNPArr(fname, mydir=None):
    """ loads numpy.array from {mydir}/{fname} """
    if mydir is None:
        mydir = outDir
    file = mydir +"/" + fname
    try:
        npArray = np.load(file)
    except (IOError, ValueError):
        print("An I/O error or a ValueError occurred")
    except:
        print("An unexpected error occurred")
        raise
    return npArray
FilePfx={
    1: "roots",
    2: "weights",
    3: "Hdict", # Hankel Matrices dictionary
    4: "rootDict",
    5: "weightsDict",
    6: "PCdict", # (monic) orthogonal polyonimial coeff. dict.
    7: "nPCdict",# orthonormal pc. dict
    8: "NRBdict", # dictionary of MR-elements bounds
    9: "mKeys", # multi-keys in order of evaluation points
    10: "sid2mk", # sample id / eval point nr -> multi-key
    11: "mk2sid", # multi-key -> sample id
    12: "rCfs", # rescalling coefs multi-key->cf
    13: "polOnSid" # polyonimals evaluated on roots / samples (p,sid)
}
FileSfx={
    1: ".txt",
    2: ".txt",
    3: ".p",
    4: ".p",
    5: ".p",
    6: ".p",
    7: ".p",
    8: ".p",
    9: ".p",
    10: ".p",
    11: ".p",
    12: ".p",
    13: ".npy"
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
