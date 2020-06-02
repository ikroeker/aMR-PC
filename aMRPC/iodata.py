"""
iodata.py - povides io-routines and file-name conventions
@author: kroeker
"""

import os.path as op
import pickle
import numpy as np
import pandas as pd


dataDir = "../data" # data directory
inDir = dataDir # directory for input data
outDir = dataDir # directory for output data
def write_eval_points(points, fname, **kwargs):
    """
    writes evals points in asci file
    additionals args: dir and template
    """
    if 'dir' in kwargs.keys():
        mydir = kwargs['dir']
    else:
        mydir = outDir
    if 'tmplt' in kwargs.keys():
        template = kwargs['tmplt']
    else:
        template = ' {: 8.7e}  {: 8.7e}  {: 8.7e}  {: 8.7e}\n'
    file = mydir + "/" + fname
    lines, _ = points.shape
    f_h = open(file, 'w')
    for l_idx in range(lines):
        data_line = points[l_idx]
        #print(dataLine)
        str_line = template.format(*data_line)
        f_h.write(str_line)
    f_h.close()

def gen_file_url(fname, mydir=None):
    """
    generates file url for load routines

    Parameters
    ----------
    fname : string
        file name.
    mydir : string, optional
        directory. The default is None.

    Returns
    -------
    string
        file url.

    """
    if mydir is None:
        mydir = inDir
    return mydir + "/" + fname

def chk_file_url(fname, mydir=None):
    """
    checks if file fname exists

    Parameters
    ----------
    fname : string
        file name.
    mydir : string, optional
        directory name. The default is None.

    Returns
    -------
    bool
        true or false depending of existence.

    """
    url = gen_file_url(fname, mydir)
    return op.exists(url)

def load_eval_points(fname, mydir=None):
    """
    loads eval points from an ascii file
    """
    url = gen_file_url(fname, mydir)
    if op.exists(url):
        dataframe = pd.read_csv(url, header=None, sep='\s+ ', engine='python')
        #dataframe=np.loadtxt(url)
    else:
        print(url, " does not exist!")
    return dataframe

def store_data_dict(out_dict, fname, mydir=None):
    """ stores dictionary in {dir}/{fname}.p using pickle """
    if mydir is None:
        mydir = outDir
    file = gen_file_url(fname, mydir)
    try:
        f_h = open(file, "wb")
        pickle.dump(out_dict, f_h)
        f_h.close()
    except (IOError, ValueError):
        print("An I/O error or a ValueError occurred")
    except:
        print("An unexpected error occurred")
        raise

def load_data_dict(fname, mydir=None):
    """ load picle stored data from {dir}/{fname}.p """
    if mydir is None:
        mydir = outDir
    file = gen_file_url(fname, mydir)
    f_h = open(file, "rb")
    try:
        in_dict = pickle.load(f_h)
        f_h.close()
    except (IOError, ValueError):
        print("An I/O error or a ValueError occurred")
    except:
        print("An unexpected error occurred")
        raise
    return in_dict

def store_np_arr(np_array, fname, mydir=None):
    """ stores numpy.array npArray in {dir}/{fname} """
    if mydir is None:
        mydir = outDir
    file = gen_file_url(fname, mydir)
    try:
        np.save(file, np_array)
    except (IOError, ValueError):
        print("An I/O error or a ValueError occurred")
    except:
        print("An unexpected error occurred")
        raise

def load_np_arr(fname, mydir=None):
    """ loads numpy.array from {mydir}/{fname} """
    if mydir is None:
        mydir = outDir
    file = gen_file_url(fname, mydir)
    try:
        np_array = np.load(file)
    except (IOError, ValueError):
        print("An I/O error or a ValueError occurred")
    except:
        print("An unexpected error occurred")
        raise
    return np_array
FilePfx = {
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
FileSfx = {
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
def gen_fname(fkt, **kwargs):
    """ generates filename"""
    pfx_chk = fkt in FilePfx.keys()
    sfx_chk = fkt in FileSfx.keys()
    assert pfx_chk and sfx_chk
    fname = FilePfx.get(fkt)

    if 'txt' in kwargs.keys():
        fname += kwargs['txt']
    if 'Nr' in kwargs.keys():
        fname += '_Nr'+kwargs['Nr']
    if 'No' in kwargs.keys():
        fname += '_No' +kwargs['No']
    if 'ths' in kwargs.keys():
        fname += '_ths'+kwargs['ths']

    fname += FileSfx.get(fkt) # add sfx to the filename
    return fname

def main():
    """ main function for testing """
    rpts = np.random.randn(100, 4)
    print(rpts.shape)
    write_eval_points(rpts, 'pts.txt')
    data = load_eval_points('pts.txt')
    print(data.describe())

if  __name__ == "__main__":
    main()
