"""
Extended unit tests for aMRPC/iodata.py

Covers functions not tested in test_02_iodata.py:
- gen_file_url
- chk_file_url
- gen_fname
"""
import os
import numpy as np
import context
import aMRPC.iodata as iod

DATA_DIR = './tests/data'


# --- gen_file_url ---

def test_gen_file_url_with_dir():
    """gen_file_url concatenates directory and filename."""
    url = iod.gen_file_url('test.txt', '/some/dir')
    assert url == '/some/dir/test.txt'


def test_gen_file_url_default_dir():
    """gen_file_url uses inDir when no directory is given."""
    old_in = iod.inDir
    iod.inDir = '/default/dir'
    url = iod.gen_file_url('test.txt')
    assert url == '/default/dir/test.txt'
    iod.inDir = old_in


# --- chk_file_url ---

def test_chk_file_url_existing():
    """chk_file_url returns True for an existing file."""
    assert iod.chk_file_url('InputParameters.txt', DATA_DIR)


def test_chk_file_url_nonexistent():
    """chk_file_url returns False for a non-existing file."""
    assert not iod.chk_file_url('nonexistent_file_xyz.txt', DATA_DIR)


# --- gen_fname ---

def test_gen_fname_basic():
    """gen_fname produces correct filename for known file type."""
    # FilePfx[1] = "roots", FileSfx[1] = ".txt"
    fname = iod.gen_fname(1)
    assert fname == 'roots.txt'


def test_gen_fname_with_nr():
    """gen_fname appends Nr parameter."""
    fname = iod.gen_fname(1, Nr='3')
    assert fname == 'roots_Nr3.txt'


def test_gen_fname_with_no():
    """gen_fname appends No parameter."""
    fname = iod.gen_fname(1, No='4')
    assert fname == 'roots_No4.txt'


def test_gen_fname_with_multiple_params():
    """gen_fname appends multiple parameters in order."""
    fname = iod.gen_fname(3, Nr='2', No='3', ths='0.01')
    # FilePfx[3] = "Hdict", FileSfx[3] = ".p"
    assert fname == 'Hdict_Nr2_No3_ths0.01.p'


def test_gen_fname_with_txt():
    """gen_fname appends txt parameter."""
    fname = iod.gen_fname(1, txt='_custom')
    assert fname == 'roots_custom.txt'


def test_gen_fname_all_file_types():
    """gen_fname works for all registered file types."""
    for fkt in iod.FilePfx:
        fname = iod.gen_fname(fkt)
        assert fname.startswith(iod.FilePfx[fkt])
        assert fname.endswith(iod.FileSfx[fkt])


# --- Round-trip tests with DataFrame ---

def test_write_load_eval_points_roundtrip():
    """Write and load back evaluation points preserving shape and values."""
    iod.inDir = DATA_DIR
    iod.outDir = DATA_DIR
    fname = 'roundtrip_test.txt'
    n, m = 50, 4
    rpts = np.random.randn(n, m)
    iod.write_eval_points(rpts, fname)
    data = iod.load_eval_points(fname)
    k, l = data.shape
    assert k == n
    assert l == m
    # Values should be close (limited by ASCII formatting precision)
    np.testing.assert_allclose(data.values, rpts, atol=1e-5)
    # Cleanup
    url = iod.gen_file_url(fname, DATA_DIR)
    if os.path.exists(url):
        os.remove(url)


def test_load_eval_points_nonexistent():
    """Loading a non-existent file should return an empty DataFrame."""
    data = iod.load_eval_points('does_not_exist_xyz.txt', DATA_DIR)
    assert data.empty
