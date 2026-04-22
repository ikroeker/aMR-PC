"""
Extended unit tests for aMRPC/sobol.py

Covers functions not tested in test_06_sobol.py:
- gen_sidx_subsets
- sobol_tot_sen_pc
- sobol_tot_sens
- sobol_idx_pc (edge cases, vector output)
- sum of all Sobol indices = 1
"""
import math
import numpy as np
import pandas as pd
import context
import aMRPC.sobol as sob
import aMRPC.datatools as dt
import aMRPC.utils as u

# Use Ishigami function for validation
SAMPLE_CNT = 100000
P_ORDER = 7
METHOD = 1
SRCS = [0, 1, 2]
DIM = len(SRCS)
ALPHAS = u.gen_multi_idx(P_ORDER, DIM)
P_PC = ALPHAS.shape[0]

A = 7
B = 0.1
TOL = 2e-2

ISH_FCT = lambda X: A * np.sin(X[:, 1])**2 + (1 + B * np.power(X[:, 2], 4)) * np.sin(X[:, 0])


def ishigami_exact_sensitivity(a, b):
    """Analytical Sobol indexes of the Ishigami function."""
    D = (a**2) / 8 + b * (math.pi**4) / 5 + (b**2) * (math.pi**8) / 18 + 0.5
    DS = {}
    DS[frozenset([0])] = b * (math.pi**4) / 5 + (b**2) * (math.pi**8) / 50 + 0.5
    DS[frozenset([1])] = (a**2) / 8
    DS[frozenset([2])] = 0
    DS[frozenset([0, 1])] = 0
    DS[frozenset([1, 2])] = 0
    DS[frozenset([0, 2])] = 8 * b**2 * math.pi**8 / 225
    DS[frozenset([0, 1, 2])] = 0
    sob_cfs = {}
    for cf_idx, var_cf in DS.items():
        sob_cfs[cf_idx] = var_cf / D
    return D, sob_cfs


def gen_pc_coefs():
    """Generate PC coefficients via LS on the Ishigami function."""
    np.random.seed(3)
    x = np.random.uniform(-math.pi, math.pi, (SAMPLE_CNT, 3))
    dataset = pd.DataFrame(x)

    pc_nr_range = [0]
    h_dict = dt.genHankel(dataset, SRCS, pc_nr_range, P_ORDER)
    pc_roots, pc_weights = dt.gen_roots_weights(h_dict, METHOD)
    pc_dict = dt.gen_pcs(h_dict, METHOD)
    npc_dict = dt.gen_npcs_mm(pc_dict, h_dict)
    nrb_dict = dt.gen_nr_range_bds(dataset, SRCS, pc_nr_range)

    roots_eval, _, mk_lst_l = dt.get_rw_4nrs(np.zeros(DIM), SRCS,
                                              pc_roots, pc_weights)
    mk_lst = list(set(mk_lst_l))
    _, mk2sid = dt.gen_mkey_sid_rel(roots_eval, mk_lst, nrb_dict)
    pol_vals = dt.gen_pol_on_samples_arr(roots_eval, npc_dict, ALPHAS, mk2sid)
    y_rt = ISH_FCT(roots_eval)

    cf_ls = np.zeros((len(y_rt), P_PC))
    for sids in mk2sid.values():
        phi = pol_vals[:, sids].T
        v_ls, _, _, _ = np.linalg.lstsq(phi, y_rt[sids], rcond=-1)
        cf_ls[sids, :] = v_ls

    pc_cfs = cf_ls[0]
    return pc_cfs


# --- gen_sidx_subsets ---

def test_gen_sidx_subsets_single():
    """gen_sidx_subsets of a single element returns itself."""
    result = sob.gen_sidx_subsets(frozenset([0]))
    assert frozenset([0]) in result
    assert len(result) == 1


def test_gen_sidx_subsets_two():
    """gen_sidx_subsets of {0,1} returns {0}, {1}, {0,1}."""
    result = sob.gen_sidx_subsets(frozenset([0, 1]))
    assert frozenset([0]) in result
    assert frozenset([1]) in result
    assert frozenset([0, 1]) in result
    assert len(result) == 3


def test_gen_sidx_subsets_three():
    """gen_sidx_subsets of {0,1,2} has 7 subsets."""
    result = sob.gen_sidx_subsets(frozenset([0, 1, 2]))
    assert len(result) == 7  # 2^3 - 1


def test_gen_sidx_subsets_matches_gen_idx_subsets():
    """gen_sidx_subsets on full set should match gen_idx_subsets."""
    full = frozenset(range(DIM))
    result_sidx = sob.gen_sidx_subsets(full)
    result_idx = sob.gen_idx_subsets(DIM)
    assert result_sidx == result_idx


# --- gen_idx_subsets ---

def test_gen_idx_subsets_dim1():
    """gen_idx_subsets for dim=1 should return {frozenset({0})}."""
    result = sob.gen_idx_subsets(1)
    assert len(result) == 1
    assert frozenset([0]) in result


def test_gen_idx_subsets_dim2():
    """gen_idx_subsets for dim=2 should return 3 subsets."""
    result = sob.gen_idx_subsets(2)
    assert len(result) == 3
    assert frozenset([0]) in result
    assert frozenset([1]) in result
    assert frozenset([0, 1]) in result


# --- sobol_tot_sen_pc ---

def test_sobol_tot_sen_pc():
    """Total sensitivity for Ishigami matches analytical values."""
    pc_cfs = gen_pc_coefs()
    src_idx = sob.gen_idx_subsets(DIM)
    _, sob_cfs = ishigami_exact_sensitivity(A, B)

    # Total sensitivity for x_0 = S_0 + S_{0,1} + S_{0,2} + S_{0,1,2}
    tot_0 = sob.sobol_tot_sen_pc(pc_cfs, ALPHAS, src_idx, [0])
    expected_tot_0 = (sob_cfs[frozenset([0])]
                      + sob_cfs[frozenset([0, 1])]
                      + sob_cfs[frozenset([0, 2])]
                      + sob_cfs[frozenset([0, 1, 2])])
    assert abs(tot_0 - expected_tot_0) < TOL

    # Total sensitivity for x_1
    tot_1 = sob.sobol_tot_sen_pc(pc_cfs, ALPHAS, src_idx, [1])
    expected_tot_1 = (sob_cfs[frozenset([1])]
                      + sob_cfs[frozenset([0, 1])]
                      + sob_cfs[frozenset([1, 2])]
                      + sob_cfs[frozenset([0, 1, 2])])
    assert abs(tot_1 - expected_tot_1) < TOL


# --- sobol_tot_sens (from dictionary) ---

def test_sobol_tot_sens():
    """sobol_tot_sens from Sobol dictionary matches sobol_tot_sen_pc."""
    pc_cfs = gen_pc_coefs()
    src_idx = sob.gen_idx_subsets(DIM)

    # Build Sobol dictionary
    sob_dict = {}
    for src in src_idx:
        sob_dict[src] = sob.sobol_idx_pc(pc_cfs, ALPHAS, list(src))

    tot_0_pc = sob.sobol_tot_sen_pc(pc_cfs, ALPHAS, src_idx, [0])
    tot_0_dict = sob.sobol_tot_sens(sob_dict, src_idx, [0])
    assert abs(tot_0_pc - tot_0_dict) < 1e-10


# --- Sum of Sobol indices = 1 ---

def test_sobol_indices_sum_to_one():
    """All Sobol indices for aPC Ishigami should sum to 1."""
    pc_cfs = gen_pc_coefs()
    src_idx = sob.gen_idx_subsets(DIM)
    total = 0
    for src in src_idx:
        total += sob.sobol_idx_pc(pc_cfs, ALPHAS, list(src))
    assert abs(total - 1.0) < TOL


# --- Sobol with zero variance edge case ---

def test_sobol_idx_pc_zero_variance():
    """sobol_idx_pc handles zero variance (constant function) producing 0 or NaN."""
    # Constant function: all coefficients except c0 are zero
    pc_cfs = np.zeros(P_PC)
    pc_cfs[0] = 5.0
    result = sob.sobol_idx_pc(pc_cfs, ALPHAS, [0])
    # With zero variance, result is 0/0 = NaN for scalar case (no eps protection)
    # or 0 if var_pc was set to 1 by the eps threshold
    assert result == 0 or np.isnan(result)
