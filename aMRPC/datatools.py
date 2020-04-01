"""
datatools.py - provides data management functions,
- load data
- generate polynomial bases and multi-wavelet (mw) details
- generate and apply quadrature
- preform mw-adaptivity
- save generated data

@author: kroeker
"""
import pandas as pd
import numpy as np
from . import polytools as pt
from . import utils as u
#from . import wavetools as wt

def genHankel(dataframe, srcs, nr_range, n_o):
    """ generates Hankel matrixes for each Nri, writes in h_dict """
    #Nr=max(NrRange)
    h_dict = {}

    for src in srcs:
        data = dataframe[src]
        for anr in nr_range:
            for nri in range(2**anr):
                l_b, r_b = cmp_lrb(anr, nri)
                qlb = data.quantile(l_b)
                qrb = data.quantile(r_b)
                mask = cmp_quant_domain(data, qlb, qrb)
                key = u.gen_dict_key(anr, nri, src)
                h_dict[key] = pt.Hankel(n_o+1, data[mask])
                #print(H)
    return h_dict

def gen_nr_range_bds(dataframe, srcs, nr_range):
    """ generates dictionary with boundaries of MR-elements """
    nrb_dict = {}
    for src in srcs:
        data = dataframe[src]
        for anr in nr_range:
            for nri in range(2**anr):
                l_b, r_b = cmp_lrb(anr, nri)
                qlb = data.quantile(l_b)
                qrb = data.quantile(r_b)
                key = u.gen_dict_key(anr, nri, src)
                nrb_dict[key] = (qlb, qrb)
    return nrb_dict

def gen_roots_weights(h_dict, method):
    """
    generates dictionaries with roots and weights
    generated by methods 0- Gautschi / 1-Karniadakis & Kirby
    """
    roots = {}
    weights = {}
    for key in h_dict:
        r, w = pt.gen_rw(h_dict[key], method)
        roots[key] = r
        weights[key] = w
    return roots, weights

def cmp_lrb(n_r, nri):
    """ computes left and right bounds for use in dataframe.quantile() """
    rcf = 2**(n_r)
    l_b = nri/rcf
    r_b = (nri+1)/rcf
    return l_b, r_b

def cmp_quant_domain(data, qlb, qrb):
    """ generates bool array with 1 for x in [qlb,qrb], 0 else """
    b_mask = (data >= qlb) & (data <= qrb)
    return b_mask

def cmp_mw_quant_domain(roots, nrb_dict, nrs, nris, cols):
    """
    generates bool array with 1 for r inside of
    [a_0,b_0]x..x[a_d,b_d], 0 else
    """
    n = roots.shape[0]
    ndim = len(nrs)
    assert ndim == len(nris)
    b_mask = np.ones(n, dtype=bool)
    for d, c in enumerate(cols):
        key = u.gen_dict_key(nrs[d], nris[d], c)
        qlb, qrb = nrb_dict[key]
        b_mask = b_mask & cmp_quant_domain(roots[c], qlb, qrb)
    return b_mask

def cmp_mv_quant_domain_mk(roots, nrb_dict, mkey):
    """
    generates bool array with 1 for r inside of
    [a_0,b_0]x..x[a_d,b_d], 0 else
    for given multikey mkey
    """
    n, sdim = roots.shape
    assert sdim == len(mkey)
    b_mask = np.ones(n, dtype=bool)
    for d in range(sdim):
        key = mkey[d]
        qlb, qrb = nrb_dict[key]
        b_mask = b_mask & cmp_quant_domain(roots[:, d], qlb, qrb)
    return b_mask

def gen_pcs(h_dict, method):
    """
    generated dictionaries with matrices of monic orthogonal
    polynomials, method 0: Gautschi style, 1: Sergey Style.
    """
    cfs = {}
    for key in h_dict:
        cfs[key] = pt.gen_pc_mx(h_dict[key], method)
    return cfs

def gen_npcs(pc_dict, roots, weights):
    """ generates dictionary with coeficients of orthonormal polynomials """
    n_cfs = {}
    for key in pc_dict:
        n_cfs[key] = pt.gen_npc_mx(pc_dict[key], roots[key], weights[key])
    return n_cfs

def gen_npcs_mm(pc_dict, H_dict):
    n_cfs = {}
    for key in pc_dict:
        n_cfs[key] = pt.gen_npc_mx_mm(pc_dict[key], H_dict[key])
    return n_cfs

def pcfs4eval(pc_dict, mkey, alpha):
    """ provides PC-Cfs for multi-polynomial with degrees in alpha """
    mdeg = max(alpha)
    alen = len(alpha)
    assert alen == len(mkey)
    rcfs = np.zeros((alen, mdeg+1))
    for d in range(alen):
        ad = alpha[d]
        cfs = pc_dict[mkey[d]]
        rcfs[d, :] = cfs[ad, 0:mdeg+1]
    return rcfs

def Gauss_quad(func, roots, weights):
    """ Gauss quadrature with roots and weights """
    assert len(roots) == len(weights)
    return np.inner(func(roots), weights)

def inner_prod(f_la, g_la, roots, weights):
    """ inner product <f,g,>, computed by Gauss quadrature """
    return Gauss_quad(lambda x: f_la(x)*g_la(x), roots, weights)

def Gauss_quad_idx(fct, multi_key, roots, weights):
    """ multi-dimensional Gauss quad on multiKey of
    f_0(x_0)*...*f_dim-1(x_dim-1), Func=(f_0,...,f_dim-1)
    """
    dim = len(multi_key)
    r_mk, w_mk = gen_rw_4mkey(multi_key, roots, weights)
    if isinstance(fct, tuple):
        assert dim == len(fct)
        ret = Gauss_quad_arr(fct, r_mk, w_mk)
    else:
        ret = Gauss_quad_fct(fct, r_mk, w_mk)
    return ret

def inner_prod_multi_idx(fkt_f, fkt_g, multi_key, roots, weights):
    """ <F,G>, for for multi-index multiKey """
    dim = len(multi_key)
    r_mk, w_mk = gen_rw_4mkey(multi_key, roots, weights)
    assert type(fkt_f) == type(fkt_g)
    t_f = isinstance(fkt_f, tuple)
    t_g = isinstance(fkt_g, tuple)
    if t_f and t_g:
        flen = len(fkt_f)
        assert flen == len(fkt_g)
        assert flen == dim
        ret = inner_prod_tuples(fkt_f, fkt_g, r_mk, w_mk)
    else:
        ret = inner_prod_fct(fkt_f, fkt_g, r_mk, w_mk)
    return ret

def Gauss_quad_arr(fct_tup, roots, weights):
    dim = len(fct_tup)
    evals = roots.shape[0]
    t_a = dim == roots.shape[1]
    t_b = roots.size == weights.size
    assert t_a and t_b
    S = 0
    for l in range(evals):
        tmp = 1
        for d in range(dim):
            key = (l, d)
            tmp = tmp*fct_tup[d](roots[key])*weights[key]
        S += tmp
    return S

def inner_prod_tuples(F, G, roots, weights):
    """ <F,G>, F,G are given by tuples """
    dim = len(F)
    evals = roots.shape[0]
    a = dim == len(G)
    b = roots.size == weights.size
    c = dim == roots.shape[1]
    assert a and b and c
    S = 0
    for l in range(evals):
        tmp = 1
        for d in range(dim):
            key = (l, d)
            x = roots[key]
            tmp = tmp*F[d](x)*G[d](x)*weights[key]
        S += tmp
    return S

def Gauss_quad_fct(fct, roots, weights):
    assert roots.shape == weights.shape
    arr = fct(roots)*weights
    return sum(np.prod(arr, axis=1))


def inner_prod_fct(f_la, g_la, roots, weights):
    """ <f_la, g_la> on roots-n-weights, for lambdas f_la, and g_la """
    assert roots.shape == weights.shape
    quad_arr = f_la(roots)*g_la(roots)*weights
    return sum(np.prod(quad_arr, axis=1))


def inner_prod_arr_fct(arr, f_la, roots, weights, srcs):
    """ inner product of data in Arr and F(Roots) weighted with Weigths """
    assert roots.shape == weights.shape
    quad_arr = f_la(roots[srcs])*weights[srcs]
    prod_arr = np.prod(quad_arr, axis=1)
    return np.inner(arr, prod_arr)


def gen_rw_4mkey(mkey, roots, weights):
    """ generates roots and weights arrays from dict's Roots and Weights for multikey mkey """
    cols = len(mkey)
    # ls=[]
    # for d in range(cols):
    #     key=mKey[d]
    #     #print("k->r",key,Roots[key])
    #     lk=len(Roots[key])
    #     ls.append(lk)
    ls = [len(roots[key]) for key in mkey]
    lens = np.array(ls)
    lines = lens.prod()
    I = u.midx4quad(lens)
    r_roots = np.zeros((lines, cols))
    r_weights = np.zeros([lines, cols])
    for c in range(cols):
        key = mkey[c]
        r = roots[key]
        w = weights[key]
        idx = I[:, c]
        #print(idx,r)

        r_roots[:, c] = r[idx]
        r_weights[:, c] = w[I[:, c]]
    return r_roots, r_weights

def get_rw_4mkey(mkLst, roots, weights):
    """ generates eval. points and weights  np.arrays and
    (point number)->mkey list   for multi-keys in mkArr list """
    #tcnt=len(mkLst)
    R = np.array([])
    W = np.array([])
    mk_lst_long = [] #  multi-key in order of apperance
    points4mk = 0
    for mkey in mkLst:
        r, w = gen_rw_4mkey(mkey, roots, weights)
        points4mk = len(r)
        mk_lst_long = mk_lst_long+[mkey for c in range(points4mk)]
        if len(R) == 0:
            R = r
            W = w
        else:
            R = np.concatenate([R, r], axis=0)
            W = np.concatenate([W, w], axis=0)
    return R, W, mk_lst_long

def get_rw_4nrs(nrs, srcs, roots, weights):
    """ generates eval. points and weights for a Nr level """
    dim = len(nrs)
    nris = np.zeros(dim)
    nri_cnt = np.zeros(dim, dtype=int)
    divs = np.zeros(dim)
    R = np.array([])
    W = np.array([])
    mk_lst_long = [] #  multi-key in order of apperance

    for d in range(dim):
        #aNr = Nrs[d]
        nri_cnt[d] = 2**nrs[d]
        divs[d] = np.prod(nri_cnt[0:d])

    tnri_cnt = np.prod(nri_cnt)
    for l in range(tnri_cnt):
        for d in range(dim):
            nris[d] = (l//divs[d] % nri_cnt[d])
        mkey = u.gen_multi_key(nrs, nris, srcs)
        r, w = gen_rw_4mkey(mkey, roots, weights)
        #r=np.reshape(r,(-1,dim))
        mk_lst_long = mk_lst_long+[mkey for c in range(len(r))]
        if len(R) == 0:
            R = r
            W = w
        else:
            R = np.concatenate([R, r], axis=0)
            W = np.concatenate([W, w], axis=0)
    return R, W, mk_lst_long

def gen_quant_dict(dataframe, srcs, nr_range, wvt):
    """
    generates dictionary of quantiles on roots for each Nri and Nr in NrRange
    accorting to roots stored in already initialized object of wavetools wt
    using data in dataframe for columns in srcs
    """
    q_dict = {}

    for src in srcs:
        data = dataframe[src]
        for anr in nr_range:
            for nri in range(2**anr):
                quants = wvt.cmp_data_on_roots(data, anr, nri)
                key = u.gen_dict_key(anr, nri, src)
                q_dict[key] = quants
    return q_dict

def gen_detail_dict(q_dict, wvt, dicts=0):
    """
    generates dictionary of Details on roots for each set of quantiles
    stored in Qdict, using initalized wavetool wvt
    dits=0: sum(abs(details)), 1: lDetails only, 2 both
    """
    det_dict = {}
    ldet_dict = {}
    s = dicts in (0, 2)
    l = dicts >= 1
    for key, data in q_dict.items():
        #Nr=key[u.ParPos['Nr']]
        ldetails = wvt.cmp_details(data)
        if l:
            ldet_dict[key] = ldetails
        if s:
            det_dict[key] = sum(abs(ldetails))
    if dicts == 0:
        ret = det_dict
    elif dicts == 1:
        ret = ldet_dict
    else:
        return  det_dict, ldet_dict
    return ret

def mark_dict4keep(d_dict, thres):
    """ marks the details>= threshold for keep """
    k_dict = {}
    for key, data in d_dict.items():
        b = data >= thres
        k_dict[key] = b
    return k_dict

def get_true_nodes(k_dict, key):
    """
    checks leafs of the tree bottom ab, leafs only highest "True"-level on True
    """
    ret = 0
    if k_dict[key]:
        ret = 1

    nri = key[u.ParPos['Nri']]
    nr = key[u.ParPos['Nr']]
    src = key[u.ParPos['src']]
    lnri = 2*nri # left kid
    rnri = lnri+1 # right kid
    lkey = u.gen_dict_key(nr+1, lnri, src)
    rkey = u.gen_dict_key(nr+1, rnri, src)
    lex = lkey in k_dict.keys()
    rex = rkey in k_dict.keys()
    if lex and rex:
        l = get_true_nodes(k_dict, lkey)
        r = get_true_nodes(k_dict, rkey)
        kids = l + r
        if kids > 0:
            k_dict[key] = False
            if  l == 0:
                k_dict[lkey] = True
                ret += 1
            if  r == 0:
                k_dict[rkey] = True
                ret += 1
                ret += kids
    return ret

def get_top_keys(k_dict, srcs):
    """ returns set with top level (True) keys only (bottom up)"""
    t_keys = k_dict.copy()
    for src in srcs:
        root_key = u.gen_dict_key(0, 0, src) #multi-key on zero-level (root)
        if root_key in t_keys.keys():
            cnt = get_true_nodes(t_keys, root_key)
            if cnt == 0:
                t_keys[root_key] = True # set root node to True if no leafs are selected
    return t_keys

def gen_mkey_list(k_dict, srcs):
    """ generates array of multi-keys from the dictionary Kdict """
    isrcs = u.inv_src_arr(srcs)
    k_lst = [[] for s in isrcs]
    srclen = len(isrcs)
    sidx = u.ParPos['src']
    for key, chk in k_dict.items():
        if chk:
            idx = key[sidx]
            k_lst[isrcs[idx]].append(key)
    alen = [len(c) for c in k_lst]
    I = u.midx4quad(alen)
    ilen, _ = np.shape(I)
    # required also for 1-dim case, to generate multikey -> tuple(tuple)
    return [tuple([k_lst[c][I[i, c]] for c in range(srclen)]) for i in range(ilen)]


def gen_mkey_sid_rel(samples, mk_lst, nrb_dict):
    """
    generates long sample->[multi-key ]
    multi-key -> np.array([sample id]) dictionaries

    return : sid2mk, mk2sids
    """
    sample_cnt, _ = samples.shape
    sids = np.arange(sample_cnt)
    sid2mk = {}
    mk2sids = {}
    for mkey in mk_lst:
        B = cmp_mv_quant_domain_mk(samples, nrb_dict, mkey)
        mk2sids[mkey] = sids[B]
        for sid in mk2sids[mkey]:
            if sid in sid2mk:
                sid2mk[sid] += mkey
            else:
                sid2mk[sid] = [mkey]
    return sid2mk, mk2sids

def sample2mkey(sample, mk_lst, nrb_dict, find_all_mkeys=False):
    """ finds first, all multi-key in NR-Bounds dictrionary corresponding to the
    multi-element containing the sample
    """
    ndim = len(sample)
    smk_list = []
    for mkey in mk_lst:
        chk = True
        for d in range(ndim):
            qlb, qrb = nrb_dict[mkey[d]]
            chk = chk & cmp_quant_domain(sample[d], qlb, qrb)
        if chk:
            if find_all_mkeys:
                smk_list += [mkey]
            else:
                smk_list = [mkey]
    return smk_list

def cmp_resc_cfl(anr_list):
    """
    computes rescaling cfs c=<phi^Nr_l,0,phi^0_0,0>.
    Is relevant for computing Exp. / Var. from coefficients only
    for aNRlist [aNr_0,...,aNr_d]
    """
    cft = 1
    for anr in anr_list:
        cft /= 2**(anr)
    return cft

def cmp_resc_cf(mkey):
    """
    computes rescaling coeficients c=<phi^Nr_l,0,phi^0_0,0>.
    Is relevant for computing Exp. / Var. from coefficients only
    for multi-key mKey
    """
    dim = len(mkey)
    nr_pos = u.ParPos['aNr']
    cft = 1
    for d in range(dim):
        key = mkey[d]
        cft /= 2**(key[nr_pos])
    return cft

def gen_rcf_dict(mk_list):
    """
    Generates dictionary with rescaling coefficients for ech
    multi-key in mkList [(mk),...]
    """
    rcf_dict = {}
    for mkey in mk_list:
        rcf_dict[mkey] = cmp_resc_cf(mkey)
    return rcf_dict

def gen_amrpc_rec(samples, mk_list, alphas, f_cfs, npc_dict, nrb_dict,
                  mk2sid):
    """
    Generates function reconstruction
    f(sample, x) = sum_(p in alphas) f_cfs(sample, p,  x) * pol(alpha_p, sample)

    Parameters
    ----------
    samples : mp.array
        samples for evaluation, samples[i] = [s_0, s_1, ..., s_n].
    mk_list : list of tuples
        (unique) list of multi-keys ((key,0),...,(key, n)).
    alphas : np.array
        matrix of multi-indexes representing pol. degrees of multi-variate polynomials.
    f_cfs : np.array
        reconstr. coefficients f_cfs[sample,alpha_p,idx_x].
    npc_dict : dict
        dictionary of normed picewise polynomials.
    nrb_dict : dict
        dictionary of stochastic-element boundaries.
    mk2sid : dict
        (multi key) -> sample id dictionary.

    Returns
    -------
    f_rec : np.array
        ampc reconstruction of the function f, f_rec[sample_id, idx_x].

    """
    n_s = samples.shape[0]
    n_x = f_cfs.shape[2]
    f_rec = np.zeros((n_s, n_x))

    _, mk2sid_loc = gen_mkey_sid_rel(samples, mk_list, nrb_dict)
    p_vals = gen_pol_on_samples_arr(samples, npc_dict, alphas, mk2sid_loc)
    for mkey, sids_l in mk2sid_loc.items():
        sids = mk2sid[mkey]
        for idx_p in range(alphas.shape[0]):
            for sid_l in sids_l:
                f_rec[sid_l, :] += f_cfs[sids[0], idx_p, :] * p_vals[idx_p, sid_l]

    return f_rec

def gen_pol_on_samples_arr(samples, npc_dict, alphas, mk2sid):
    """
    generates np.array with pol. vals for each sample and pol. degree
    samples : samples for evaluation (evtl. samples[srcs] )
    nPCdict: pol. coeff. dictionary
    Alphas: matrix of multiindexes representing pol. degrees
    mk2sid: multi-key -> sample id's (sid lists should be disjoint)
    return: pol_vals[sample id] = [pol_i:p_i(x_0),...p_i(x_end)]
    """
    n_s, _ = samples.shape
    p_max = alphas.shape[0]
    pol_vals = np.zeros((p_max, n_s))

    for mkey in mk2sid:
        sids = mk2sid[mkey]
        for idx_p in range(p_max):
            pcfs = pcfs4eval(npc_dict, mkey, alphas[idx_p])
            pvals = pt.pc_eval(pcfs, samples[sids, :])
            pol_vals[idx_p, sids] = np.prod(pvals, axis=1)
    return pol_vals

def gen_amrpc_dec_ls(data, pol_vals, mk2sid):
    """
    computes the armpc-decomposition coefficients f_p of
    f(x,theta) = sum_p f_p(x) * pol_p(sample)
    on each sample, (sid, p, x) by least-squares

    Parameters
    ----------
    data : np.array[sample_id, space_point_nr]
        evaluations of f on samples theta for each space point x
    pol_vals : np.array[sample_id, pol_degree]
        eval of picevise polynomials for each sample_id and pol_degree.
    mk2sid : dictionary
        MR-related multi-key -> sample id.

    Returns
    -------
    cf_ls_4s: np.array of f_i for [sid, p, x_i]

    """
    # compute function coefficients by least-squares
    # Fct coefs on each sample, (sid, p, x): by LS
    n_tup = data.shape
    if len(n_tup) > 1:
        n_x = n_tup[1]
    else:
        n_x = 1
    n_s = n_tup[0]
    p_max = pol_vals.shape[0]
    cf_ls_4s = np.zeros((n_s, p_max, n_x))
    for sids in mk2sid.values():
        phi = pol_vals[:, sids].T
        for idx_x in range(n_x):
            # v, resid, rank, sigma = linalg.lstsq(A,y)
            # solves Av = y using least squares
            # sigma - singular values of A
            if n_s > 1:
                #v_ls, resid, rank, sigma = np.linalg.lstsq(
                #    Phi, data[sids, idx_x], rcond=None) # LS - output
                v_ls, _, _, _ = np.linalg.lstsq(
                    phi, data[sids, idx_x], rcond=None) # LS - output
            else:
                v_ls = data[idx_x]/phi
            cf_ls_4s[sids, :, idx_x] = v_ls
    return cf_ls_4s

def gen_amrpc_dec_q(data, pol_vals, mk2sid, weights):
    """
    computes the armpc-decomposition coefficients f_p of
    f(x,theta) = sum_p f_p(x) * pol_p(sample)
    on each sample, (sid, p, x) by using Gauss quadrature

    Parameters
    ----------
    data : np.array
        [sample_id, space_point_nr]
        evaluations of f on samples theta for each space point x.
    pol_vals : np.array
        np.array[sample_id, pol_degree]
        eval of picevise polynomials for each sample_id and pol_degree.
    mk2sid : dictionary
        MR-related multi-key -> sample id.
    weights : np.array
        Gaussian weights.

    Returns
    -------
    cf_q_4s : np.array
         f_i for [sid, p, x_i].

    """
    n_s, n_x = data.shape
    p_max = pol_vals.shape[0]
    cf_q_4s = np.zeros((n_s, p_max, n_x))
    weight_prod = np.prod(np.array(weights), axis=1)
    weighted_data = data.T * weight_prod # weighted samples
    #Fkt coefs.  on each sample, (sid, p, x) by quad

    for p in range(p_max):
        data_pol_4_p = weighted_data * pol_vals[p, :]
        for idx_x in range(n_x):
            for sids in mk2sid.values():
                if n_s > 1:
                    cf_q_4s[sids, p, idx_x] = sum(data_pol_4_p[idx_x, sids])
                else:
                    cf_q_4s[sids, p, idx_x] = data_pol_4_p[idx_x]
    return cf_q_4s

def cf_2_mean_var(cf_4s, rc_dict, mk2sid):
    """
    Computes mean and variance from the aMR-PC decompositions coeficients

    Parameters
    ----------
    cf_4s : np.array
        [sample_id, pol_degree, x_idx], function coefs.
    rc_dict : dictionary
        (multi-key)->rescaling coefficent.
    mk2sid : dictionary
        (multi-key)->[sample_id].

    Returns
    -------
    mean : np.array
        expectation for all x.
    variance : np.array
        variance for all x.

    """
    _, p_max, n_x = cf_4s.shape
    mean = np.zeros(n_x)
    variance = np.zeros(n_x)
    for mkey, sids in mk2sid.items():
        sid = sids[0]  # first sample related to multi-key
        mean += cf_4s[sid, 0, :] * rc_dict[mkey]
        for p_d in range(p_max):
            variance += rc_dict[mkey] * (cf_4s[sid, p_d, :]**2)
    variance -= mean**2

    return mean, variance

def main():
    """ some tests """
    # data location
    url = '../data/InputParameters.txt'

    # load data
    dataframe = pd.read_csv(url, header=None, sep='\s+ ', engine='python')
    n_r = 2
    nr_range = np.arange(n_r+1)
    n_o = 2
    srcs = np.arange(0, 1)
    method = 0
    h_dict = genHankel(dataframe, srcs, nr_range, n_o)
    print(h_dict)
    # further with test004
    roots, weights = gen_roots_weights(h_dict, method)
    print(roots, weights)
