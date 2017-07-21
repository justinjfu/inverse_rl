import numpy as np
import scipy as sp
import scipy.stats

def rle(inarray):
        """ run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.array(inarray)                  # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])

def split_list_by_lengths(values, lengths):
    """

    >>> split_list_by_lengths([0,0,0,1,1,1,2,2,2], [2,2,5])
    [[0, 0], [0, 1], [1, 1, 2, 2, 2]]
    """
    assert np.sum(lengths) == len(values)
    idxs = np.cumsum(lengths)
    idxs = np.insert(idxs, 0, 0)
    return [ values[idxs[i]:idxs[i+1] ] for i in range(len(idxs)-1)]

def clip_sing(X, clip_val=1):
    U, E, V = np.linalg.svd(X, full_matrices=False)
    E = np.clip(E, -clip_val, clip_val)
    return U.dot(np.diag(E)).dot(V)

def gauss_log_pdf(params, x):
    mean, log_diag_std = params
    N, d = mean.shape
    cov =  np.square(np.exp(log_diag_std))
    diff = x-mean
    exp_term = -0.5 * np.sum(np.square(diff)/cov, axis=1)
    norm_term = -0.5*d*np.log(2*np.pi)
    var_term = -0.5 * np.sum(np.log(cov), axis=1)
    log_probs = norm_term + var_term + exp_term
    return log_probs #sp.stats.multivariate_normal.logpdf(x, mean=mean, cov=cov)

def categorical_log_pdf(params, x, one_hot=True):
    if not one_hot:
        raise NotImplementedError()
    probs = params[0]
    return np.log(np.max(probs * x, axis=1))

