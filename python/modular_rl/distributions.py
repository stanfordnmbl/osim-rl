import numpy as np

def categorical_sample(prob_nk):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_nk = np.asarray(prob_nk)
    assert prob_nk.ndim == 2
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    return np.argmax(csprob_nk > np.random.rand(N,1), axis=1)

TINY = np.finfo(np.float64).tiny

def categorical_kl(p_nk, q_nk):
    p_nk = np.asarray(p_nk,dtype=np.float64)
    q_nk = np.asarray(q_nk,dtype=np.float64)
    ratio_nk = p_nk / (q_nk+TINY) # so we don't get warnings
    # next two lines give us zero when p_nk==q_nk==0 but inf when q_nk==0 
    ratio_nk[p_nk==0] = 1
    ratio_nk[(q_nk==0) & (p_nk!=0)] = np.inf
    return (p_nk * np.log(ratio_nk)).sum(axis=1)

def categorical_entropy(p_nk):
    p_nk = np.asarray(p_nk,dtype=np.float64)
    p_nk = p_nk.copy()
    p_nk[p_nk == 0] = 1
    return (-p_nk * np.log(p_nk)).sum(axis=1)
