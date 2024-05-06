import numpy as np
from scipy.io import wavfile
import scipy as sp

def autocorrelate(x, d, K):
    """
    Calculates the autocorrelation matrix Rx and the autocorrelation vector rdx
    from the K past samples of x and d
    """
    # Compute correlations
    Rx = np.zeros(K)
    rdx = np.zeros(K)

    for i in range(K):
        Rx[i] = np.mean(x[:(len(x)-K)] * x[i:(len(x) + i - K)])
        rdx[i] = np.mean(x[:(len(x)-K)] * d[i:(len(x) + i - K)])
    
    return sp.linalg.toeplitz(Rx), rdx

def read_normalized(path):
    fs, sig = wavfile.read(path)
    return fs, sig, sig / 32767