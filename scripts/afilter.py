import numpy as np
from scipy import signal
import sigutil

def optimal_lms(x, d):
    """
    Computes an adaptive filtering step using the optimal LMS algorithm. The naive approach is very slow, hence this should
    probably only be used for benchmarking purposes
    Args:
        x: Samples x[n-k]...x[n] of noise corrolated with the noise corrupting s[n]
        d: Samples s[n-k]...s[n] but corrupted by noise
    Returns:
        a single value s'[n] approximating s[n]
    """
    n = len(x) - 1

    R_Xn = sigutil.autocorrelate(x)
    r_DX = sigutil.correlate(d, x)

    f_n = np.linalg.solve(R_Xn, r_DX)

    return d[n] - np.convolve(f_n, x)[n]