import numpy as np
from scipy import signal
import scripts.sigutil as sigutil

def fixed_optimal_lms(x, d, K):
    """
    Computes a fixed non-adaptive filter using the optimal LMS algorithm. 
    This naive approach is very slow, hence this should probably only be used 
    for benchmarking purposes.
    Args:
        x: Signal of noise corrolated with the noise corrupting s[n]
        d: Signal s[n] but corrupted by noise
        K: Number of past samples
    Returns:
        a single value s'[n] approximating s[n]
    """
    Rx, rdx = sigutil.autocorrelate(x, d, K)
    f = np.linalg.solve(Rx, rdx)
    return f, d - np.convolve(x, f)[0:len(d)]

def fixed_iterative_lms(x, d, K, N_it):
    """
    Computes a fixed non-adaptive filter using the iterative LMS algorithm.
    This approach is faster than the optimal one, but might be less accurate.
    """
    Rx, rdx = sigutil.autocorrelate(x, d, K)
    eigenvalues, _ = np.linalg.eig(Rx)
    lambda_max = np.max(eigenvalues)
    lambda_min = np.min(eigenvalues)

    # Choose step size mu such that 0 < mu < 2/lambda_max
    mu = 2/(lambda_max + lambda_min)

    if mu >= 2/lambda_max:
        raise Exception("Step size cannot be bigger than 2/lambda_max!")
    
    f_it = np.zeros(K)

    for _ in range(N_it):
        f_it = f_it + mu * (rdx - Rx @ f_it)

    return f_it, d - np.convolve(x, f_it)[0:len(d)]

def adaptive_iterative_lms(x, d, K, N_it, algoType='LMS', mu=None, lambda_=None, delta=None, callback=None):
    """
    Computes an adaptive filter using the algoType algorithm. algoType is a string that allows
    the selection between three algorithms:
    - LMS: Standard LMS
    - NLMS: Normalized LMS (a type of nonlinear LMS)
    - RLS: Recursive LS (another type of nonlinear LMS)
    This takes an optional callback that can be used to plot things during the computation.
    """
    f_ad = np.zeros(K)
    e_ad = np.zeros(len(d))

    # Adaptive filtering
    for i in range(K, len(e_ad)):
        X = x[i-K:i]

        # Solve normal equation using algoType method
        if(algoType == 'LMS'):
            for _ in range(N_it):
                f_ad = f_ad + mu * X * (d[i] - f_ad.T @ X)

        elif(algoType == 'NLMS'):
            eps = 0.05

            for _ in range(N_it):
                normalizationFactor = mu/(eps + np.dot(X, X))
                f_ad = f_ad + normalizationFactor * X * (d[i] - f_ad.T @ X)

        elif(algoType == 'RLS'):
            omega = 1/delta * np.identity(K)
            
            for _ in range(N_it):
                z = omega @ X
                g = z / (lambda_ + np.dot(X, z))
                f_ad = f_ad + g * (d[i] - f_ad.T @ X)
                omega = 1/lambda_ * (omega - np.dot(g.reshape(-1, 1), z.reshape(1, -1)))

        else:
            raise ValueError('Invalid algo type ! Algo type must be either LMS or NLMS or RLS!')

        # Update output
        e_ad[i] = d[i] - f_ad.T @ X

        # Plot the iterative filter at some time instants (every 7 seconds starting from 16 seconds)
        if callback:
            callback(i, f_ad[::-1])

    f_ad = f_ad[::-1]

    return f_ad, e_ad