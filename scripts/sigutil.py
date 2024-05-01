import numpy as np

def autocorrelate(x, k):
    """
    Calculates x[n] conv x[n-k]
    x is centered to elimiate the 0-frequency component
    """
    #return np.convolve(np.mean(x[:end-Nf), x(i:end-k-1-Nf)));

def autocorrelate(x):
    """
    Calculates the correlation matrix of size (k+1)*(k+1) of the signal
    Arguments:
        x: x[n-k]..x[n]
    """
    k = len(x) - 1

    R_Xn = np.zeros(k+1)

    for i in range(k+1):
        R_Xn[i]= np.mean(...)
    ...                
    
def correlate(x, y):
    ...