# bachelier.py

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bachelier_iv(F, T, K, price, opt_type='C'):
    # Intrinsic for each
    intrinsic = max(0.0, F-K) if opt_type.upper() == 'C' else max(0.0, K-F)
    p = max(price, intrinsic + 1e-8)
    def bach_price(sigma):
        d = (F - K)/(sigma*np.sqrt(T))
        if opt_type.upper() == 'C':
            return sigma*np.sqrt(T)*norm.pdf(d) + (F-K)*norm.cdf(d)
        else:  # Put
            return sigma*np.sqrt(T)*norm.pdf(-d) + (K-F)*norm.cdf(-d)
    for a,b in [(1e-8,0.5),(0.5,2.0),(2.0,10.0)]:
        if (bach_price(a)-p)*(bach_price(b)-p) < 0:
            return brentq(lambda x: bach_price(x)-p, a, b, maxiter=200)
    return np.nan


def bachelier_price(F, K, T, sigma):                                        ### REVISIT ### ## FEATURE ## #Need vectorized
    d = (F - K)/(sigma*np.sqrt(T))
    return sigma*np.sqrt(T)*norm.pdf(d) + (F-K)*norm.cdf(d)

def bachelier_vega(F, K, T, sigma):                                         ### REVISIT ### ## FEATURE ## #Need vectorized
    d = (F - K)/(sigma*np.sqrt(T))
    return np.sqrt(T)*norm.pdf(d)