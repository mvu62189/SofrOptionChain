# bachelier.py

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bachelier_iv(F, T, K, ask_price):
    def price(sigma):
        d = (F - K)/(sigma*np.sqrt(T))
        return sigma*np.sqrt(T)*norm.pdf(d) + (F-K)*norm.cdf(d)
    intrinsic = max(0.0, F - K)
    p = max(ask_price, intrinsic + 1e-8)
    for a,b in [(1e-8,0.5),(0.5,2.0),(2.0,10.0)]:
        if (price(a)-p)*(price(b)-p) < 0:
            return brentq(lambda x: price(x)-p, a, b, maxiter=200)
    return np.nan

def bachelier_price(F, K, T, sigma):
    d = (F - K)/(sigma*np.sqrt(T))
    return sigma*np.sqrt(T)*norm.pdf(d) + (F-K)*norm.cdf(d)

def bachelier_vega(F, K, T, sigma):
    d = (F - K)/(sigma*np.sqrt(T))
    return np.sqrt(T)*norm.pdf(d)