# analytics_engine/sabr/rnd_from_sabr.py
import numpy as np
from analytics_engine.sabr.sabr_v2 import sabr_vol_normal
from analytics_engine.sabr.bachelier import bachelier_price

def price_from_sabr(strikes, F, T, alpha, beta, rho, nu):
    """Price European options using SABR-fitted vols under Bachelier model."""
    prices = []
    for K in strikes:
        sigma = sabr_vol_normal(F, K, T, alpha, beta, rho, nu)
        p = bachelier_price(F, K, T, sigma)
        prices.append(p)
    return np.array(prices)

def second_derivative(f, x, h=1e-2):
    """Simple central difference 2nd derivative."""
    return (f(x + h) - 2*f(x) + f(x - h)) / (h ** 2)

def compute_rnd(strikes, F, T, alpha, beta, rho, nu):
    """Apply Breeden-Litzenberger on SABR-based prices."""
    f = lambda K: bachelier_price(F, K, T, sabr_vol_normal(F, K, T, alpha, beta, rho, nu))
    pdf = [max(0, second_derivative(f, K)) for K in strikes]
    return np.array(pdf)
