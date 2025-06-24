# rnd_utils.py
import numpy as np
from typing import Dict
from sabr_rnd import compute_rnd, second_derivative, price_from_sabr
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter

def market_rnd(strikes, mid_prices):
    # 1) sort strikes & prices
    idx = np.argsort(strikes)
    ks = strikes[idx]
    ps = mid_prices[idx]

    # 2) build a cubic‐spline price function
    price_func = interp1d(
        ks, ps,
        kind='cubic',
        fill_value='extrapolate',
        bounds_error=False
    )

    # 3) finite-difference second derivative at each strike (clamp negatives)
    #    -- determine a sensible h from the smallest strike gap
    dks = np.diff(ks)
    h = float(np.min(dks)) if dks.size else 1e-4
    rnd_vals = np.array([
        max(0.0, second_derivative(price_func, K, h=h))
        for K in ks
    ])

    # 4) normalize to integrate to 1
    area = np.trapezoid(rnd_vals, ks)
    if area > 0:
        rnd_vals = rnd_vals / area
    return rnd_vals

def model_rnd(strikes: np.ndarray, F: float, T: float,
              params: Dict[str, float]) -> np.ndarray:
    """
    Compute SABR-based RND using calibrated params.
    """
    model_prices = price_from_sabr(strikes,F, T,
                                   alpha=params['alpha'],
                                   beta=params['beta'],
                                   rho=params['rho'],
                                   nu=params['nu'])
    rnd = np.gradient(np.gradient(model_prices, strikes), strikes)
    area = np.trapezoid(rnd, strikes)
    return rnd / area