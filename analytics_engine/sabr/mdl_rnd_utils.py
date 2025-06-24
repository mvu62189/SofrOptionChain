# rnd_utils.py
import numpy as np
from typing import Dict
from sabr_rnd import compute_rnd, second_derivative, price_from_sabr
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter

'''
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
'''

def market_rnd(strikes: np.ndarray, mid_prices: np.ndarray) -> np.ndarray:
    """
    Smooth Breeden–Litzenberger RND:
      1) drop zero‐price strikes (illiquid)
      2) cubic‐interp on remaining strikes
      3) compute RND on a fine grid
      4) normalize
      5) interpolate back onto original strikes
    """
    from scipy.interpolate import interp1d
    from sabr_rnd import second_derivative

    # --- 1) Keep only liquid strikes (mid_price>0) ---
    mask = mid_prices > 0
    ks = strikes[mask]
    ps = mid_prices[mask]
    if ks.size < 3:
        # too few points to form a density → return zeros
        return np.zeros_like(strikes)

    # --- 2) Build a cubic interpolator ---
    price_func = interp1d(
        ks, ps, kind='cubic',
        fill_value='extrapolate', bounds_error=False
    )

    # --- 3) Evaluate on a fine uniform grid ---
    n_grid = max(100, ks.size * 5)
    ks_fine = np.linspace(ks.min(), ks.max(), n_grid)
    ps_fine = price_func(ks_fine)

    # Finite‐difference second derivative on the fine grid
    
    # h = ks_fine[1] - ks_fine[0]
    dks = np.diff(ks)
    h = float(np.min(dks)) if dks.size else 1e-4
    
    rnd_fine = np.array([
        max(0.0, second_derivative(price_func, K, h=h))
        for K in ks_fine
    ])

    # --- 4) Normalize to integrate to 1 ---
    area = np.trapz(rnd_fine, ks_fine)
    if area > 0:
        rnd_fine /= area

    # --- 5) Re‐sample density back to original strikes & clamp ---
    rnd_interp = interp1d(
        ks_fine, rnd_fine, kind='cubic',
        fill_value=0.0, bounds_error=False
    )
    rnd_on_strikes = rnd_interp(strikes)

    # clamp negatives (shouldn’t occur) and filter out exact zeros
    rnd_clipped = np.clip(rnd_on_strikes, a_min=0.0, a_max=None)
    
    # replace zero densities with NaN so they aren’t plotted
    # rnd_clipped[rnd_clipped < 0.1] = np.nan
    return rnd_clipped
    # return np.clip(rnd_on_strikes, a_min=0.0, a_max=None)

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