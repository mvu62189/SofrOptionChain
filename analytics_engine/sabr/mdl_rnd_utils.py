# rnd_utils.py
import numpy as np
from typing import Dict
from sabr_rnd import compute_rnd, second_derivative, price_from_sabr
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter

def market_rnd_old(strikes: np.ndarray, mid_prices: np.ndarray) -> np.ndarray:    
    """
    Market RND via smoothing spline on price:
      1) drop zero‐price strikes (illiquid)
      2) fit a smoothing spline to (K, C(K))
      3) take its second derivative → raw density
      4) clamp, normalize
      5) interpolate back to original strikes
    """
    from scipy.interpolate import interp1d, UnivariateSpline

    # 1) filter out illiquid strikes
    mask = mid_prices > 0
    ks = strikes[mask]
    ps = mid_prices[mask]
    if ks.size < 4:
        return np.full_like(strikes, np.nan)

    # 2) smoothing spline on price: control roughness via 's'
    s = len(ks) * np.var(ps) * 0.01
    price_spl = UnivariateSpline(ks, ps, k=3, s=s)

    # 3) second derivative at each (liquid) strike
    rnd_raw = price_spl.derivative(n=2)(ks)

    # 4) clamp negatives, normalize
    rnd_clamped = np.clip(rnd_raw, a_min=0.0, a_max=None)
    area = np.trapezoid(rnd_clamped, ks)
    if area > 0:
        rnd_clamped /= area

    # 5) map back onto full strikes
    rnd_interp = interp1d(
        ks, rnd_clamped,
        kind='cubic', fill_value='extrapolate', bounds_error=False
    )
    result = rnd_interp(strikes)
    # hide zero densities
    result[result == 0.0] = np.nan
    return result

def market_rnd(strikes: np.ndarray, mid_prices: np.ndarray) -> np.ndarray:
    """
    Smooth Breeden–Litzenberger RND:
      1) drop zero‐price strikes (illiquid)
      2) cubic‐interp on remaining strikes
      3) compute RND on a fine grid
      4) normalize
      5) interpolate back onto original strikes
    """
    


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
    h = ks_fine[1] - ks_fine[0]
    rnd_fine = np.array([
        max(0.0, second_derivative(price_func, K, h=h))
        for K in ks_fine
    ])

    # ── 4a) Smooth the raw density with a Gaussian filter ──
    from scipy.ndimage import gaussian_filter1d
    # 2-stage smoothing: light global + heavier low-strike
    rnd_fine = gaussian_filter1d(rnd_fine, sigma=3.0, mode='constant')
    # extra smooth on the “low‐strike” portion
    #n_low = int(len(ks_fine) * 0.20)  # first 20% of strikes
    #if n_low > 3:
    #    rnd_fine[:n_low] = gaussian_filter1d(rnd_fine[:n_low], sigma=0.1, mode='constant')
    
    spline = UnivariateSpline(ks_fine, rnd_fine, k=3, s=len(ks_fine)*np.var(rnd_fine)*0.3)
    rnd_smooth = spline(ks_fine)

    w = min(len(ks_fine) // 2 * 2 + 1, 51)  # e.g. up to 51 or grid-size
    rnd_sg = savgol_filter(rnd_fine, window_length=w, polyorder=3, mode='interp')
    # rnd_sg = np.clip(rnd_sg, 0.0, None)

    # clamp any tiny negatives back to zero
    # rnd_fine = np.clip(rnd_fine, a_min=0.0, a_max=None)

    # ── 4b) Normalize to integrate to 1 ──
    area = np.trapezoid(rnd_sg, ks_fine)
    if area > 0:
        rnd_sg /= area
    
    # --- 5) Re‐sample density back to original strikes & clamp ---
    rnd_interp = interp1d(
        ks_fine, rnd_sg, kind='cubic',
        fill_value=0.0, bounds_error=False
    )
    rnd_on_strikes = rnd_interp(strikes)

    # clamp negatives (shouldn’t occur) and filter out exact zeros
    # rnd_clipped = np.clip(rnd_on_strikes, a_min=0.0, a_max=None)
    
    # replace zero densities with NaN so they aren’t plotted
    # rnd_clipped[rnd_clipped < 0.1] = np.nan
    return rnd_on_strikes
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