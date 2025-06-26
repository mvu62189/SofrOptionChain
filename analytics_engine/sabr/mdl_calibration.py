# calibration.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from typing import Tuple, Dict
from sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast, sabr_vol_normal, sabr_vol_lognormal
from sabr_v2 import b76_vega
import json, os


#
# ── Historical β Calibration Utilities ──

GLOBAL_PARAMS_PATH = "analytics_results/model_params/sabr/global_params.json"

def load_global_beta() -> float:
    """Load last‐saved global beta (default 0)."""
    if os.path.exists(GLOBAL_PARAMS_PATH):
        try:
            data = json.load(open(GLOBAL_PARAMS_PATH))
            return float(data.get("beta", 0))
        except:
            return 1
    return 1

def save_global_beta(beta: float):
    """Persist global beta to disk."""
    with open(GLOBAL_PARAMS_PATH, "w") as f:
        json.dump({"beta": beta}, f)

def load_and_prepare_for_beta(df: pd.DataFrame):
    """
    Given a raw snapshot DataFrame, apply the same OTM/liquidity/IV logic
    as in process_snapshot_file up through producing:
      strikes_fit, vols_fit, F, T
    """
    import numpy as np
    from datetime import datetime
    from iv_utils import implied_vol

    # 1) Extract strike and mid price, filter liquid quotes
    df = df.copy()
    df['strike'] = df['ticker'].str.extract(r'\b(\d+\.\d+)\b')[0].astype(float)
    df['mid_price'] = np.where(
        (df['bid']>0) & (df['ask']>0),
        0.5*(df['bid'] + df['ask']),
        np.nan
    )
    df = df[(df['mid_price'] > 0)].reset_index(drop=True)
    if df.empty:
        return np.array([]), np.array([]), None, None

    # 2) Forward & time-to-expiry
    F = float(df['future_px'].iloc[0])
    snap_ts = datetime.strptime(df['snapshot_ts'].iloc[0], '%Y%m%d %H%M%S')
    expiry   = pd.to_datetime(df['expiry_date'].iloc[0]).date()
    T = (expiry - snap_ts.date()).days/365.0

    # 3) OTM selection
    df['type'] = df['type'].str.upper()
    df_otm = df[(
        (df['type']=='C') & (df['strike'] > F)
    ) | (
        (df['type']=='P') & (df['strike'] < F)
    )].reset_index(drop=True)
    if df_otm.empty:
        return np.array([]), np.array([]), F, T

    # 4) Implied vol, drop unreasonables
    df_otm['iv'] = df_otm.apply(
        lambda r: implied_vol(
            F=float(r['future_px']), T=T,
            K=r['strike'], price=r['mid_price'],
            opt_type=r['type'], 
            # engine='bachelier'
            engine='black76'
        ) if not np.isnan(r['mid_price']) else np.nan,
        axis=1
    )
    df_otm = df_otm[(df_otm['iv'] < 100) & (df_otm['volume'] > 0)]
    if df_otm.empty:
        return np.array([]), np.array([]), F, T

    # 5) Strike trimming & sorting
    lo, hi = df_otm['strike'].min(), df_otm['strike'].max()
    df_otm = df_otm[
        (df_otm['strike'] >= lo - 0.5)
        & (df_otm['strike'] <= hi + 0.5)
    ].sort_values('strike').reset_index(drop=True)

    # 6) Final arrays for fitting
    strikes = df_otm['strike'].values
    ivs     = df_otm['iv'].values
    mask    = ~np.isnan(ivs)
    order   = np.argsort(strikes[mask])
    strikes_fit = strikes[mask][order]
    vols_fit    = ivs[mask][order]

    return strikes_fit, vols_fit, F, T


def calibrate_global_beta(file_list: list[str]) -> float:
    """
    Find β ∈ [0,1] that minimizes total SSE across the given snapshots.
    Saves and returns the optimal β.
    """
    from scipy.optimize import minimize_scalar
    import pandas as _pd

    # 1) Prepare data tuples (strikes_fit, vols_fit, F, T)
    markets = []
    for path in file_list:
        df = _pd.read_parquet(path)
        # replicate your OTM & liquidity filtering to get strikes_fit, vols_fit, F, T
        # (you can copy the same logic from process_snapshot_file)
        strikes_fit, vols_fit, F, T = load_and_prepare_for_beta(df)
        markets.append((strikes_fit, vols_fit, F, T))

    def total_sse(beta):
        sse = 0.0
        for K, vols, F, T in markets:
            # 1) calibrate (α,ρ,ν) holding this beta fixed
            alpha_b, beta_b, rho_b, nu_b = calibrate_sabr_fast(
                K, vols, F, T,
                init_params=(0.66745, beta, 0.79241, 2.46749)
            )
            # 2) compute model IV at each strike
            iv_pred = sabr_vol_lognormal(F, K, T, alpha_b, beta_b, rho_b, nu_b)
            # 3) accumulate squared‐error
            sse += ((iv_pred - vols) ** 2).sum()
        return sse

    res = minimize_scalar(
        total_sse,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol":1e-3}
    )
    beta_opt = float(res.x)
    save_global_beta(beta_opt)
    return beta_opt


def fit_sabr_original(strikes: np.ndarray, F: float, T: float,
             vols: np.ndarray, method: str = 'fast',
             manual_params: Dict[str, float] = None) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Calibrate SABR to `vols` at strikes, return params and model IV curve.
    If `manual_params` is provided, overwrite the calibrated params.
    """

    # → choose initial beta: manual override or historical global
    if manual_params is not None:
        init_beta = 1.0
    else:
        init_beta = load_global_beta()

    # build a 4-tuple for the SABR routines
    if manual_params is not None:
        init_seq = (
            float(manual_params['alpha']),
            init_beta,
            float(manual_params['rho']),
            float(manual_params['nu'])
        )
    else:
        # defaults from last known calibration
        init_seq = (0.66745, init_beta, 0.79241, 2.46749)

    # choose calibration engine
    if method == 'fast':
        # pass a tuple (alpha0,beta0,rho0,nu0)
        alpha, beta, rho, nu = calibrate_sabr_fast(strikes, vols, F, T, init_seq)
    else:
        alpha, beta, rho, nu = calibrate_sabr_full(strikes, vols, F, T)
    # override with manual inputs if given
    if manual_params:
        alpha = manual_params.get('alpha', alpha)
        beta  = manual_params.get('beta', beta)
        rho   = manual_params.get('rho', rho)
        nu    = manual_params.get('nu', nu)
    params: Dict[str,float] = {'alpha': alpha, 'beta': beta, 'rho': rho, 'nu': nu}
    iv_model = sabr_vol_lognormal(F, strikes, T, alpha, beta, rho, nu)
    return params, iv_model


def fit_sabr_atm_alpha(strikes: np.ndarray, F: float, T: float,
             vols: np.ndarray, method: str = 'fast',
             manual_params: Dict[str, float] = None) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Calibrate SABR to `vols` at strikes, return params and model IV curve.
    This version is modified to force a perfect fit at the ATM strike.
    """
    # For this targeted approach, we will not use the manual parameter overrides
    # and will focus on the ATM-first calibration.
    
    # 1. Determine fixed parameters (beta)
    # We continue to use the globally calibrated beta for stability.
    beta = load_global_beta()

    # 2. Find the true ATM volatility from the market data
    # Find the index of the strike closest to the forward price
    try:
        atm_strike_idx = np.abs(strikes - F).argmin()
        atm_market_vol = vols[atm_strike_idx]
    except (ValueError, IndexError):
        # Fallback if there are no strikes: calibrate using the old method
        return fit_sabr_original(strikes, F, T, vols, method, manual_params)

    # 3. Solve for Alpha to perfectly match the ATM vol
    # Using the simplified SABR ATM formula: IV_atm ≈ alpha / F^(1-beta)
    # This gives us the alpha that anchors our model at the ATM point.
    alpha = atm_market_vol * (F ** (1 - beta))

    # 4. Calibrate Rho and Nu to fit the rest of the smile
    # Define an objective function to find the best rho and nu, given our fixed alpha and beta.
    def objective(params):
        rho, nu = params
        # We pass the fixed alpha and beta, and the test rho and nu
        model_vols = sabr_vol_lognormal(F, strikes, T, alpha, beta, rho, nu)
        # Calculate the error against the market
        error = np.sum((model_vols - vols)**2)
        return error

    # Initial guesses and bounds for rho and nu
    initial_guess = [0.0, 1.0]  # Start with neutral rho, moderate nu
    bounds = [(-0.999, 0.999), (1e-4, 5.0)] # Physical bounds for rho and nu

    # Run the optimization for rho and nu
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    rho, nu = result.x

    # 5. Return the final parameters and the resulting model smile
    params: Dict[str,float] = {'alpha': alpha, 'beta': beta, 'rho': rho, 'nu': nu}
    iv_model = sabr_vol_lognormal(F, strikes, T, alpha, beta, rho, nu)
    
    return params, iv_model

def fit_sabr(strikes: np.ndarray, F: float, T: float,
             vols: np.ndarray, method: str = 'fast', # method is no longer used but kept for interface consistency
             manual_params: Dict[str, float] = None) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Calibrate SABR using an interpolated, Vega-weighted objective function.

    This method first creates a smooth cubic spline from the raw market vols,
    then calibrates SABR parameters by minimizing the Vega-squared weighted
    error on a fine grid of strikes.
    """

    # --- Manual Override Path ---
    if manual_params:
        # If manual params are provided, skip calibration and just calculate the curve.
        alpha = manual_params.get('alpha', 0.1)
        # Use the provided beta, or fall back to the global one if not provided.
        beta = manual_params.get('beta', load_global_beta()) 
        rho = manual_params.get('rho', 0.0)
        nu = manual_params.get('nu', 1.0)
        
        params = {'alpha': alpha, 'beta': beta, 'rho': rho, 'nu': nu}
        
        # Calculate the IV curve on the provided `strikes` grid (which is the `strikes_fit` grid)
        iv_model_on_fit_strikes = sabr_vol_lognormal(F, strikes, T, alpha, beta, rho, nu)
        
        # Return empty debug data as no interpolation happens in this path.
        debug_data = {"interp_strikes": [], "interp_vols": []} 
        
        return params, iv_model_on_fit_strikes, debug_data

    # --- Automatic Calibration Path ---
    if len(strikes) < 4:
        # Not enough points to create a reliable spline, return empty
        return {}, np.array([])

    # --- 1. Interpolation Step ---
    # Create a smooth, continuous function (cubic spline) from the discrete market points.
    vol_spline = CubicSpline(strikes, vols, bc_type='natural')

    # Create a new, fine-grained grid of strikes for calibration.
    # Using more points makes the calibration more robust.
    fine_strikes = np.arange(strikes.min(), strikes.max(), 0.0625)
    # Include endpoints
    if fine_strikes[-1] < strikes.max():
        fine_strikes = np.append(fine_strikes, strikes.max())
    # Get the interpolated market volatilities on this new fine grid.
    market_vols_interp = vol_spline(fine_strikes)


    # --- 2. Vega-Weighted Optimization Step ---
    # Load the historically calibrated beta for stability.
    beta = load_global_beta()

    # Calculate Vega weights. We use the interpolated market vols for this calculation,
    # as this provides a stable set of weights throughout the optimization.
    # Squaring the vega is a common practice for variance-based weighting.
    vega_weights = b76_vega(F, T, fine_strikes, market_vols_interp) ** 2
    
    # Normalize weights to prevent the objective function value from becoming too large
    if np.sum(vega_weights) > 0:
        vega_weights /= np.sum(vega_weights)
    else:
        # If all vegas are zero, use equal weighting
        vega_weights = np.ones_like(fine_strikes)

    # Define the objective function to be minimized
    def objective(params):
        alpha, rho, nu = params
        # Calculate SABR vols on our fine grid
        model_vols = sabr_vol_lognormal(F, fine_strikes, T, alpha, beta, rho, nu)
        
        # Calculate the weighted squared error
        error = np.sum(vega_weights * (model_vols - market_vols_interp)**2)
        return error

    # Provide initial guesses and bounds for the optimizer
    initial_guess = [0.01, 0.0, 1.0]  # alpha, rho, nu
    bounds = [(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)]

    # Run the global optimization
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    alpha, rho, nu = result.x

    # --- 3. Return Final Results ---
    # Final parameters
    params: Dict[str, float] = {'alpha': alpha, 'beta': beta, 'rho': rho, 'nu': nu}
    
    # Generate the final model IV curve on the original market strikes for comparison
    iv_model_on_market_strikes = sabr_vol_lognormal(F, strikes, T, alpha, beta, rho, nu)

    debug_data = {
    "interp_strikes": fine_strikes,
    "interp_vols": market_vols_interp
    }
    return params, iv_model_on_market_strikes, debug_data


