# sabr_v2.py

import numpy as np
from scipy.optimize import minimize, differential_evolution
from bachelier import bachelier_vega

def sabr_vol_normal(F, K, T, alpha, rho, nu):
    """
    Calculates SABR volatility using the correct and stable Hagan 2002 Normal SABR formula (beta=0).
    """
    # Ensure inputs are numpy arrays for vectorization
    F = np.asanyarray(F)
    K = np.asanyarray(K)

    # Handle the ATM case where F and K are very close
    atm_mask = np.isclose(F, K, rtol=1e-5, atol=1e-6)

    # To prevent division by zero or log(0) for invalid alpha or nu
    if alpha <= 1e-6:
        return np.full_like(F, fill_value=alpha) # Return ATM alpha approx if alpha is tiny
    if nu <= 1e-7: # If nu is zero, the model is just flat alpha
        return np.full_like(F, fill_value=alpha)
        
    # --- For non-ATM options ---
    z = (nu / alpha) * (F - K)
    
    # Check for cases where z is close to zero, which should be handled by the ATM formula
    z_mask = np.isclose(z, 0)

    # The core of the formula: x(z) calculation
    # We add a small epsilon to the sqrt term to prevent floating point errors
    sqrt_term = np.sqrt(1 - 2 * rho * z + z**2 + 1e-12)
    
    # The denominator, x(z)
    denominator = np.log((sqrt_term + z - rho) / (1 - rho))

    A = 1.0 + ((2.0 - 3.0 * rho**2) / 24.0) * (nu**2) * T

    # Calculate non-ATM vol, avoiding division by zero.
    non_atm_vol = alpha * A * z / np.where(z_mask | np.isclose(denominator, 0), 1, denominator) # Avoid division by zero
    
    # --- For ATM options ---
    # This is the correct Hagan expansion for Normal SABR vol at K=F
    atm_vol = alpha * (1 + ((2 - 3 * rho**2) / 24) * nu**2 * T)

    # Combine the results: use ATM vol where K is close to F or z is close to 0
    final_vol = np.where(atm_mask | z_mask, atm_vol, non_atm_vol)
    
    return final_vol


def calibrate_sabr_full(strikes, market_vols, F, T):
    """
    Calibrates Normal SABR parameters (alpha, rho, nu) to market vols.
    Beta is fixed to 0.
    """
    # Beta is fixed at 0, so we only optimize 3 parameters
    bounds = [
        (1e-4, 5.0),      # alpha
        (-0.999, 0.999),  # rho
        (1e-4, 5.0)       # nu
    ]
    
    def objective(params):
        alpha, rho, nu = params
        model_vols = sabr_vol_normal(F, strikes, T, alpha, rho, nu)
        
        vegas = np.array([bachelier_vega(F, K, T, sigma) for K, sigma in zip(strikes, market_vols)])
        w = vegas * vegas
        sq_errs = (model_vols - market_vols)**2
        return np.sum(sq_errs * w)

    de = differential_evolution(objective, bounds, seed=42, popsize=40, maxiter=300, polish=True)
    
    # Reconstruct full parameter array with fixed beta
    alpha, rho, nu = de.x
    beta = 0.9
    return np.array([alpha, beta, rho, nu])


def calibrate_sabr_fast(strikes, market_vols, F, T, init_params):
    """
    Warm-start Normal SABR recalibration. Beta is fixed to 0.
    """
    # Extract initial guess for alpha, rho, nu. Ignore beta.
    alpha_init, _, rho_init, nu_init = init_params
    x0 = [alpha_init, rho_init, nu_init]

    bounds = [
        (1e-4, 5.0),      # alpha
        (-0.999, 0.999),  # rho
        (1e-4, 5.0)       # nu
    ]

    def objective(params):
        alpha, rho, nu = params
        model_vols = sabr_vol_normal(F, strikes, T, alpha, rho, nu)
        sq_errs = (model_vols - market_vols)**2
        return np.mean(sq_errs)
        
    res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B', options={'ftol':1e-12,'maxiter':100})
    
    # Reconstruct full parameter array with fixed beta
    alpha, rho, nu = res.x
    beta = 0.9
    return np.array([alpha, beta, rho, nu])


def calibrate_sabr_fast_region_weighted(strikes, market_vols, F, T, init_params, call_weight=1.0, put_weight=2.0):
    """
    Warm-start Normal SABR recalibration with region weights. Beta is fixed to 0.
    """
    alpha_init, _, rho_init, nu_init = init_params
    x0 = [alpha_init, rho_init, nu_init]
    
    bounds = [
        (1e-4, 5.0),      # alpha
        (-0.999, 0.999),  # rho
        (1e-4, 5.0)       # nu
    ]
    
    vega = bachelier_vega(F, strikes, T, market_vols)
    region = np.where(strikes <= F, call_weight, put_weight)
    w = (vega * region)**2
    if w.sum() > 0:
        w /= w.sum()
    else:
        w = np.ones_like(strikes) / len(strikes)

    def objective(params):
        alpha, rho, nu = params
        model_vols = sabr_vol_normal(F, strikes, T, alpha, rho, nu)
        err = model_vols - market_vols
        return np.sum(w * err * err)

    res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B', options={'ftol':1e-14,'maxiter':100})
    
    # Reconstruct full parameter array with fixed beta
    alpha, rho, nu = res.x
    beta = 0.9
    return np.array([alpha, beta, rho, nu])
