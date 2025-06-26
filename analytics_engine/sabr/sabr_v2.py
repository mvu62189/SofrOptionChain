# sabr_v2.py

import numpy as np
from scipy.optimize import minimize, differential_evolution
from bachelier import bachelier_vega
from black76 import b76_vega

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


def sabr_vol_lognormal(F, K, T, alpha, beta, rho, nu):
    """
    Hagan’s 2002 lognormal‐SABR approximation (β≠0).
    Returns Black-76 implied vol for each strike K.
    """
    F = np.asanyarray(F)
    K = np.asanyarray(K)

    # ATM case
    atm = np.isclose(F, K, rtol=1e-8, atol=1e-8)

    iv = np.zeros_like(F)

    if np.any(atm):
        iv[atm] = (alpha / (F[atm]**(1 - beta))) * (
            1
            + (
                ((1 - beta)**2 / 24) * (alpha**2) / (F[atm] ** (2 - 2*beta))
                + (rho * beta * nu * alpha) / (4 * F[atm]**(1 - beta))
                + (2 - 3 * rho**2) * nu**2 / 24
            ) * T
        )


    # For non-ATM strikes, these are arrays. For ATM, some values will be NaN/inf due to log(1)=0.
    logFK = np.log(F / K)
    FKbeta = (F * K)**((1 - beta) / 2)
    z = (nu / alpha) * FKbeta * logFK

    # x(z) calculation
    sqrt_term = np.sqrt(1 - 2 * rho * z + z**2)
    x_of_z = np.log((sqrt_term + z - rho) / (1 - rho))

    # The core term is z / x(z). The limit of this as z -> 0 is 1.
    # We use np.where to handle the ATM case where z is zero, avoiding division by zero.
    z_over_x = np.where(atm, 1.0, z / x_of_z)
    
    # First part of the formula
    A = alpha / (FKbeta * (1 + ((1 - beta)**2 / 24) * (logFK**2) + ((1 - beta)**4 / 1920) * (logFK**4)))
    
    # Time-dependent part of the formula
    B = 1 + (
        ((1 - beta)**2 * alpha**2) / (24 * FKbeta**2) +
        (rho * beta * nu * alpha) / (4 * FKbeta) +
        ((2 - 3 * rho**2) * nu**2) / 24
    ) * T

    # Combine terms for the final lognormal volatility.
    # For the ATM case (where atm_mask is True), z_over_x is 1, and the formula correctly
    # collapses to the ATM approximation because A also simplifies correctly.
    iv = A * z_over_x * B
    
    return iv

def calibrate_sabr_full_normal(strikes, market_vols, F, T):
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
        
        vegas = np.array([b76_vega(F, K, T, sigma) for K, sigma in zip(strikes, market_vols)])
        # vegas = np.array([bachelier_vega(F, K, T, sigma) for K, sigma in zip(strikes, market_vols)])
        w = vegas * vegas
        sq_errs = (model_vols - market_vols)**2
        return np.sum(sq_errs * w)

    de = differential_evolution(objective, bounds, seed=42, popsize=40, maxiter=300, polish=True)
    
    # Reconstruct full parameter array with fixed beta
    alpha, rho, nu = de.x
    beta = 1
    return np.array([alpha, beta, rho, nu])


def calibrate_sabr_fast_normal(strikes, market_vols, F, T, init_params):
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
    beta = 1
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
    
    # vega = bachelier_vega(F, strikes, T, market_vols)
    vega = b76_vega(F, strikes, T, market_vols)
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
    beta = 1
    return np.array([alpha, beta, rho, nu])


def calibrate_sabr_full(strikes, market_vols, F, T):
    """
    Calibrate all four SABR params (α,β,ρ,ν) to Black-76 IVs.
    """
    bounds = [
        (1e-4, 5.0),      # α
        (0.0, 1.0),       # β
        (-0.999, 0.999),  # ρ
        (1e-4, 5.0)       # ν
    ]
    def obj(p):
        a, b, r, n = p
        ivm = sabr_vol_lognormal(F, strikes, T, a, b, r, n)
        return np.sum((ivm - market_vols)**2)
    res = differential_evolution(obj, bounds, seed=42, maxiter=200)
    return res.x  # [α,β,ρ,ν]

def calibrate_sabr_fast(strikes, market_vols, F, T, init_params):
    """
    Warm‐start lognormal SABR, optimizing α,ρ,ν with β fixed.
    """
    a0, b0, r0, n0 = init_params
    x0 = [a0, r0, n0]
    bounds = [(1e-4,5.0), (-0.999,0.999), (1e-4,5.0)]
    def obj(v):
        a, r, n = v
        ivm = sabr_vol_lognormal(F, strikes, T, a, b0, r, n)
        return np.mean((ivm - market_vols)**2)
    res = minimize(obj, x0, bounds=bounds, method='L-BFGS-B')
    return np.array([res.x[0], b0, res.x[1], res.x[2]])