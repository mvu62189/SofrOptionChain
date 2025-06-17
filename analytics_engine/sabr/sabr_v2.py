# sabr_v2.py

import sys, time, json
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import differential_evolution, minimize, brentq
from analytics_engine.sabr.bachelier import bachelier_vega

def sabr_vol_normal(F, K, T, alpha, beta, rho, nu):
    """Hagans normal-SABR implied vol."""
    if abs(F - K) < 1e-12:
        term2 = 1 + (
            ((2 - 3*rho*rho)*nu*nu)/24
            + (beta*(beta-1)*alpha*alpha)/(24*F**(2-2*beta))
            + (rho*beta*nu*alpha)/(4*F**(1-beta))
        ) * T
        return alpha * F**(beta-1) * term2
    FKb = (F*K)**((1-beta)/2)
    z   = (nu/alpha)*FKb*(F - K)
    x_z = 1 - rho*z/2 if abs(z) < 1e-6 else np.log((np.sqrt(1 - 2*rho*z + z*z) + z - rho)/(1 - rho))
    factor = alpha * FKb
    corr   = 1 + (
        ((2 - 3*rho*rho)*nu*nu)/24
        + (beta*(beta-1)*alpha*alpha)/(24*FKb*FKb)
        + (rho*beta*nu*alpha)/(4*FKb)
    ) * T
    return factor * (z/x_z) * corr

# ─── A) Full‐grid DE + polish (run once) ───────────────────────────────────
def calibrate_sabr_full(strikes: np.ndarray,
                        market_vols: np.ndarray,
                        F: float,
                        T: float) -> np.ndarray:
    bounds = [
        (1e-4, 5.0),  # alpha
        (0.0, 1.0),   # beta
        (-0.999,0.999),# rho
        (1e-4, 5.0)   # nu
    ]

    def objective(params):
        alpha, beta, rho, nu = params
        model_vols = np.array([
            sabr_vol_normal(F, K, T, alpha, beta, rho, nu) for K in strikes
        ])
        vegas = np.array([bachelier_vega(F, K, T, sigma) for K, sigma in zip(strikes, market_vols)])
        # weight by vega
        sq_errs = (model_vols - market_vols)**2
        # normalize by total weight
        return np.sum(sq_errs * vegas) / np.sum(vegas)
    # DE
    de = differential_evolution(
        objective, bounds, seed=42, popsize=20, maxiter=200, polish=False)
    # polish
    res = minimize(objective, x0=de.x, bounds=bounds, method='L-BFGS-B',
                   options={'ftol':1e-12,'maxiter':200})
    return res.x if res.success else de.x


# ─── B) Fast warm‐start recalibration ───────────────────────────────────────
def calibrate_sabr_fast(strikes: np.ndarray,
                        market_vols: np.ndarray,
                        F: float,
                        T: float,
                        init_params: np.ndarray) -> np.ndarray:
    """Warm‐start L-BFGS calibration from init_params."""
    bounds = [(1e-4,5.0),(0.0,1.0),(-0.999,0.999),(1e-4,5.0)]
    def objective(params):
        alpha, beta, rho, nu = params
        model_vols = np.array([
            sabr_vol_normal(F, K, T, alpha, beta, rho, nu) for K in strikes
        ])
        # vega weights
        vegas = np.array([
            bachelier_vega(F, K, T, mkt_iv)
            for K, mkt_iv in zip(strikes, market_vols)
        ])
        # MSE/SSE
        sq_err = (model_vols - market_vols)**2
        total_wt = np.sum(vegas)
        if total_wt <= 0:
            return np.mean(sq_err)
        return np.sum(sq_err * vegas) / total_wt
    res = minimize(objective, x0=init_params, bounds=bounds,
                   method='L-BFGS-B', options={'ftol':1e-14,'maxiter':100})
    return res.x if res.success else init_params
