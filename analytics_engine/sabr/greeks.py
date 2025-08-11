# analytics_engine/sabr/greeks.py

import numpy as np
from scipy.stats import norm

def calculate_greeks(F, K, T, sigma, opt_type='C'):
    """
    Calculates all relevant Black-76 Greeks in a vectorized manner.
    Assumes r=0 for futures options.

    Returns a dictionary of Greek values.
    """
    # Ensure inputs are numpy arrays for vectorization
    K = np.asanyarray(K)
    sigma = np.asanyarray(sigma)
    #T = np.asanyarray(T)

    # Handle cases where T or sigma are zero or negative to avoid math errors
    # We create a mask of valid conditions for calculation
    valid_mask = (T > 1e-9) & (sigma > 1e-9) & (K > 0)

    # Initialize result arrays with zeros
    delta = np.zeros_like(K, dtype=float)
    gamma = np.zeros_like(K, dtype=float)
    vega = np.zeros_like(K, dtype=float)
    theta = np.zeros_like(K, dtype=float)
    vanna = np.zeros_like(K, dtype=float)
    charm = np.zeros_like(K, dtype=float)

    # Perform calculations only on the valid subset
    if np.any(valid_mask):
        # Select only the valid elements for the calculation
        F_v = F
        K_v = K[valid_mask]
        
        sigma_v = sigma[valid_mask]

        T_v = T

        # Calculate d1 and d2 for the valid subset
        sqrt_T = np.sqrt(T_v)
        d1 = (np.log(F_v / K_v) + 0.5 * sigma_v**2 * T_v) / (sigma_v * sqrt_T)
        d2 = d1 - sigma_v * sqrt_T
        
        # Standard normal probability density function
        phi_d1 = norm.pdf(d1)

        # --- Calculate Greeks for the valid subset ---
        
        # Vega (per 1% vol change, so we divide by 100)
        vega_val = (F_v * phi_d1 * sqrt_T) / 100.0

        # Gamma
        gamma_val = phi_d1 / (F_v * sigma_v * sqrt_T)

        # Theta (per day, so we divide by 365)
        theta_val = (- (F_v * phi_d1 * sigma_v) / (2 * sqrt_T)) / 365.0
        
        # Vanna (dVega/dSpot)
        vanna_val = (vega_val / F_v) * (1 - d1 / (sigma_v * sqrt_T))

        # Charm calculation needs to handle T_v potentially being zero
        if T_v > 1e-9:
            charm_val = (phi_d1 * (d2 * sigma_v * sqrt_T)) / (2 * T_v * sigma_v * sqrt_T)
            charm_val /= 365.0
        else:
            charm_val = np.zeros_like(d1)


        # Delta depends on option type
        if opt_type.upper() == 'C':
            delta_val = norm.cdf(d1)
        else: # Put
            delta_val = norm.cdf(d1) - 1
            # Adjust charm for puts
            if T_v > 1e-9:
                charm_val = (-phi_d1 * (d2 * sigma_v * sqrt_T)) / (2 * T_v * sigma_v * sqrt_T)
                charm_val /= 365.0

        # Place the calculated values back into the full-sized arrays
        np.place(delta, valid_mask, delta_val)
        np.place(gamma, valid_mask, gamma_val)
        np.place(vega, valid_mask, vega_val)
        np.place(theta, valid_mask, theta_val)
        np.place(vanna, valid_mask, vanna_val)
        np.place(charm, valid_mask, charm_val)

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'vanna': vanna,
        'charm': charm
    }
