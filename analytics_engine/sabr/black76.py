import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def b76_price(F: float, K, T: float, sigma, opt_type: str = 'C') -> np.ndarray:
    """
    Vectorized Black-76 forward-price of a European option.
    Handles both scalar and array inputs for K and sigma.
    """
    # Ensure inputs that can be arrays are treated as such for robust calculations
    K = np.asanyarray(K)
    sigma = np.asanyarray(sigma)

    # This handles all degenerate cases.
    if opt_type.upper() == 'C':
        prices = np.maximum(F - K, 0.0)
    else:
        prices = np.maximum(K - F, 0.0)

    # Identify the subset of options where the Black-76 formula is valid
    valid_mask = (sigma > 1e-9) & (T > 0) & (K > 0)

    # Only perform calculations on the valid subset to avoid math errors (e.g., division by zero)
    if np.any(valid_mask):
        # Select only the valid elements for the calculation
        K_v = K[valid_mask]
        sigma_v = sigma[valid_mask]

        d1 = (np.log(F / K_v) + 0.5 * sigma_v**2 * T) / (sigma_v * np.sqrt(T))
        d2 = d1 - sigma_v * np.sqrt(T)

        if opt_type.upper() == 'C':
            formula_prices = F * norm.cdf(d1) - K_v * norm.cdf(d2)
        else:
            formula_prices = K_v * norm.cdf(-d2) - F * norm.cdf(-d1)
        
        # Use np.where to combine results: where the mask is True, use the formula price,
        # otherwise, keep the default intrinsic value.
        prices = np.where(valid_mask, formula_prices, prices)
    
    return prices

def black76_iv(F: float, T: float, K: float, price: float, opt_type: str = 'C') -> float:
    """
    Implied vol under Black-76 (lognormal) given forward F, expiry T,
    strike K, option mid‐price, and opt_type ('C' or 'P').
    """
    # 1) Floor at intrinsic
    intrinsic = max(F-K, 0.0) if opt_type.upper()=='C' else max(K-F, 0.0)
    p = max(price, intrinsic + 1e-8)

    # 2) Root‐find implied vol in [1e-8, 5.0]
    try:
        return brentq(lambda s: b76_price(F, K, T, s, opt_type=opt_type) - p, 1e-8, 5.0, maxiter=200)
    except ValueError:
        return np.nan

def b76_vega(F: float, T: float, K, sigma):
    """
    Vectorized Black-76 Vega: ∂Price/∂σ = F * φ(d1) * sqrt(T)
    where d1 = [ln(F/K) + 0.5 σ^2 T] / (σ √T).

    This version both scalar and numpy array inputs for K and sigma.
    """
    # First, handle the simple case of scalar inputs.
    if np.isscalar(K):
        if sigma <= 0 or T <= 0 or K <= 0:
            return 0.0
        
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        return F * norm.pdf(d1) * np.sqrt(T)

    # If the inputs are not scalar, proceed with the vectorized logic for arrays.
    K_arr = np.asanyarray(K)
    sigma_arr = np.asanyarray(sigma)

    # Initialize a vega array of the correct shape, filled with zeros.
    vega = np.zeros_like(K_arr, dtype=float)

    # Create a boolean mask to identify elements where the calculation is valid.
    valid_mask = (sigma_arr > 1e-9) & (K_arr > 0) & (T > 0)

    # Perform calculations only on the subset of valid elements.
    if np.any(valid_mask):
        # Select only the valid elements for the calculation
        K_valid = K_arr[valid_mask]
        sigma_valid = sigma_arr[valid_mask]

        # Calculate d1 only for the valid subset
        d1 = (np.log(F / K_valid) + 0.5 * sigma_valid**2 * T) / (sigma_valid * np.sqrt(T))

        # Calculate vega for the valid subset
        vega_valid = F * norm.pdf(d1) * np.sqrt(T)

        # Place the calculated vega values back into the correct positions in the full array
        np.place(vega, valid_mask, vega_valid)

    return vega