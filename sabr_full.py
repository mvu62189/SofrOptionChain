### -----------------------------------------------------------------
# black76.py
### -----------------------------------------------------------------

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def b76_price(F: float, K: float, T: float, sigma: float, opt_type: str = 'C') -> float:
    """
    Black-76 forward‐price of a European option:
      C = F·N(d1) – K·N(d2)   (call)
      P = K·N(–d2) – F·N(–d1) (put)
    where d1 = [ln(F/K)+0.5σ²T]/(σ√T), d2 = d1 – σ√T.
    """
    if sigma <= 0 or T <= 0 or K <= 0:
        # degenerate → intrinsic
        return max(F-K, 0.0) if opt_type.upper()=='C' else max(K-F, 0.0)

    d1 = (np.log(F/K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type.upper() == 'C':
        return F * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        return K * norm.cdf(-d2) - F * norm.cdf(-d1)

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


def b76_vega_old_notvectorized(F: float, T: float, K: float, sigma: float) -> float:
    """
    Black-76 Vega: ∂Price/∂σ = F * φ(d1) * sqrt(T)
    where d1 = [ln(F/K) + 0.5 σ^2 T] / (σ √T).

    Returns 0.0 if sigma or T are non-positive.
    """
    # guard against degenerate inputs
    if sigma <= 0 or T <= 0 or K <= 0:
        return 0.0

    d1 = (np.log(F/K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return F * norm.pdf(d1) * np.sqrt(T)

def b76_vega_vectorized1(F: float, T: float, K: float, sigma: float) -> float:
    """
    Vectorized Black-76 Vega: ∂Price/∂σ = F * φ(d1) * sqrt(T)
    where d1 = [ln(F/K) + 0.5 σ^2 T] / (σ √T).

    Returns 0.0 for any non-positive inputs in K or sigma.
    This version is vectorized to handle numpy arrays for K and sigma.
    """
    # Ensure inputs that can be arrays are treated as such for robust calculations
    K = np.asanyarray(K)
    sigma = np.asanyarray(sigma)

    # Initialize a vega array of the correct shape, filled with zeros.
    # The result will be 0 for any invalid inputs.
    vega = np.zeros_like(K, dtype=float)

    # Create a boolean mask to identify elements where the calculation is valid.
    # F is a scalar and T is assumed positive.
    valid_mask = (sigma > 1e-9) & (K > 0)

    # Only perform calculations on the subset of valid elements to avoid errors
    # (e.g., division by zero, log of a non-positive number).
    if np.any(valid_mask):
        # Select only the valid elements for the calculation
        K_valid = K[valid_mask]
        sigma_valid = sigma[valid_mask]

        # Calculate d1 only for the valid subset
        d1 = (np.log(F / K_valid) + 0.5 * sigma_valid**2 * T) / (sigma_valid * np.sqrt(T))

        # Calculate vega for the valid subset
        vega_valid = F * norm.pdf(d1) * np.sqrt(T)

        # Place the calculated vega values back into the correct positions in the full array
        np.place(vega, valid_mask, vega_valid)

    return vega


def b76_vega(F: float, T: float, K, sigma):
    """
    Vectorized Black-76 Vega: ∂Price/∂σ = F * φ(d1) * sqrt(T)
    where d1 = [ln(F/K) + 0.5 σ^2 T] / (σ √T).

    This version is robust and handles both scalar and numpy array inputs for K and sigma.
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

### -----------------------------------------------------------------
# bachelier.py
### -----------------------------------------------------------------

# bachelier.py

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
'''
def bachelier_iv(F, T, K, call_price):
    def bach_price(sigma):
        d = (F - K)/(sigma*np.sqrt(T))
        return sigma*np.sqrt(T)*norm.pdf(d) + (F-K)*norm.cdf(d)
    intrinsic = max(0.0, F - K)
    p = max(call_price, intrinsic + 1e-8)
    
    # p = price       # comment this out if use intrinsic above
    
    for a,b in [(1e-8,0.5),(0.5,2.0),(2.0,10.0)]:
        if (bach_price(a)-p)*(bach_price(b)-p) < 0:
            return brentq(lambda x: bach_price(x)-p, a, b, maxiter=200)
    return np.nan
'''
def bachelier_iv(F, T, K, price, opt_type='C'):
    # Intrinsic for each
    intrinsic = max(0.0, F-K) if opt_type.upper() == 'C' else max(0.0, K-F)
    p = max(price, intrinsic + 1e-8)
    def bach_price(sigma):
        d = (F - K)/(sigma*np.sqrt(T))
        if opt_type.upper() == 'C':
            return sigma*np.sqrt(T)*norm.pdf(d) + (F-K)*norm.cdf(d)
        else:  # Put
            return sigma*np.sqrt(T)*norm.pdf(-d) + (K-F)*norm.cdf(-d)
    for a,b in [(1e-8,0.5),(0.5,2.0),(2.0,10.0)]:
        if (bach_price(a)-p)*(bach_price(b)-p) < 0:
            return brentq(lambda x: bach_price(x)-p, a, b, maxiter=200)
    return np.nan


def bachelier_price(F, K, T, sigma):
    d = (F - K)/(sigma*np.sqrt(T))
    return sigma*np.sqrt(T)*norm.pdf(d) + (F-K)*norm.cdf(d)

def bachelier_vega(F, K, T, sigma):
    d = (F - K)/(sigma*np.sqrt(T))
    return np.sqrt(T)*norm.pdf(d)


### -----------------------------------------------------------------
# iv_utils.py
### -----------------------------------------------------------------

# vol_utils.py

#from bachelier import bachelier_iv
#from black76 import black76_iv

IV_ENGINES = {
    'bachelier': bachelier_iv,
    'black76':   black76_iv,
}

def implied_vol(F, T, K, price, opt_type, engine='black76'):
    """
    Universal interface: converts puts→calls, enforces intrinsic floor,
    then calls the selected pricing‐model inversion engine.
    """
#    intrinsic = max(0.0, F - K)
#    # put→call parity
#    p_call = price + intrinsic if opt_type.upper() == 'P' else price
#    # floor extrinsic
#    p_call = max(p_call, intrinsic + 1e-8)
    # delegate to chosen engine
    solver = IV_ENGINES[engine]
    return solver(F, T, K, price, opt_type=opt_type)



### -----------------------------------------------------------------
# sabr_v2.py
### -----------------------------------------------------------------
# sabr_v2.py

import numpy as np
from scipy.optimize import minimize, differential_evolution
#from bachelier import bachelier_vega
#from black76 import b76_vega

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


### -----------------------------------------------------------------
# mdl_rnd.py
### -----------------------------------------------------------------

# analytics_engine/sabr/sabr_rnd.py
import os
import json
import argparse
import numpy as np
#from sabr_v2 import sabr_vol_normal, sabr_vol_lognormal
#from bachelier import bachelier_price
#from sabr_run import load_and_prepare
#from black76 import black76_iv, b76_price


def price_from_sabr(strikes, F, T, alpha, beta, rho, nu):
    """Price European options using SABR-fitted vols under Bachelier model."""
    prices = []
    for K in strikes:
        #sigma = sabr_vol_normal(F, K, T, alpha, rho, nu)
        #p = bachelier_price(F, K, T, sigma)
        sigma = sabr_vol_lognormal(F, K, T, alpha, beta, rho, nu)
        p = b76_price(F, K, T, sigma)
        prices.append(p)
    return np.array(prices)

def second_derivative(f, x, h=None):
    """
    If `f` is an array of values at `x`, approximate d2f/dx2 via two passes of np.gradient.
    Otherwise `f` is assumed callable and we do the finite‐difference f(x±h).
    """

    # array‐based branch
    if isinstance(f, np.ndarray):
        # first derivative, then second derivative
        return np.gradient(np.gradient(f, x), x)
    # callable branch
    if h is None:
        # assume uniform spacing
        h = x[1] - x[0]
    return (f(x + h) - 2*f(x) + f(x - h)) / (h ** 2)

def compute_rnd(strikes, F, T, alpha, rho, nu):
    beta = 1.0  # Fixed beta 
    """Apply Breeden-Litzenberger on SABR-based prices."""
    #f = lambda K: bachelier_price(F, K, T, sabr_vol_normal(F, K, T, alpha, rho, nu))
    f = lambda K: b76_price(F, K, T, sabr_vol_lognormal(F, K, T, alpha, rho, nu))
    pdf = [max(0, second_derivative(f, K)) for K in strikes]
    return np.array(pdf)

def main():
    p = argparse.ArgumentParser("Compute RND from snapshot + SABR params")
    p.add_argument("parquet", help="Path to snapshot parquet file")
    p.add_argument(
        "--params-dir",
        default="analytics_results/model_params",
        help="Root of your model_params tree"
    )
    p.add_argument(
        "--mode", choices=["auto","full","fast"], default="auto",
        help="(Only matters if you call sabr_run from here instead of reusing its output)"
    )
    p.add_argument(
        "--output", default=None,
        help="Optional .npy file to write the resulting PDF"
    )
    args = p.parse_args()

    # 1) Load the snapshot to get strikes, F, T
    strikes, vols, F, T = load_and_prepare(args.parquet)

    # 2) Figure out which expiry folder to look in
    code      = os.path.basename(args.parquet).split("_")[0]     # e.g. "SFRM5"
    code_dir  = os.path.join(args.params_dir, "sabr", code)

    # 3) Pick the latest JSON
    files     = sorted(f for f in os.listdir(code_dir) if f.endswith(".json"))
    param_path = os.path.join(code_dir, files[-1])

    # 4) **Load named params** instead of unpacking a list
    with open(param_path) as f:
        params = json.load(f)
        alpha = params['alpha']
        beta  = params['beta']
        rho   = params['rho']
        nu    = params['nu']

    # 5) Compute & output the RND
    pdf = compute_rnd(strikes, F, T, alpha, beta, rho, nu)
    if args.output:
        np.save(args.output, pdf)
        print(f"Saved PDF to {args.output}")
    else:
        for k, p in zip(strikes, pdf):
            print(f"{k}: {p}")

if __name__ == "__main__":
    main()



### -----------------------------------------------------------------
# sabr_run.py
### -----------------------------------------------------------------

# app.py (formerly read_parquet_v3.py)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import hashlib
import matplotlib.pyplot as plt

# Calibration & pricing imports
#from iv_utils import implied_vol
#from sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast, sabr_vol_normal
#from sabr_rnd import price_from_sabr, second_derivative, compute_rnd
#from bachelier import bachelier_price

# New modular imports
#from mdl_load import discover_snapshot_files, save_uploaded_files
#from mdl_calibration import fit_sabr, load_global_beta, calibrate_global_beta
#from mdl_rnd_utils import market_rnd, model_rnd
#rom mdl_plot import plot_vol_smile, plot_rnd

st.set_page_config(layout="wide", page_title="SOFR Option Chain Diagnostics")
st.title("SOFR Option Chain Diagnostics")

# --- 1. File selection via modular loader ---
file_dict = discover_snapshot_files("snapshots")
selected_folders = st.sidebar.multiselect(
    "Folders to load:", options=list(file_dict.keys()), default=[]
)

# Clear Cache button
col_main, col_clear = st.columns([1, 9])


all_files = []
for folder in selected_folders:
    st.sidebar.markdown(f"**{folder}/**")
    files = file_dict.get(folder, [])
    chosen = st.sidebar.multiselect(
        f"Files in {folder}/", options=files, default=[], key=folder
    )
    all_files.extend(chosen)

uploaded = save_uploaded_files(
    st.sidebar.file_uploader(
        "Or add Parquet files", type="parquet", accept_multiple_files=True
    )
)

files_to_show = all_files + uploaded
if not files_to_show:
    st.warning("No files selected or uploaded.")
    st.stop()


### --- 2. Manual SABR Calibration (one file at a time) ---
with st.sidebar.form(key='manual_sabr_form', clear_on_submit=False):
    st.markdown("### Manual SABR Calibration")
    # 2.a: choose exactly one file
    manual_file = st.selectbox(
        "File to recalibrate",
        options=files_to_show,
        format_func=lambda f: os.path.basename(f)
    )
    st.markdown("#### Parameter inputs")
    alpha_in = st.number_input(
        "alpha", min_value=1e-4, max_value=5.0,
        value=0.1, step=1e-4, format="%.5f"
    )
    beta_in = st.number_input(
        "beta", min_value=0.0, max_value=1.0,
        value=0.5, step=1e-4, format="%.5f"
    )
    rho_in = st.number_input(
        "rho", min_value=-0.99999, max_value=0.99999,
        value=0.0, step=1e-5, format="%.5f"
    )
    nu_in = st.number_input(
        "nu", min_value=1e-4, max_value=5.0,
        value=0.1, step=1e-4, format="%.5f"
    )
    recalibrate = st.form_submit_button("Recalibrate")
manual_params = dict(alpha=alpha_in, beta=beta_in, rho=rho_in, nu=nu_in)
st.session_state['manual_file'] = manual_file

# --- 3. Visibility toggles for Vol & RND ---
def get_visibility_state(label, files, default=True):
    key = f"{label}_visible"
    if key not in st.session_state:
        st.session_state[key] = {file_path: default for file_path in files}
    for file_path in files:
        basename = os.path.basename(file_path)
        st.session_state[key][file_path] = st.sidebar.checkbox(
            f"Show {label} ({basename})",
            value=st.session_state[key].get(file_path, default),
            key=f"{label}_{file_path}"
        )
    return st.session_state[key]

vol_visible = get_visibility_state("Vol Smile", files_to_show)
rnd_visible = get_visibility_state("RND", files_to_show)

# --- 4. Refresh buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("Refresh Vol Smile Chart"):
        st.session_state["refresh_vol"] = not st.session_state.get("refresh_vol", True)
with col2:
    if st.button("Refresh RND Chart"):
        st.session_state["refresh_rnd"] = not st.session_state.get("refresh_rnd", True)

# --- 5. Process each file  ---
@st.cache_data(show_spinner="Calibrating...", persist=True)
def process_snapshot_file(parquet_path, manual_params):
    df_raw = pd.read_parquet(parquet_path)
    df = df_raw.copy()
    df['strike'] = df['ticker'].str.extract(r'\b(\d+\.\d+)\b')[0].astype(float)
    liquid = df[(df['bid']>0)&(df['ask']>0)]
    if liquid.empty: return None
    lo, hi = liquid['strike'].min(), liquid['strike'].max()
    df_trim = df[(df['strike']>=lo)&(df['strike']<=hi)].reset_index(drop=True)
    df_trim['mid_price'] = np.where((df_trim['bid']>0)&(df_trim['ask']>0), 0.5*(df_trim['bid']+df_trim['ask']), np.nan)
    df_trim['type'] = df_trim['type'].str.upper()
    F = float(df_trim['future_px'].iloc[0])
    df_otm = df_trim[((df_trim['type']=='C')&(df_trim['strike']>F))|((df_trim['type']=='P')&(df_trim['strike']<F))].reset_index(drop=True)
    df_otm = df_otm.sort_values(by='strike').reset_index(drop=True)
    if df_otm.empty: return None
    snap_dt = datetime.strptime(df_otm['snapshot_ts'].iloc[0], '%Y%m%d %H%M%S')
    expiry = pd.to_datetime(df_otm['expiry_date'].iloc[0]).date()
    T = (expiry - snap_dt.date()).days/365.0
    
    df_otm['iv'] = df_otm.apply(lambda r: implied_vol(F=float(r['future_px']), T=T, K=r['strike'], price=r['mid_price'], opt_type=r['type'], engine='black76') if not np.isnan(r['mid_price']) else np.nan, axis=1)
    
    df_otm = df_otm[df_otm['iv']<100]
    liquid2 = df_otm[df_otm['volume']>0]
    if liquid2.empty: return None
    lo2, hi2 = liquid2['strike'].min(), liquid2['strike'].max()
    df_otm = df_otm[(df_otm['strike']>=lo2-0.52)&(df_otm['strike']<=hi2+0.52)]
    df_otm['spread'] = df_otm['ask'] - df_otm['bid']
    df_otm = df_otm[df_otm['spread']<=0.012]
    strikes = df_otm['strike'].values
    market_iv = df_otm['iv'].values
    mask = ~np.isnan(market_iv)
    fit_order = np.argsort(strikes[mask])
    strikes_fit = strikes[mask][fit_order]
    vols_fit = market_iv[mask][fit_order]
    
    # Automatic calibration
    params_fast, iv_model_fit, debug_data = fit_sabr(strikes_fit, F, T, vols_fit, method='fast')
    
    # ---  MANUAL CALIBRATION ---
    params_man, iv_manual = (None, None) # Initialize iv_manual as None for plotting
    if recalibrate and st.session_state.get('manual_file') == parquet_path:
        # Correctly unpack the 3-item tuple returned by fit_sabr
        manual_results = fit_sabr(strikes_fit, F, T, vols_fit, method='fast', manual_params=manual_params)
        
        if manual_results and len(manual_results) == 3:
            params_man, iv_manual_fit, debug_data = manual_results
            
            # Interpolate the manual IV from the fit grid back to the main strike grid
            if iv_manual_fit is not None and len(iv_manual_fit) > 0:
                 iv_manual = np.interp(strikes, strikes_fit, iv_manual_fit)
    
    # Interpolate the automatic model's IV back to the main grid for consistent plotting
    model_iv_on_market_strikes = np.interp(strikes, strikes_fit, iv_model_fit)
    # --- END OF CORRECTED SECTION ---

    mid_prices = df_otm['mid_price'].values
    rnd_mkt = market_rnd(strikes, mid_prices)
    rnd_sabr = model_rnd(strikes, F, T, params_fast)
    rnd_man  = model_rnd(strikes, F, T, params_man) if params_man else None

    area_market = float(np.trapezoid(rnd_mkt, strikes))
    area_model  = float(np.trapezoid(rnd_sabr, strikes))

    return {'strikes': strikes, 'market_iv': market_iv,
            'model_iv': model_iv_on_market_strikes, 
            'iv_manual': iv_manual, 
            'rnd_market': rnd_mkt, 'rnd_sabr': rnd_sabr,
            'rnd_manual': rnd_man, 'params_fast': params_fast,
            'params_manual': params_man, 'mid_prices': mid_prices,
            'area_model': area_model, 'area_market': area_market,
            'forward_price': F,
            'debug_data': debug_data
            }
    
results = {f: process_snapshot_file(f, manual_params) for f in files_to_show}

# --- Display Forward Prices ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Forward Prices")
for fname, res in results.items():
    if res and res.get('forward_price'):
        label = os.path.basename(fname)
        fwd_price = res['forward_price']
        st.sidebar.markdown(f"**{label}:** `{fwd_price:.4f}`")


col1, col2 = st.columns([1,3])
with col1:
    if st.button("Historical β-Calibrate"):
        st.info("Optimizing β across selected snapshots…")
        β_opt = calibrate_global_beta(files_to_show)
        st.success(f"Global β optimized: {β_opt:.4f}")
        process_snapshot_file.clear()
        st.warning("Calibration cache cleared. Please refresh your browser (F5) to apply the new β.")
        st.stop()
with col2:
    st.metric("Current β", f"{load_global_beta():.4f}")

# --- 6. Plot via plotting module ---
if st.session_state.get("refresh_vol", True):
    show_mkt_iv    = st.checkbox("Show Market IV",    value=True, key="toggle_mkt_iv")
    show_model_iv  = st.checkbox("Show SABR Model IV", value=True, key="toggle_model_iv")
    show_manual_iv = st.checkbox("Show Manual IV",     value=True, key="toggle_manual_iv")

    fig = plot_vol_smile(results, vol_visible, show_mkt_iv, show_model_iv, show_manual_iv)
    st.pyplot(fig, clear_figure=True)


if st.session_state.get("refresh_rnd", True):
    show_mkt_rnd    = st.checkbox("Show Market RND",    value=True,   key="toggle_mkt_rnd")
    show_model_rnd  = st.checkbox("Show SABR RND",      value=True,   key="toggle_model_rnd")
    show_manual_rnd = st.checkbox("Show Manual RND",    value=False,  key="toggle_manual_rnd")

    fig2 = plot_rnd(results, rnd_visible, show_mkt_rnd, show_model_rnd, show_manual_rnd)
    st.pyplot(fig2, clear_figure=True)

# --- 7. Debug & parameter tables ---
with col_clear:
    if st.button("Clear Cache"):
        process_snapshot_file.clear()
        st.warning("Calibration cache cleared. Refresh (F5) to rerun calibration.")
        st.stop()

with st.expander("Debug 2.0: Snapshot Data & Params"):
    for f, res in results.items():
        if not res: continue
        st.markdown(f"**{os.path.basename(f)}**")
        st.write("Params Fast:", res['params_fast'])
        st.write("Params Manual:", res['params_manual'])
        
        debug_strikes = np.array(res['strikes'])
        debug_sorted = np.all(np.diff(debug_strikes) > 0)

        debug_model_rnd = {
            'integral':      round(res.get('area_model', 0), 6),
            'all_nonneg':    bool(np.all(res.get('rnd_sabr', [0]) >= 0))
        }
        debug_market_rnd = {
            'integral':      round(res.get('area_market', 0), 6),
            'all_nonneg':    bool(np.all(res.get('rnd_market', [0]) >= 0))
        }
        debug_info = {
            'strikes_sorted':  debug_sorted,
            'market_rnd':      debug_market_rnd,
            'model_rnd':       debug_model_rnd
        }
        st.write(debug_info)

with st.expander("Calibration Debug: Interpolated Smile Target"):
    st.info("This plot shows the raw market points (blue dots) and the smooth curve (red line) that the SABR model is actually calibrated against.")
    
    for fname, res in results.items():
        if res and res.get('debug_data'):
            st.markdown(f"#### {os.path.basename(fname)}")
            
            raw_strikes = res['strikes']
            raw_vols = res['market_iv']
            debug_info = res['debug_data']
            interp_strikes = debug_info['interp_strikes']
            interp_vols = debug_info['interp_vols']
            
            fig_debug, ax_debug = plt.subplots()
            ax_debug.plot(interp_strikes, interp_vols, 'r-', label="Interpolated Curve (Calibration Target)")
            ax_debug.plot(raw_strikes, raw_vols, 'bo', label="Raw Market IV Points", markersize=5)
            
            ax_debug.set_title("Interpolation Sanity Check")
            ax_debug.set_xlabel("Strike")
            ax_debug.set_ylabel("Implied Volatility")
            ax_debug.legend()
            ax_debug.grid(True)
            
            st.pyplot(fig_debug, clear_figure=True)



### -----------------------------------------------------------------
# mdl_load.py
### -----------------------------------------------------------------

# mdl_load.py
# This module load parquet files from storage

import glob
import os
from typing import Dict, List


def discover_snapshot_files(root_folder: str = "snapshots") -> Dict[str, List[str]]:
    """
    Scan `root_folder` recursively for .parquet files.
    Returns a dict mapping each subfolder (relative to root) to list of file paths.
    """
    local_files = sorted(glob.glob(f"{root_folder}/**/*.parquet", recursive=True))
    file_dict: Dict[str, List[str]] = {}
    for f in local_files:
        folder = os.path.relpath(os.path.dirname(f), root_folder)
        file_dict.setdefault(folder, []).append(f)
    return file_dict


def save_uploaded_files(uploaded_files, upload_dir: str = "uploaded_files") -> List[str]:
    """
    Save Streamlit-uploaded files to disk and return list of saved paths.
    """
    os.makedirs(upload_dir, exist_ok=True)
    uploaded_paths: List[str] = []
    for f in uploaded_files:
        file_path = os.path.join(upload_dir, f.name)
        with open(file_path, "wb") as out_f:
            out_f.write(f.read())
        uploaded_paths.append(file_path)
    return uploaded_paths



### -----------------------------------------------------------------
# mdl_calibration.py
### -----------------------------------------------------------------

# calibration.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline, make_interp_spline, UnivariateSpline
from typing import Tuple, Dict
#from sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast, sabr_vol_normal, sabr_vol_lognormal
#from sabr_v2 import b76_vega
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
    #from iv_utils import implied_vol

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
   
    #vol_spline = CubicSpline(strikes, vols, bc_type='natural')
    #vol_spline = make_interp_spline(strikes, vols, k=3)
        # --- 1. Weighted Smoothing Spline Step ---
    
    # --- a) Define weights to clamp the fit at the ATM region ---
    # We use a Gaussian function to create smoothly decaying weights away from the forward price F.
    # The 'sigma' parameter controls how quickly the weights fall off. A smaller sigma means a tighter clamp.
    sigma = 0.10 * F  # Example: 5% of the forward price. Tune this as needed.
    weights = np.exp(-0.5 * ((strikes - F) / sigma)**2)
    # Give a huge boost to the point(s) closest to ATM to ensure a near-perfect fit there.
    atm_idx = np.abs(strikes - F).argmin()
    weights[atm_idx] = 100.0 # A very high weight for the closest point
    
    # --- b) Define the smoothing factor 's' ---
    # 's' controls the trade-off between smoothness and fitting the data points.
    # s=0 means interpolation (pass through all points).
    # A larger 's' creates a smoother curve. We start with a small value.
    # This is a heuristic and may require tuning.
    s = len(vols) * np.var(vols) * 0.01 
    
    # --- c) Create the weighted smoothing B-spline ---
    # UnivariateSpline creates a B-spline representation.
    # It minimizes: sum(w[i] * (y[i] - spline(x[i]))**2) + s * integral(spline''(x)**2 dx)
    vol_spline = UnivariateSpline(strikes, vols, w=weights, s=s, k=3)
 
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



### -----------------------------------------------------------------
# mdl_plot.py
### -----------------------------------------------------------------

# plot_iv.py
# This module plot IV and RND

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_vol_smile_old(results: dict, vol_visible: dict) -> plt.Figure:
    """
    Create a volatility smile plot from `results`.
    `vol_visible[fname]` toggles visibility for each file.
    """ 
    fig, ax = plt.subplots()
    for fname, res in results.items():
        if not res or not vol_visible.get(fname):
            continue
        label = os.path.basename(fname)
        ax.plot(res['strikes'], res['market_iv'],    marker='o', linestyle='-', label=f"Market ({label})")
        ax.plot(res['strikes'], res['model_iv'],     marker='x', linestyle='--', label=f"SABR ({label})")
        if res.get('model_iv_manual') is not None:
            ax.plot(res['strikes'], res['model_iv_manual'], marker='s', linestyle=':', label=f"Manual ({label})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("IV")
    ax.set_title("Volatility Smile (OTM)")
    ax.legend()
    return fig

def plot_vol_smile(results: dict, vol_visible: dict, show_mkt_iv, show_model_iv, show_manual_iv) -> plt.Figure:
    """
    Create a volatility smile plot from `results`.
    Highlights the ATM strike with a star marker and displays the forward price.
    `vol_visible[fname]` toggles visibility for each file.
    """ 
    fig, ax = plt.subplots(figsize=(10, 7)) # Make figure a bit bigger

    # Get the default color cycle from matplotlib
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i, (fname, res) in enumerate(results.items()):
        # Skip if file is toggled off or has no result
        if not res or not vol_visible.get(fname):
            continue
            
        # Assign a consistent color for the current file
        color = colors[i % len(colors)]
        label_base = os.path.basename(fname)

        # Extract data from the results dictionary
        strikes = res.get('strikes')
        fwd_price = res.get('forward_price')
        
        # Skip if essential data is missing
        if strikes is None or len(strikes) == 0 or fwd_price is None:
            continue

        # Find the index of the strike closest to the forward price (ATM)
        atm_idx = np.abs(strikes - fwd_price).argmin()
        
        # Create a boolean mask to select all non-ATM points
        otm_mask = np.ones_like(strikes, dtype=bool)
        otm_mask[atm_idx] = False
        
        # Add a vertical line at the forward price to the plot
        ax.axvline(x=fwd_price, color=color, linestyle=':', linewidth=1.5, label=f'Fwd ({label_base}) = {fwd_price:.4f}')

        # --- Plot Market IV ---
        market_iv = res.get('market_iv')
        if show_mkt_iv and market_iv is not None:
            # Plot non-ATM points with small dots
            ax.plot(strikes[otm_mask], market_iv[otm_mask], marker='o', markersize=5, linestyle='-', label=f"Market ({label_base})", color=color)
            # Highlight the ATM point with a large, edged star
            ax.plot(strikes[atm_idx], market_iv[atm_idx], marker='*', markersize=15, linestyle='None', markeredgecolor='black', markerfacecolor=color)

        # --- Plot Model IV ---
        model_iv = res.get('model_iv')
        if show_model_iv and model_iv is not None:
            # Plot non-ATM points with a dashed line
            ax.plot(strikes[otm_mask], model_iv[otm_mask], marker='o', markersize=5, linestyle='--', label=f"SABR ({label_base})", color=color, alpha=0.8)
            # Highlight the ATM point
            ax.plot(strikes[atm_idx], model_iv[atm_idx], marker='*', markersize=15, linestyle='None', markeredgecolor='black', markerfacecolor=color, alpha=0.7)

        # --- Plot Manual IV ---
        manual_iv = res.get('iv_manual')
        if show_manual_iv and manual_iv is not None:
            # Plot non-ATM points
            ax.plot(strikes[otm_mask], manual_iv[otm_mask], marker='s', markersize=4, linestyle=':', label=f"Manual ({label_base})", color=color)
            # Highlight the ATM point
            ax.plot(strikes[atm_idx], manual_iv[atm_idx], marker='*', markersize=15, linestyle='None', markeredgecolor='black', markerfacecolor=color, alpha=0.7)

    # Configure plot aesthetics
    ax.set_xlabel("Strike")
    ax.set_ylabel("IV")
    ax.set_title("Volatility Smile (OTM)")
    ax.legend(fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    
    return fig

def plot_rnd(results: dict, rnd_visible: dict, show_mkt_rnd: bool, show_model_rnd: bool, show_manual_rnd: bool) -> plt.Figure:
    """
    Create a Risk-Neutral Density plot from `results`.
    `rnd_visible[fname]` toggles visibility for each file.
    """
    fig, ax = plt.subplots()
    for fname, res in results.items():
        if not res or not rnd_visible.get(fname):
            continue
        label = os.path.basename(fname)
        if show_mkt_rnd:
            ax.plot(res['strikes'], res['rnd_market'], marker='o', linestyle='-', label=f"Market RND ({label})")
        
        if show_model_rnd:
            ax.plot(res['strikes'], res['rnd_sabr'],   marker='x', linestyle='--', label=f"SABR RND ({label})")
        
        if show_manual_rnd and res.get('rnd_manual') is not None:
            ax.plot(res['strikes'], res['rnd_manual'], marker='s', linestyle=':', label=f"Manual RND ({label})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("RND (normalized)")
    ax.set_title("Risk-Neutral Density (RND)")
    ax.legend()
    return fig



### -----------------------------------------------------------------
# mdl_run.py
### -----------------------------------------------------------------

# app.py (formerly read_parquet_v3.py)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import hashlib
import matplotlib.pyplot as plt

# Calibration & pricing imports
#from iv_utils import implied_vol
#from sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast, sabr_vol_normal
#from sabr_rnd import price_from_sabr, second_derivative, compute_rnd
#from bachelier import bachelier_price

# New modular imports
#from mdl_load import discover_snapshot_files, save_uploaded_files
#from mdl_calibration import fit_sabr, load_global_beta, calibrate_global_beta
#from mdl_rnd_utils import market_rnd, model_rnd
#from mdl_plot import plot_vol_smile, plot_rnd

st.set_page_config(layout="wide", page_title="SOFR Option Chain Diagnostics")
st.title("SOFR Option Chain Diagnostics")

# --- 1. File selection via modular loader ---
file_dict = discover_snapshot_files("snapshots")
selected_folders = st.sidebar.multiselect(
    "Folders to load:", options=list(file_dict.keys()), default=[]
)

# Clear Cache button
col_main, col_clear = st.columns([1, 9])


all_files = []
for folder in selected_folders:
    st.sidebar.markdown(f"**{folder}/**")
    files = file_dict.get(folder, [])
    chosen = st.sidebar.multiselect(
        f"Files in {folder}/", options=files, default=[], key=folder
    )
    all_files.extend(chosen)

uploaded = save_uploaded_files(
    st.sidebar.file_uploader(
        "Or add Parquet files", type="parquet", accept_multiple_files=True
    )
)

files_to_show = all_files + uploaded
if not files_to_show:
    st.warning("No files selected or uploaded.")
    st.stop()


### --- 2. Manual SABR Calibration (one file at a time) ---
with st.sidebar.form(key='manual_sabr_form', clear_on_submit=False):
    st.markdown("### Manual SABR Calibration")
    # 2.a: choose exactly one file
    manual_file = st.selectbox(
        "File to recalibrate",
        options=files_to_show,
        format_func=lambda f: os.path.basename(f)
    )
    st.markdown("#### Parameter inputs")
    alpha_in = st.number_input(
        "alpha", min_value=1e-4, max_value=5.0,
        value=0.1, step=1e-4, format="%.5f"
    )
    beta_in = st.number_input(
        "beta", min_value=0.0, max_value=1.0,
        value=0.5, step=1e-4, format="%.5f"
    )
    rho_in = st.number_input(
        "rho", min_value=-0.99999, max_value=0.99999,
        value=0.0, step=1e-5, format="%.5f"
    )
    nu_in = st.number_input(
        "nu", min_value=1e-4, max_value=5.0,
        value=0.1, step=1e-4, format="%.5f"
    )
    recalibrate = st.form_submit_button("Recalibrate")
manual_params = dict(alpha=alpha_in, beta=beta_in, rho=rho_in, nu=nu_in)
st.session_state['manual_file'] = manual_file

# --- 3. Visibility toggles for Vol & RND ---
def get_visibility_state(label, files, default=True):
    key = f"{label}_visible"
    if key not in st.session_state:
        st.session_state[key] = {file_path: default for file_path in files}
    for file_path in files:
        basename = os.path.basename(file_path)
        st.session_state[key][file_path] = st.sidebar.checkbox(
            f"Show {label} ({basename})",
            value=st.session_state[key].get(file_path, default),
            key=f"{label}_{file_path}"
        )
    return st.session_state[key]

vol_visible = get_visibility_state("Vol Smile", files_to_show)
rnd_visible = get_visibility_state("RND", files_to_show)

# --- 4. Refresh buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("Refresh Vol Smile Chart"):
        st.session_state["refresh_vol"] = not st.session_state.get("refresh_vol", True)
with col2:
    if st.button("Refresh RND Chart"):
        st.session_state["refresh_rnd"] = not st.session_state.get("refresh_rnd", True)

# --- 5. Process each file  ---
@st.cache_data(show_spinner="Calibrating...", persist=True)
def process_snapshot_file(parquet_path, manual_params):
    df_raw = pd.read_parquet(parquet_path)
    df = df_raw.copy()
    df['strike'] = df['ticker'].str.extract(r'\b(\d+\.\d+)\b')[0].astype(float)
    liquid = df[(df['bid']>0)&(df['ask']>0)]
    if liquid.empty: return None
    lo, hi = liquid['strike'].min(), liquid['strike'].max()
    df_trim = df[(df['strike']>=lo)&(df['strike']<=hi)].reset_index(drop=True)
    df_trim['mid_price'] = np.where((df_trim['bid']>0)&(df_trim['ask']>0), 0.5*(df_trim['bid']+df_trim['ask']), np.nan)
    df_trim['type'] = df_trim['type'].str.upper()
    F = float(df_trim['future_px'].iloc[0])
    df_otm = df_trim[((df_trim['type']=='C')&(df_trim['strike']>F))|((df_trim['type']=='P')&(df_trim['strike']<F))].reset_index(drop=True)
    df_otm = df_otm.sort_values(by='strike').reset_index(drop=True)
    if df_otm.empty: return None
    snap_dt = datetime.strptime(df_otm['snapshot_ts'].iloc[0], '%Y%m%d %H%M%S')
    expiry = pd.to_datetime(df_otm['expiry_date'].iloc[0]).date()
    T = (expiry - snap_dt.date()).days/365.0
    
    df_otm['iv'] = df_otm.apply(lambda r: implied_vol(F=float(r['future_px']), T=T, K=r['strike'], price=r['mid_price'], opt_type=r['type'], engine='black76') if not np.isnan(r['mid_price']) else np.nan, axis=1)
    
    df_otm = df_otm[df_otm['iv']<100]
    liquid2 = df_otm[df_otm['volume']>0]
    if liquid2.empty: return None
    lo2, hi2 = liquid2['strike'].min(), liquid2['strike'].max()
    df_otm = df_otm[(df_otm['strike']>=lo2-0.52)&(df_otm['strike']<=hi2+0.52)]
    df_otm['spread'] = df_otm['ask'] - df_otm['bid']
    df_otm = df_otm[df_otm['spread']<=0.012]
    strikes = df_otm['strike'].values
    market_iv = df_otm['iv'].values
    mask = ~np.isnan(market_iv)
    fit_order = np.argsort(strikes[mask])
    strikes_fit = strikes[mask][fit_order]
    vols_fit = market_iv[mask][fit_order]
    
    # Automatic calibration
    params_fast, iv_model_fit, debug_data = fit_sabr(strikes_fit, F, T, vols_fit, method='fast')
    
    # ---  MANUAL CALIBRATION ---
    params_man, iv_manual = (None, None) # Initialize iv_manual as None for plotting
    if recalibrate and st.session_state.get('manual_file') == parquet_path:
        # Correctly unpack the 3-item tuple returned by fit_sabr
        manual_results = fit_sabr(strikes_fit, F, T, vols_fit, method='fast', manual_params=manual_params)
        
        if manual_results and len(manual_results) == 3:
            params_man, iv_manual_fit, debug_data = manual_results
            
            # Interpolate the manual IV from the fit grid back to the main strike grid
            if iv_manual_fit is not None and len(iv_manual_fit) > 0:
                 iv_manual = np.interp(strikes, strikes_fit, iv_manual_fit)
    
    # Interpolate the automatic model's IV back to the main grid for consistent plotting
    model_iv_on_market_strikes = np.interp(strikes, strikes_fit, iv_model_fit)
    # --- END OF CORRECTED SECTION ---

    mid_prices = df_otm['mid_price'].values
    rnd_mkt = market_rnd(strikes, mid_prices)
    rnd_sabr = model_rnd(strikes, F, T, params_fast)
    rnd_man  = model_rnd(strikes, F, T, params_man) if params_man else None

    area_market = float(np.trapezoid(rnd_mkt, strikes))
    area_model  = float(np.trapezoid(rnd_sabr, strikes))

    return {'strikes': strikes, 'market_iv': market_iv,
            'model_iv': model_iv_on_market_strikes, 
            'iv_manual': iv_manual, 
            'rnd_market': rnd_mkt, 'rnd_sabr': rnd_sabr,
            'rnd_manual': rnd_man, 'params_fast': params_fast,
            'params_manual': params_man, 'mid_prices': mid_prices,
            'area_model': area_model, 'area_market': area_market,
            'forward_price': F,
            'debug_data': debug_data
            }
    
results = {f: process_snapshot_file(f, manual_params) for f in files_to_show}

# --- Display Forward Prices ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Forward Prices")
for fname, res in results.items():
    if res and res.get('forward_price'):
        label = os.path.basename(fname)
        fwd_price = res['forward_price']
        st.sidebar.markdown(f"**{label}:** `{fwd_price:.4f}`")


col1, col2 = st.columns([1,3])
with col1:
    if st.button("Historical β-Calibrate"):
        st.info("Optimizing β across selected snapshots…")
        β_opt = calibrate_global_beta(files_to_show)
        st.success(f"Global β optimized: {β_opt:.4f}")
        process_snapshot_file.clear()
        st.warning("Calibration cache cleared. Please refresh your browser (F5) to apply the new β.")
        st.stop()
with col2:
    st.metric("Current β", f"{load_global_beta():.4f}")

# --- 6. Plot via plotting module ---
if st.session_state.get("refresh_vol", True):
    show_mkt_iv    = st.checkbox("Show Market IV",    value=True, key="toggle_mkt_iv")
    show_model_iv  = st.checkbox("Show SABR Model IV", value=True, key="toggle_model_iv")
    show_manual_iv = st.checkbox("Show Manual IV",     value=True, key="toggle_manual_iv")

    fig = plot_vol_smile(results, vol_visible, show_mkt_iv, show_model_iv, show_manual_iv)
    st.pyplot(fig, clear_figure=True)


if st.session_state.get("refresh_rnd", True):
    show_mkt_rnd    = st.checkbox("Show Market RND",    value=True,   key="toggle_mkt_rnd")
    show_model_rnd  = st.checkbox("Show SABR RND",      value=True,   key="toggle_model_rnd")
    show_manual_rnd = st.checkbox("Show Manual RND",    value=False,  key="toggle_manual_rnd")

    fig2 = plot_rnd(results, rnd_visible, show_mkt_rnd, show_model_rnd, show_manual_rnd)
    st.pyplot(fig2, clear_figure=True)

# --- 7. Debug & parameter tables ---
with col_clear:
    if st.button("Clear Cache"):
        process_snapshot_file.clear()
        st.warning("Calibration cache cleared. Refresh (F5) to rerun calibration.")
        st.stop()

with st.expander("Debug 2.0: Snapshot Data & Params"):
    for f, res in results.items():
        if not res: continue
        st.markdown(f"**{os.path.basename(f)}**")
        st.write("Params Fast:", res['params_fast'])
        st.write("Params Manual:", res['params_manual'])
        
        debug_strikes = np.array(res['strikes'])
        debug_sorted = np.all(np.diff(debug_strikes) > 0)

        debug_model_rnd = {
            'integral':      round(res.get('area_model', 0), 6),
            'all_nonneg':    bool(np.all(res.get('rnd_sabr', [0]) >= 0))
        }
        debug_market_rnd = {
            'integral':      round(res.get('area_market', 0), 6),
            'all_nonneg':    bool(np.all(res.get('rnd_market', [0]) >= 0))
        }
        debug_info = {
            'strikes_sorted':  debug_sorted,
            'market_rnd':      debug_market_rnd,
            'model_rnd':       debug_model_rnd
        }
        st.write(debug_info)

with st.expander("Calibration Debug: Interpolated Smile Target"):
    st.info("This plot shows the raw market points (blue dots) and the smooth curve (red line) that the SABR model is actually calibrated against.")
    
    for fname, res in results.items():
        if res and res.get('debug_data'):
            st.markdown(f"#### {os.path.basename(fname)}")
            
            raw_strikes = res['strikes']
            raw_vols = res['market_iv']
            debug_info = res['debug_data']
            interp_strikes = debug_info['interp_strikes']
            interp_vols = debug_info['interp_vols']
            
            fig_debug, ax_debug = plt.subplots()
            ax_debug.plot(interp_strikes, interp_vols, 'r-', label="Interpolated Curve (Calibration Target)")
            ax_debug.plot(raw_strikes, raw_vols, 'bo', label="Raw Market IV Points", markersize=5)
            
            ax_debug.set_title("Interpolation Sanity Check")
            ax_debug.set_xlabel("Strike")
            ax_debug.set_ylabel("Implied Volatility")
            ax_debug.legend()
            ax_debug.grid(True)
            
            st.pyplot(fig_debug, clear_figure=True)
