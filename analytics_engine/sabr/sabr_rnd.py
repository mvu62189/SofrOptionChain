# analytics_engine/sabr/sabr_rnd.py
import os
import json
import argparse
import numpy as np
from sabr_v2 import sabr_vol_normal
from bachelier import bachelier_price
from sabr_run import load_and_prepare



def price_from_sabr(strikes, F, T, alpha, beta, rho, nu):
    """Price European options using SABR-fitted vols under Bachelier model."""
    prices = []
    for K in strikes:
        sigma = sabr_vol_normal(F, K, T, alpha, rho, nu)
        p = bachelier_price(F, K, T, sigma)
        prices.append(p)
    return np.array(prices)

def second_derivative(f, x, h=None):
    """
    If `f` is an array of values at `x`, approximate d2f/dx2 via two passes of np.gradient.
    Otherwise `f` is assumed callable and we do the finite‐difference f(x±h).
    """
    import numpy as _np
    # array‐based branch
    if isinstance(f, _np.ndarray):
        # first derivative, then second derivative
        return _np.gradient(_np.gradient(f, x), x)
    # callable branch
    if h is None:
        # assume uniform spacing
        h = x[1] - x[0]
    return (f(x + h) - 2*f(x) + f(x - h)) / (h ** 2)

def compute_rnd(strikes, F, T, alpha, rho, nu):
    beta = 0.5  # Fixed beta 
    """Apply Breeden-Litzenberger on SABR-based prices."""
    f = lambda K: bachelier_price(F, K, T, sabr_vol_normal(F, K, T, alpha, rho, nu))
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
