import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from analytics_engine.sabr.sabr_run import load_and_prepare
from analytics_engine.sabr.sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast, sabr_vol_normal
from analytics_engine.sabr.sabr_rnd import compute_rnd
from analytics_engine.sabr.bachelier import bachelier_price

def plot_smile_comparison(strikes, market_vols, model_vols, title):
    plt.figure(figsize=(8, 4))
    plt.plot(strikes, market_vols, 'o', label='Market IV')
    plt.plot(strikes, model_vols, '-', label='Model IV')
    plt.title(title)
    plt.xlabel('Strike')
    plt.ylabel('Normal IV')
    plt.legend()
    plt.show()

def plot_rnd_comparison(strikes, market_rnd, model_rnd, title):
    plt.figure(figsize=(64, 32))
    plt.plot(strikes, market_rnd, 'o', label='Market RND')
    plt.plot(strikes, model_rnd, '-', label='Model RND')
    plt.title(title)
    plt.xlabel('Strike')
    plt.ylabel('PDF')
    plt.legend()
    plt.show()

def compute_market_rnd(strikes, F, T, market_vols, h=1e-2):
    # Breeden-Litzenberger on market prices
    f = lambda K, vol: bachelier_price(F, K, T, vol)
    rnd = []
    for i, K in enumerate(strikes):
        # central difference using market vols (interpolating vol at K±h)
        vol_up = np.interp(K+h, strikes, market_vols)
        vol_dn = np.interp(K-h, strikes, market_vols)
        p_up = f(K+h, vol_up)
        p   = f(K,   market_vols[i])
        p_dn = f(K-h, vol_dn)
        rnd.append(max(0, (p_up - 2*p + p_dn) / h**2))
    return np.array(rnd)

def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

def test_sabr_full_and_fast_integration():
    # Paths to two snapshots with the same expiry code
    snapshot1 = 'snapshots/20250617/122819/SFRU5_sep.parquet'
    snapshot2 = 'snapshots/20250618/161129/SFRU5_sep.parquet'
    params_dir = 'analytics_results/model_params'

    # Full calibration on snapshot1
    strikes1, market_vols1, F1, T1 = load_and_prepare(snapshot1)
    full_params = calibrate_sabr_full(strikes1, market_vols1, F1, T1)
    alpha_f, beta_f, rho_f, nu_f = full_params
    model_vols1 = np.array([sabr_vol_normal(F1, K, T1, *full_params) for K in strikes1])

    # Plot IV smile comparison
    plot_smile_comparison(strikes1, market_vols1, model_vols1, 'Full Calibration Smile')

    # Compute and plot RND
    market_rnd1 = compute_market_rnd(strikes1, F1, T1, market_vols1)
    model_rnd1  = compute_rnd(strikes1, F1, T1, *full_params)
    plot_rnd_comparison(strikes1, market_rnd1, model_rnd1, 'Full Calibration RND')

    # Print error metrics
    print(f"Full calibration IV RMSE: {rmse(market_vols1, model_vols1):.6f}")
    print(f"Full calibration RND RMSE: {rmse(market_rnd1, model_rnd1):.6f}")

    # Fast calibration on snapshot2 (warm-start)
    strikes2, market_vols2, F2, T2 = load_and_prepare(snapshot2)
    fast_params = calibrate_sabr_fast(strikes2, market_vols2, F2, T2, full_params)
    model_vols2 = np.array([sabr_vol_normal(F2, K, T2, *fast_params) for K in strikes2])

    # Plot IV smile comparison
    plot_smile_comparison(strikes2, market_vols2, model_vols2, 'Fast Calibration Smile')

    # Compute and plot RND
    market_rnd2 = compute_market_rnd(strikes2, F2, T2, market_vols2)
    model_rnd2  = compute_rnd(strikes2, F2, T2, *fast_params)
    plot_rnd_comparison(strikes2, market_rnd2, model_rnd2, 'Fast Calibration RND')

    # Print error metrics
    print(f"Fast calibration IV RMSE: {rmse(market_vols2, model_vols2):.6f}")
    print(f"Fast calibration RND RMSE: {rmse(market_rnd2, model_rnd2):.6f}")
