import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

# ------------------- Bachelier IV Solver -------------------
def bachelier_iv(price, forward, strike, tau):
    def objective(sigma):
        d = (forward - strike) / (sigma * np.sqrt(tau))
        return sigma * np.sqrt(tau) * norm.pdf(d) + (forward - strike) * norm.cdf(d) - price

    try:
        return brentq(objective, 1e-6, 10.0)
    except Exception:
        return np.nan

# ------------------- SABR Model -------------------
def sabr_vol(F, K, T, alpha, beta, rho, nu):
    if F == K:
        return alpha / (F ** (1 - beta)) * (1 + ((1 - beta)**2 / 24) * (alpha**2 / (F**(2 - 2*beta))) * T +
                                            (rho * beta * nu * alpha / (4 * F**(1 - beta))) * T +
                                            ((2 - 3 * rho**2) * nu**2 / 24) * T)
    z = (nu / alpha) * (F * K)**((1 - beta) / 2) * np.log(F / K)
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    return (alpha / ((F * K)**((1 - beta) / 2) * (1 + ((1 - beta)**2 / 24) * (np.log(F / K))**2))) * (z / x_z) * \
           (1 + (((1 - beta)**2 / 24) * (alpha**2 / ((F * K)**(1 - beta))) +
                 (rho * beta * nu * alpha / (4 * (F * K)**((1 - beta) / 2))) +
                 ((2 - 3 * rho**2) * nu**2 / 24)) * T)

# ------------------- SABR Fitter -------------------
def fit_sabr(strikes, vols, F, T):
    def loss(params):
        alpha, rho, nu = params
        if not (0 < alpha < 10 and -0.999 < rho < 0.999 and 0 < nu < 10):
            return 1e6
        errors = [((sabr_vol(F, k, T, alpha, 0.5, rho, nu) - iv)**2) for k, iv in zip(strikes, vols)]
        return np.mean(errors)

    init = [0.01, 0.0, 0.5]
    bounds = [(1e-4, 10), (-0.999, 0.999), (1e-4, 10)]
    res = minimize(loss, init, bounds=bounds, method='L-BFGS-B')
    return res.x if res.success else None

# ------------------- Smile Plotter -------------------
def plot_smile(strikes, market_iv, model_iv, title, path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(strikes, market_iv, 'o', label='Market IVM')
    plt.plot(strikes, model_iv, '-', label='SABR Fit')
    plt.xlabel('Strike')
    plt.ylabel('IVM (bp)')
    plt.title(title)
    plt.legend()
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()

# ------------------- Runner -------------------
def run_sabr_on_snapshot(df, expiry, forward, tau, output_dir):
    df = df[(df['expiry'] == expiry) & (df['call_iv_bid'].notna())].copy()
    df['option_price'] = 0.5 * (df['bid'] + df['ask'])
    df['derived_iv'] = df.apply(lambda row: bachelier_iv(
        price=row['option_price'],
        forward=forward,
        strike=row['strike'],
        tau=tau
    ), axis=1)

    valid = df[['strike', 'derived_iv']].dropna()
    if valid.empty:
        print(f"No valid data for expiry {expiry}")
        return

    strikes = valid['strike'].values
    vols = valid['derived_iv'].values

    params = fit_sabr(strikes, vols, forward, tau)
    if params is None:
        print(f"Fitting failed for {expiry}")
        return

    alpha, rho, nu = params
    model_ivs = [sabr_vol(forward, k, tau, alpha, 0.5, rho, nu) for k in strikes]

    os.makedirs(output_dir, exist_ok=True)
    title = f"SABR Fit - {expiry}"
    path = os.path.join(output_dir, f"sabr_fit_{expiry}.png")
    plot_smile(strikes, vols, model_ivs, title, path)

    param_path = os.path.join(output_dir, f"sabr_params_{expiry}.csv")
    pd.DataFrame([{
        'expiry': expiry,
        'alpha': alpha,
        'beta': 0.5,
        'rho': rho,
        'nu': nu,
        'forward': forward,
        'tau': tau,
        'timestamp': df['timestamp'].iloc[0]
    }]).to_csv(param_path, index=False)

    print(f"Finished {expiry} → Params saved to {param_path}")
