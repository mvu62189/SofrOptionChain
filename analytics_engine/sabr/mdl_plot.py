# plot_iv.py
# This module plot IV and RND

import matplotlib.pyplot as plt
import os

def plot_vol_smile(results: dict, vol_visible: dict) -> plt.Figure:
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
        ax.plot(res['strikes'], res['model_iv_manual'], marker='s', linestyle=':', label=f"Manual ({label})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("IV")
    ax.set_title("Volatility Smile (OTM)")
    ax.legend()
    return fig

def plot_rnd(results: dict, rnd_visible: dict) -> plt.Figure:
    """
    Create a Risk-Neutral Density plot from `results`.
    `rnd_visible[fname]` toggles visibility for each file.
    """
    fig, ax = plt.subplots()
    for fname, res in results.items():
        if not res or not rnd_visible.get(fname):
            continue
        label = os.path.basename(fname)
        ax.plot(res['strikes'], res['rnd_market'], marker='o', linestyle='-', label=f"Market RND ({label})")
        ax.plot(res['strikes'], res['rnd_sabr'],   marker='x', linestyle='--', label=f"SABR RND ({label})")
        if res.get('rnd_manual') is not None:
            ax.plot(res['strikes'], res['rnd_manual'], marker='s', linestyle=':', label=f"Manual RND ({label})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("RND (normalized)")
    ax.set_title("Risk-Neutral Density (RND)")
    ax.legend()
    return fig