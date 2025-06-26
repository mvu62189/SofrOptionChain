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