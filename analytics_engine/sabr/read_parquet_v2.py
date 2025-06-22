import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ---- YOUR UTILITY IMPORTS HERE ----
from iv_utils import implied_vol
from sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast, sabr_vol_normal
from sabr_rnd import price_from_sabr, second_derivative, compute_rnd
from bachelier import bachelier_price

st.set_page_config(layout="wide", page_title="SOFR Option Chain Diagnostics")

st.title("SOFR Option Chain Diagnostics (Multi-Snapshot)")

# --- File management: combine disk and uploaded files ---
st.sidebar.header("Snapshot File Management")
local_files = sorted(glob.glob("snapshots/**/*.parquet", recursive=True))
uploaded_files = st.sidebar.file_uploader(
    "Add Parquet files", type="parquet", accept_multiple_files=True, key="file_upload"
)
uploaded_file_paths = []
for f in uploaded_files:
    # Save uploaded files to temp location for consistent handling
    file_path = os.path.join("uploaded_files", f.name)
    os.makedirs("uploaded_files", exist_ok=True)
    with open(file_path, "wb") as out_f:
        out_f.write(f.read())
    uploaded_file_paths.append(file_path)

all_files = local_files + uploaded_file_paths
if not all_files:
    st.warning("No Parquet files found or uploaded.")
    st.stop()

# --- File selector ---
files_to_show = st.sidebar.multiselect(
    "Choose snapshot files to show", options=all_files, default=all_files[:1], key="file_select"
)
if not files_to_show:
    st.warning("Select at least one file to display charts.")
    st.stop()

# --- Chart visibility checkboxes ---
def get_visibility_state(label, files, default=True):
    key = f"{label}_visible"
    if key not in st.session_state:
        st.session_state[key] = {fname: default for fname in all_files}
    for fname in files:
        st.session_state[key][fname] = st.sidebar.checkbox(
            f"Show {label} ({os.path.basename(fname)})",
            value=st.session_state[key].get(fname, default),
            key=f"{label}_{fname}"
        )
    return st.session_state[key]

vol_visible = get_visibility_state("Vol Smile", files_to_show, default=True)
rnd_visible = get_visibility_state("RND", files_to_show, default=True)

# --- Manual SABR parameter input ---
with st.sidebar.form(key='manual_sabr_form', clear_on_submit=False):
    st.markdown("### Manual SABR Parameters (all files)")
    alpha_in = st.number_input("alpha", min_value=1e-4, max_value=5.0, value=0.1, step=1e-4, format="%.5f")
    beta_in  = st.number_input("beta",  min_value=0.0,   max_value=1.0, value=0.5, step=1e-2,  format="%.2f")
    rho_in   = st.number_input("rho",   min_value=-0.999, max_value=0.999, value=0.0, step=1e-3, format="%.3f")
    nu_in    = st.number_input("nu",    min_value=1e-4, max_value=5.0, value=0.1, step=1e-4, format="%.5f")
    recalibrate = st.form_submit_button(label='Recalibrate')
manual_params = dict(alpha=alpha_in, beta=beta_in, rho=rho_in, nu=nu_in)

# --- Chart refresh controls ---
if "refresh_vol" not in st.session_state:
    st.session_state["refresh_vol"] = False
if "refresh_rnd" not in st.session_state:
    st.session_state["refresh_rnd"] = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Refresh Vol Smile Chart"):
        st.session_state["refresh_vol"] = not st.session_state["refresh_vol"]
with col2:
    if st.button("Refresh RND Chart"):
        st.session_state["refresh_rnd"] = not st.session_state["refresh_rnd"]

# --- Core data processing per snapshot file ---
results = {}

def process_snapshot_file(parquet_path, manual_params=None):
    # 1. Load and basic cleaning
    df_raw = pd.read_parquet(parquet_path)
    df = df_raw.copy()
    df['strike'] = df['ticker'].str.extract(r'\b(\d+\.\d+)\b')[0].astype(float)
    # Identify liquid strikes
    liquid = df[(df.bid>0) & (df.ask>0)]
    if liquid.empty:
        return None
    lo, hi = liquid.strike.min(), liquid.strike.max()
    df_trim = df[(df.strike>=lo) & (df.strike<=hi)].reset_index(drop=True)
    # Compute mid price
    df_trim['mid_price'] = np.where(
        (df_trim.bid>0)&(df_trim.ask>0),
        0.5*(df_trim.bid + df_trim.ask),
        np.nan
    )
    # OTM filter
    df_trim['type'] = df_trim['type'].str.upper()
    F = float(df_trim.future_px.iloc[0])
    df_otm = df_trim[
        ((df_trim['type'] == 'C') & (df_trim['strike'] > F)) |
        ((df_trim['type'] == 'P') & (df_trim['strike'] < F))
    ].reset_index(drop=True)
    if df_otm.empty:
        return None
    # IV inversion (Bachelier)
    snap_dt = datetime.strptime(df_otm.snapshot_ts.iloc[0], '%Y%m%d %H%M%S')
    expiry = pd.to_datetime(df_otm.expiry_date.iloc[0]).date()
    T = (expiry - snap_dt.date()).days / 365.0
    df_otm['iv'] = df_otm.apply(
        lambda r: implied_vol(
            F=float(r.future_px), T=T, K=r.strike, price=r.mid_price, opt_type=r.type, engine='bachelier'
        ) if not np.isnan(r.mid_price) else np.nan, axis=1
    )
    df_otm = df_otm[df_otm.iv < 100]  # drop crazy IVs
    liquid = df_otm[df_otm.volume > 0]
    if liquid.empty:
        return None
    lo = liquid.strike.min()
    hi = liquid.strike.max()
    df_otm = df_otm[(df_otm.strike >= lo-0.052) & (df_otm.strike <= hi+0.052)]
    df_otm['spread'] = df_otm['ask'] - df_otm['bid']
    df_otm = df_otm[df_otm.spread <= 0.012]
    # Prepare arrays
    strikes = df_otm.strike.values
    vols = df_otm.iv.values
    mask = ~np.isnan(vols)
    strikes_fit = strikes[mask]
    vols_fit = vols[mask]
    # SABR calibration (full and fast)
    alpha_f, beta_f, rho_f, nu_f = calibrate_sabr_full(strikes_fit, vols_fit, F, T)
    alpha_fast, beta_fast, rho_fast, nu_fast = calibrate_sabr_fast(
        strikes_fit, vols_fit, F, T, init_params=(alpha_f, beta_f, rho_f, nu_f)
    )
    # Use manual params if provided (from UI)
    if manual_params is not None and recalibrate:
        alpha_use = manual_params['alpha']
        rho_use   = manual_params['rho']
        nu_use    = manual_params['nu']
    else:
        alpha_use = alpha_fast
        rho_use   = rho_fast
        nu_use    = nu_fast
    # Model smiles
    model_iv = sabr_vol_normal(F, strikes, T, alpha_fast, rho_fast, nu_fast)
    model_iv_manual = sabr_vol_normal(F, strikes, T, alpha_use, rho_use, nu_use)
    # Market RND
    market_strikes = strikes
    market_prices = df_otm.mid_price.values
    market_price_func = interp1d(
        market_strikes, market_prices,
        kind='cubic', fill_value="extrapolate", bounds_error=False
    )
    rnd_market = np.array([max(0, second_derivative(market_price_func, K)) for K in market_strikes])
    # SABR RND (use fast calibration params)
    rnd_sabr = compute_rnd(
        strikes=market_strikes, F=F, T=T,
        alpha=alpha_fast, rho=rho_fast, nu=nu_fast
    )
    # Normalize densities
    if np.sum(rnd_sabr) > 0:
        rnd_sabr = rnd_sabr / np.trapezoid(rnd_sabr, market_strikes)
    if np.sum(rnd_market) > 0:
        rnd_market = rnd_market / np.trapezoid(rnd_market, market_strikes)
    # Sort all by strikes
    idx_sort = np.argsort(strikes)
    out = {
        'strikes': strikes[idx_sort],
        'market_iv': vols[idx_sort],
        'model_iv': model_iv[idx_sort],
        'model_iv_manual': model_iv_manual[idx_sort],
        'rnd_market': rnd_market[idx_sort],
        'rnd_sabr': rnd_sabr[idx_sort],
        'params': dict(
            alpha_f=alpha_f, rho_f=rho_f, nu_f=nu_f,
            alpha_fast=alpha_fast, rho_fast=rho_fast, nu_fast=nu_fast,
            alpha_manual=alpha_use, rho_manual=rho_use, nu_manual=nu_use,
        ),
        'T': T, 'F': F, 'fname': parquet_path,
    }
    return out

for fname in files_to_show:
    try:
        results[fname] = process_snapshot_file(fname, manual_params)
    except Exception as e:
        st.warning(f"Failed to process {fname}: {e}")

# --- Plot: Vol Smile (refresh only when user requests) ---
if st.session_state["refresh_vol"]:
    fig, ax = plt.subplots()
    for fname, res in results.items():
        if not res or not vol_visible[fname]: continue
        ax.plot(res['strikes'], res['market_iv'], label=f"Market ({os.path.basename(fname)})")
        ax.plot(res['strikes'], res['model_iv'], '--', label=f"SABR Model ({os.path.basename(fname)})")
        ax.plot(res['strikes'], res['model_iv_manual'], ':', label=f"Manual Model ({os.path.basename(fname)})")
    ax.legend()
    ax.set_xlabel("Strike")
    ax.set_ylabel("IV")
    ax.set_title("Volatility Smile (OTM)")
    st.pyplot(fig, clear_figure=True)

# --- Plot: RND (refresh only when user requests) ---
if st.session_state["refresh_rnd"]:
    fig2, ax2 = plt.subplots()
    for fname, res in results.items():
        if not res or not rnd_visible[fname]: continue
        ax2.plot(res['strikes'], res['rnd_market'], label=f"Market RND ({os.path.basename(fname)})")
        ax2.plot(res['strikes'], res['rnd_sabr'], '--', label=f"SABR RND ({os.path.basename(fname)})")
    ax2.legend()
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("RND (normalized)")
    ax2.set_title("Risk-Neutral Density (RND): Market vs SABR")
    st.pyplot(fig2, clear_figure=True)

# --- Optional: Show parameters table for all snapshots ---
if st.checkbox("Show SABR Calibration Parameters Table"):
    st.markdown("### SABR Parameters (per snapshot)")
    param_table = []
    for fname, res in results.items():
        if not res: continue
        d = res['params']
        d.update({'file': os.path.basename(fname)})
        param_table.append(d)
    if param_table:
        st.dataframe(pd.DataFrame(param_table).set_index('file'))
