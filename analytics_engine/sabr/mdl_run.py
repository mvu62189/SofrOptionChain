# NOTE: NEED TO ADJUST PARQUET FILE NAMES WITH DATES ??
# SO 1 CHAIN CAN BE UPLOADED FOR BOTH DATES ??

# ** OR UPDATE FILE SELECTION PROCESS TO TIME SELECTION & CHAIN SELECTION FILTERING **

# app.py (formerly read_parquet_v3.py)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import hashlib
import matplotlib.pyplot as plt

# Calibration & pricing imports remain unchanged
from iv_utils import implied_vol
from sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast, sabr_vol_normal
from sabr_rnd import price_from_sabr, second_derivative, compute_rnd
from bachelier import bachelier_price

# New modular imports
from mdl_load import discover_snapshot_files, save_uploaded_files
from mdl_calibration import fit_sabr
from mdl_rnd_utils import market_rnd, model_rnd
from mdl_plot import plot_vol_smile, plot_rnd

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

# --- 2. Manual SABR parameters (unchanged) ---
#with st.sidebar.form(key='manual_sabr_form', clear_on_submit=False):
#    st.markdown("### Manual SABR Parameters (for all files)")
#    alpha_in = st.number_input("alpha", min_value=1e-4, max_value=5.0,
#                               value=0.1, step=1e-4, format="%.5f")
#    beta_in  = st.number_input("beta",  min_value=0.0, max_value=1.0,
#                               value=0.5, step=1e-2, format="%.5f")
#    rho_in   = st.number_input("rho",   min_value=-0.999, max_value=0.999,
#                               value=0.0, step=1e-3, format="%.5f")
#    nu_in    = st.number_input("nu",    min_value=1e-4, max_value=5.0,
#                               value=0.1, step=1e-4, format="%.5f")
#    recalibrate = st.form_submit_button(label='Recalibrate')
#manual_params = dict(alpha=alpha_in, beta=beta_in, rho=rho_in, nu=nu_in)

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
        unique_key = f"{label}_" + hashlib.md5(file_path.encode()).hexdigest()
        st.session_state[key][file_path] = st.sidebar.checkbox(
            f"Show {label} ({basename})",
            value=st.session_state[key].get(file_path, default),
            # key=f"{label}_" + hashlib.md5(file_path.encode()).hexdigest()
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

# --- 5. Process each file (caching untouched) ---
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
    df_otm['iv'] = df_otm.apply(lambda r: implied_vol(F=float(r['future_px']), T=T, K=r['strike'], price=r['mid_price'], opt_type=r['type'], engine='bachelier') if not np.isnan(r['mid_price']) else np.nan, axis=1)
    df_otm = df_otm[df_otm['iv']<100]
    liquid2 = df_otm[df_otm['volume']>0]
    if liquid2.empty: return None
    lo2, hi2 = liquid2['strike'].min(), liquid2['strike'].max()
    df_otm = df_otm[(df_otm['strike']>=lo2-0.052)&(df_otm['strike']<=hi2+0.052)]
    df_otm['spread'] = df_otm['ask'] - df_otm['bid']
    df_otm = df_otm[df_otm['spread']<=0.012]
    strikes = df_otm['strike'].values
    market_iv = df_otm['iv'].values
    mask = ~np.isnan(market_iv)
    fit_order = np.argsort(strikes[mask])
    strikes_fit = strikes[fit_order]
    vols_fit = market_iv[fit_order]
    params_fast, iv_model = fit_sabr(strikes_fit, F, T, vols_fit, method='fast')
    params_man, iv_manual = (None, None)
    if recalibrate and st.session_state.get('manual_file')==parquet_path:
        params_man, iv_manual = fit_sabr(strikes_fit, F, T, vols_fit, method='fast', manual_params=manual_params)
    model_iv = iv_model 
    mid_prices = df_otm['mid_price'].values
    rnd_mkt = market_rnd(strikes, mid_prices)
    rnd_sabr = model_rnd(strikes, F, T, params_fast)
    rnd_man  = model_rnd(strikes, F, T, params_man) if params_man else None

    # Compute integrals for debug (should be ≈1)
    area_market = float(np.trapezoid(rnd_mkt,    strikes))
    area_model  = float(np.trapezoid(rnd_sabr,   strikes))

    print("rnd_sabr shape:", rnd_sabr.shape)
    return {'strikes': strikes, 'market_iv': market_iv, 'model_iv': model_iv, 'iv_manual': iv_manual, 'rnd_market': rnd_mkt,
             'rnd_sabr': rnd_sabr, 'rnd_manual': rnd_man, 'params_fast': params_fast, 'params_manual': params_man,
              'mid_prices': mid_prices, 'area_model': area_model, 'area_market': area_market
            }
    
results = {f: process_snapshot_file(f, manual_params) for f in files_to_show}


# --- 6. Plot via plotting module ---
#if st.session_state.get("refresh_vol", True):
#    fig = plot_vol_smile(results, vol_visible)
#    st.pyplot(fig, clear_figure=True)

if st.session_state.get("refresh_vol", True):
    # 4. Series-level toggles for Vol Smile
    show_mkt_iv    = st.checkbox("Show Market IV",    value=True, key="toggle_mkt_iv")
    show_model_iv  = st.checkbox("Show SABR Model IV", value=True, key="toggle_model_iv")
    show_manual_iv = st.checkbox("Show Manual IV",     value=False, key="toggle_manual_iv")

    fig, ax = plt.subplots()
    for fname, res in results.items():
        if not res: continue
        label = os.path.basename(fname)
        if show_mkt_iv:
            ax.plot(res['strikes'], res['market_iv'],
                    marker='o', linestyle='-', label=f"Market ({label})")
        if show_model_iv:
            ax.plot(res['strikes'], res['model_iv'],
                    marker='o', linestyle='--', label=f"SABR ({label})")
        if show_manual_iv and res.get('model_iv_manual') is not None:
            ax.plot(res['strikes'], res['model_iv_manual'],
                    marker='o', linestyle=':', label=f"Manual ({label})")
    ax.legend()
    ax.set_xlabel("Strike")
    ax.set_ylabel("IV")
    ax.set_title("Volatility Smile (OTM)")
    st.pyplot(fig, clear_figure=True)

#if st.session_state.get("refresh_rnd", True):
#    fig2 = plot_rnd(results, rnd_visible)
#    st.pyplot(fig2, clear_figure=True)

if st.session_state.get("refresh_rnd", True):
    # 4. Series-level toggles for RND
    show_mkt_rnd    = st.checkbox("Show Market RND",    value=True,   key="toggle_mkt_rnd")
    show_model_rnd  = st.checkbox("Show SABR RND",      value=True,   key="toggle_model_rnd")
    show_manual_rnd = st.checkbox("Show Manual RND",    value=False,  key="toggle_manual_rnd")

    fig2, ax2 = plt.subplots()
    for fname, res in results.items():
        if not res: continue
        label = os.path.basename(fname)
        if show_mkt_rnd:
            ax2.plot(res['strikes'], res['rnd_market'],
                     marker='o', linestyle='-', label=f"Mkt RND ({label})")
        if show_model_rnd:
            ax2.plot(res['strikes'], res['rnd_sabr'],
                     marker='o', linestyle='--', label=f"SABR RND ({label})")
        if show_manual_rnd and res.get('rnd_manual') is not None:
            ax2.plot(res['strikes'], res['rnd_manual'],
                     marker='o', linestyle=':', label=f"Manual RND ({label})")
    ax2.legend()
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("RND (normalized)")
    ax2.set_title("Risk-Neutral Density (RND): Market vs SABR")
    st.pyplot(fig2, clear_figure=True)

# --- 7. Debug & parameter tables (unchanged) ---
#with st.expander("Debug: RND Calculation"):
#    for fname, res in results.items():
#        if not res: continue
#        st.markdown(f"**{os.path.basename(fname)}**")
#        st.write(res['debug'])

#with st.expander("SABR Parameters Table (per file, 3x3)"):
#    for fname, res in results.items():
#        if not res: continue
#        st.markdown(f"**{os.path.basename(fname)}**")
#        st.dataframe(res['params_table'])

### ------------ UTILITIES -------------------

# ── Cache-clear button top-right ──

with col_clear:
    if st.button("Clear Cache"):
        process_snapshot_file.clear()
        st.warning("Calibration cache cleared. Refresh (F5) to rerun calibration.")
        st.stop()

# 7. Debug 2.0 info
with st.expander("Debug 2.0: Snapshot Data & Params"):
    for f, res in results.items():
        st.markdown(f"**{os.path.basename(f)}**")
        st.write("Params Fast:", res['params_fast'])
        st.write("Params Manual:", res['params_manual'])
        st.write(res['mid_prices'], res['strikes'], res['rnd_sabr'])
        
        # Debug output: check if strikes are sorted
        debug_strikes = np.array(res['strikes'])
        debug_sorted = np.all(np.diff(debug_strikes) > 0)

        # Market‐ and Model‐RND debug: integral, shape, head/tail
        debug_model_rnd = {
            'integral':      round(res['area_model'], 6),
            'all_nonneg':    bool(np.all(res['rnd_sabr'] >= 0))
        }
        debug_market_rnd = {
            'integral':      round(res['area_market'], 6),
            'all_nonneg':    bool(np.all(res['rnd_market'] >= 0))
        }
        debug_info = {
            'strikes_sorted':  debug_sorted,
            'market_rnd':      debug_market_rnd,
            'model_rnd':       debug_model_rnd
        }
        st.write(debug_info)
