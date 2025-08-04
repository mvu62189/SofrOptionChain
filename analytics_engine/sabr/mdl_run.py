# app.py (formerly read_parquet_v3.py)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import hashlib
import matplotlib.pyplot as plt

# Calibration & pricing imports
from iv_utils import implied_vol
from sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast, sabr_vol_normal
from sabr_rnd import price_from_sabr, second_derivative, compute_rnd
from bachelier import bachelier_price

# New modular imports
from mdl_load import discover_snapshot_files, save_uploaded_files
from mdl_calibration import fit_sabr, load_global_beta, calibrate_global_beta, fit_sabr_de
from mdl_rnd_utils import market_rnd, model_rnd
from mdl_plot import plot_vol_smile, plot_rnd
from mdl_snapshot import run_snapshot
try:
    # Attempt to import the real libraries
    from mdl_snapshot import run_snapshot
    import xbbg
    import blpapi
    BLOOMBERG_AVAILABLE = True
except ImportError:
    # If the import fails, set a flag
    BLOOMBERG_AVAILABLE = False
    print("Bloomberg libraries not found. Running in local mode with historical data.")

st.set_page_config(layout="wide", page_title="SOFR Option Chain Diagnostics")
st.title("SOFR Option Chain Diagnostics")

# --- 0. Snapshot Runner ---
st.sidebar.markdown("### Data Snapshots")
if st.sidebar.button("Run New Snapshot", use_container_width=True):
    if BLOOMBERG_AVAILABLE:
        with st.spinner("Running snapshot job... This may take several minutes."):
            try:
                run_snapshot()
                st.sidebar.success("Snapshot job complete!")
                # Clear caches to force rediscovery of files and rerun processing
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.sidebar.error("Snapshot job failed.")
                st.sidebar.exception(e)
    else:
        st.sidebar.error("Bloomberg is not available")
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
    # params_fast, iv_model_fit, debug_data = fit_sabr(strikes_fit, F, T, vols_fit, method='fast')
    params_fast, iv_model_fit, debug_data = fit_sabr_de(strikes_fit, F, T, vols_fit)
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

## --- 7. Debug & parameter tables ---
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
