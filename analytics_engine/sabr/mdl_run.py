# app.py (formerly read_parquet_v3.py)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Calibration & pricing imports remain unchanged
from iv_utils import implied_vol
from sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast, sabr_vol_normal
from sabr_rnd import price_from_sabr, second_derivative, compute_rnd
from bachelier import bachelier_price

# New modular imports
from mdl_load import discover_snapshot_files, save_uploaded_files
from mdl_plot import plot_vol_smile, plot_rnd

st.set_page_config(layout="wide", page_title="SOFR Option Chain Diagnostics")
st.title("SOFR Option Chain Diagnostics (Multi-Snapshot, Cached)")

# --- 1. File selection via modular loader ---
file_dict = discover_snapshot_files("snapshots")
selected_folders = st.sidebar.multiselect(
    "Folders to load:", options=list(file_dict.keys()), default=[]
)

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
with st.sidebar.form(key='manual_sabr_form', clear_on_submit=False):
    st.markdown("### Manual SABR Parameters (for all files)")
    alpha_in = st.number_input("alpha", min_value=1e-4, max_value=5.0,
                               value=0.1, step=1e-4, format="%.5f")
    beta_in  = st.number_input("beta",  min_value=0.0, max_value=1.0,
                               value=0.5, step=1e-2, format="%.2f")
    rho_in   = st.number_input("rho",   min_value=-0.999, max_value=0.999,
                               value=0.0, step=1e-3, format="%.3f")
    nu_in    = st.number_input("nu",    min_value=1e-4, max_value=5.0,
                               value=0.1, step=1e-4, format="%.5f")
    recalibrate = st.form_submit_button(label='Recalibrate')
manual_params = dict(alpha=alpha_in, beta=beta_in, rho=rho_in, nu=nu_in)

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
            key=f"{label}_{basename}"
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
    # (existing snapshot + SABR + RND logic remains here unchanged)
    ...  # placeholder for calibration code

results = {f: process_snapshot_file(f, manual_params) for f in files_to_show}

# --- 6. Plot via plotting module ---
if st.session_state.get("refresh_vol", True):
    fig = plot_vol_smile(results, vol_visible)
    st.pyplot(fig, clear_figure=True)

if st.session_state.get("refresh_rnd", True):
    fig2 = plot_rnd(results, rnd_visible)
    st.pyplot(fig2, clear_figure=True)

# --- 7. Debug & parameter tables (unchanged) ---
with st.expander("Debug: RND Calculation"):
    for fname, res in results.items():
        if not res: continue
        st.markdown(f"**{os.path.basename(fname)}**")
        st.write(res['debug'])

with st.expander("SABR Parameters Table (per file, 3x3)"):
    for fname, res in results.items():
        if not res: continue
        st.markdown(f"**{os.path.basename(fname)}**")
        st.dataframe(res['params_table'])
