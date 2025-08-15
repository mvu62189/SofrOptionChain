# 1_RND.py (formerly mdl_run.py)

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# Calibration & pricing imports
from iv_utils   import implied_vol
from sabr_v2    import calibrate_sabr_full, calibrate_sabr_fast, sabr_vol_normal
from sabr_rnd   import price_from_sabr, second_derivative, compute_rnd
from bachelier  import bachelier_price

# New modular imports
from mdl_load       import discover_snapshot_files, save_uploaded_files
from mdl_calibration import fit_sabr, load_global_beta, calibrate_global_beta, fit_sabr_de
from mdl_rnd_utils  import market_rnd, model_rnd
from mdl_plot       import plot_vol_smile, plot_rnd
from mdl_snapshot   import run_snapshot
from mdl_processing import process_snapshot_file
try:
    # Attempt to import the libraries       # For non-bloomberg stations
    from mdl_snapshot import run_snapshot
    import xbbg
    import blpapi
    BLOOMBERG_AVAILABLE = True
except ImportError:
    # If the import fails, set a flag
    BLOOMBERG_AVAILABLE = False
    print("Bloomberg libraries not found. Running in local mode with historical data.")


st.set_page_config(layout="wide", page_title="SABR Calibration")


# --- ADD MODEL ENGINE TOGGLE IN SIDEBAR ---
st.sidebar.markdown("### Model Engine")
# Parse the selection to get key 'black76' or 'bachelier'
model_engine_display = st.sidebar.radio("Select Pricing Model",
                                        ('Black-76 (Lognormal)', 'Bachelier (Normal)'),
                                        horizontal=True,
                                        key     ='model_engine_selector'
                                        )
model_engine = 'bachelier' if 'Bachelier' in model_engine_display else 'black76'            ### REVISIT ### ## LOGIC ##
st.sidebar.markdown("---")

# Container to hold forward prices display
forward_price_container = st.container()

# --- 0. Snapshot Runner ---
st.sidebar.markdown("### Snapshots")
if st.sidebar.button("Pull New Snapshot", use_container_width=True):
    if BLOOMBERG_AVAILABLE:
        with st.spinner("Running snapshot job... "):
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
# --- 1. File selection via loader ---
file_dict = discover_snapshot_files("snapshots")
options_list = list(file_dict.keys())
# Check if a valid default exists in the session state
valid_default_exists = 'shared_folder' in st.session_state and st.session_state.shared_folder in options_list

# Conditionally call the multiselect widget
if valid_default_exists:
    # If a valid default exists, pass it to the widget
    selected_folders = st.sidebar.multiselect("Folders to load (date\time):",
                                                options=options_list,
                                                default=[st.session_state.shared_folder])
else:
    # If no valid default, call the widget WITHOUT the default argument
    selected_folders = st.sidebar.multiselect("Folders to load (date\time):",
                                                options=options_list)

# After selection, update the shared state for other pages to use       ### REVISIT ### ## FEATURE ## ## LOGIC ##
if selected_folders:
    st.session_state.shared_folder = selected_folders[0]
else:
    if 'shared_folder' in st.session_state:
        del st.session_state.shared_folder

if st.sidebar.button("Clear Cache", use_container_width=True):
    process_snapshot_file.clear()
    st.toast("Calibration cache cleared.")
    st.rerun()

s_col1, s_col2 = st.sidebar.columns(2)
with s_col1:
    if st.sidebar.button("Refresh Vol Smile"):
        st.session_state["refresh_vol"] = not st.session_state.get("refresh_vol", True)
with s_col2:
    if st.sidebar.button("Refresh RND"):
        st.session_state["refresh_rnd"] = not st.session_state.get("refresh_rnd", True)

# --- 1. File selection via modular loader ---
file_dict = discover_snapshot_files("snapshots")
all_files = []
for folder in selected_folders:
    st.sidebar.markdown(f"**{folder}/**")
    files  = file_dict.get(folder, [])
    chosen = st.sidebar.multiselect(label=f"Expiries in {folder}/",
                                options=files, default=[], key=folder,
                                format_func=os.path.basename # This line formats the display    ### REVISIT ### ## UI ##
                                    )
    all_files.extend(chosen)

uploaded = save_uploaded_files(st.sidebar.file_uploader(
                            "Or add Parquet files", type="parquet", accept_multiple_files=True))

files_to_show = all_files + uploaded
if not files_to_show:
    st.warning("No files selected or uploaded.")
    st.stop()


### --- 2. Manual SABR Calibration (one file at a time) ---
with st.sidebar.form(key='manual_sabr_form', clear_on_submit=False):
    st.markdown("### Manual SABR Calibration")
    # a. choose one file
    manual_file = st.selectbox("File to recalibrate",
                                options     =files_to_show,
                                format_func =lambda f: os.path.basename(f))
    
    st.markdown("#### Parameter inputs")
    alpha_in = st.number_input("alpha", 
                               min_value=1e-4, max_value=5.0,
                                value=0.1, step=1e-4, format="%.5f")
    beta_in = st.number_input("beta",
                              min_value=0.0, max_value=1.0,
                              value=0.5, step=1e-4, format="%.5f")
    rho_in = st.number_input("rho",
                             min_value=-0.99999, max_value=0.99999,
                             value=0.0, step=1e-5, format="%.5f")
    nu_in = st.number_input("nu",
                            min_value=1e-4, max_value=5.0,
                            value=0.1, step=1e-4, format="%.5f")
    
    recalibrate = st.form_submit_button("Recalibrate")
manual_params = dict(alpha=alpha_in, beta=beta_in, rho=rho_in, nu=nu_in, recalibrate=recalibrate)
st.session_state['manual_file'] = manual_file


## HISTORICAL BETA BUTTON                           ### REVISIT ### ## FEATURE ##

st.sidebar.markdown("---")
st.sidebar.markdown("### Historical Beta")

beta_col1, beta_col2 = st.sidebar.columns(2)
with beta_col1:
    if st.sidebar.button("Calibrate Global β"):
        with st.spinner("Optimizing β across selected snapshots…"):
            β_opt = calibrate_global_beta(files_to_show)
            st.success(f"Global β optimized: {β_opt:.4f}")
            process_snapshot_file.clear()
            st.warning("Cache cleared. Refresh (F5) to apply new β.")
            st.stop()

with beta_col2:
    st.metric("Current β", f"{load_global_beta():.4f}")

st.sidebar.markdown("---")


####################################################
# --- 3. Visibility Toggles in Side Bar---
####################################################
# --- Volatility Smile Section ---
st.sidebar.markdown("### Volatility Smile Options")

# Initialize session state for vol visibility if it doesn't exist
if 'vol_visible' not in st.session_state:
    st.session_state['vol_visible'] = {f: True for f in files_to_show}

# Create file-specific toggles and build the visibility dictionary
vol_visible = {}
for f in files_to_show:
    # Create the desired label: yyyymmdd/hhmmss/filename.parquet                    ### REVISIT ### ## UI ## ## LOGIC ##
    parts = os.path.normpath(f).split(os.sep)
    label = os.path.join(*parts[-3:]) if len(parts) >= 3 else os.path.basename(f)
    
    # Create checkbox and update the dictionary for plot function
    is_visible = st.sidebar.checkbox(label,
                            value=st.session_state['vol_visible'].get(f, True),
                            key=f"vol_{f}") # Unique key for this checkbox
    vol_visible[f] = is_visible
    st.session_state['vol_visible'][f] = is_visible # Save state for reruns
st.sidebar.markdown("---") # Visual separator

# Series toggles for Vol Smile
show_mkt_iv    = st.sidebar.checkbox("Show Market IV",     value=True, key="sidebar_toggle_mkt_iv")
show_model_iv  = st.sidebar.checkbox("Show SABR Model IV", value=True, key="sidebar_toggle_model_iv")
show_manual_iv = st.sidebar.checkbox("Show Manual IV",     value=True, key="sidebar_toggle_manual_iv")
st.sidebar.markdown("---") # Visual separator


# --- Risk-Neutral Density Section ---
st.sidebar.markdown("### Risk-Neutral Density Options")

# Initialize session state for RND visibility if it doesn't exist
if 'rnd_visible' not in st.session_state:
    st.session_state['rnd_visible'] = {f: True for f in files_to_show}

# Create file-specific toggles and build the visibility dictionary
rnd_visible = {}
for f in files_to_show:
    # Create the desired label (re-used logic)                                      ### REVISIT ### ## UI ##
    parts = os.path.normpath(f).split(os.sep)
    label = os.path.join(*parts[-3:]) if len(parts) >= 3 else os.path.basename(f)

    # Create the checkbox and update the dictionary
    is_visible = st.sidebar.checkbox(label,
                            value=st.session_state['rnd_visible'].get(f, True),
                            key=f"rnd_{f}") # Unique key for this checkbox    
    rnd_visible[f] = is_visible
    st.session_state['rnd_visible'][f] = is_visible # Save state for reruns
st.sidebar.markdown("---") # Visual separator

# Series toggles for RND
show_mkt_rnd    = st.sidebar.checkbox("Show Market RND", value=True, key="sidebar_toggle_mkt_rnd")
show_model_rnd  = st.sidebar.checkbox("Show SABR RND",   value=True, key="sidebar_toggle_model_rnd")
show_manual_rnd = st.sidebar.checkbox("Show Manual RND", value=False, key="sidebar_toggle_manual_rnd")
##############################################################
# End of sidebar UI block
###############################################################

# This loop unpacks the results and logs failures
results = {}
skipped_files_log = defaultdict(list)

for f in files_to_show:
    # Unpack the two return values from the function
    result_dict, reason = process_snapshot_file(f, manual_params, model_engine=model_engine)
    
    if reason is None:
        # On success, add the dictionary to the results
        results[f] = result_dict
    else:
        # On failure, log the reason and the file path
        path_parts = os.path.normpath(f).split(os.sep)
        source_name = os.path.join(*path_parts[-3:])
        skipped_files_log[reason].append(source_name)

# --- (Optional but Recommended) Add an expander to show skipped files ---
if skipped_files_log:
    with st.expander("Skipped Files Log"):
        for reason, files in skipped_files_log.items():
            st.subheader(reason)
            st.json(files)


# --- Display Currently Loaded Chains and Forward Prices ---
with forward_price_container:
    st.markdown("### Currently Loaded Chains")
    
    # Filter for results that have a forward price to display
    loaded_files = [
        (fname, res) for fname, res in results.items() 
        if res and res.get('forward_price')
    ]
    
    if loaded_files:
        # Create a header for the list
        col1, col2 = st.columns([3, 1])
        col1.markdown("**Chain Path**")
        col2.markdown("**Forward Price**")

        # Loop through the loaded files and display each on its own line
        for fname, res in loaded_files:
            # Create the descriptive label: yyyymmdd/hhmmss/filename.parquet                        ### REVISIT ### ## UI ##
            parts = os.path.normpath(fname).split(os.sep)
            label = os.path.join(*parts[-3:]) if len(parts) >= 3 else os.path.basename(fname)
            
            fwd_price = res['forward_price']
            
            # Use two columns for clean alignment on each line
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"`{label}`") # Use code formatting for the path
            with col2:
                st.markdown(f"**`{fwd_price:.4f}`**") # Use bold code formatting for the price
    else:
        st.info("No files loaded to display forward prices.")
    
    st.markdown("---")

# --- 6. Plot via plotting module ---
if st.session_state.get("refresh_vol", True):
    fig = plot_vol_smile(results, vol_visible, show_mkt_iv, show_model_iv, show_manual_iv)
    st.pyplot(fig, clear_figure=True)


if st.session_state.get("refresh_rnd", True):
    fig2 = plot_rnd(results, rnd_visible, show_mkt_rnd, show_model_rnd, show_manual_rnd)
    st.pyplot(fig2, clear_figure=True)

## --- 7. Debug & parameter tables ---
with st.expander("Debug 2.0: Snapshot Data & Params"):
    for f, res in results.items():
        if not res: continue
        st.markdown(f"**{os.path.basename(f)}**")                       ### REVISIT ### ## UI ##
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
            
            raw_strikes     = res['strikes']
            raw_vols        = res['market_iv']
            debug_info      = res['debug_data']
            interp_strikes  = debug_info['interp_strikes']
            interp_vols     = debug_info['interp_vols']
            
            fig_debug, ax_debug = plt.subplots()
            ax_debug.plot(interp_strikes, interp_vols, 'r-', label="Interpolated Curve (Calibration Target)")
            ax_debug.plot(raw_strikes, raw_vols, 'bo', label="Raw Market IV Points", markersize=5)
            
            ax_debug.set_title("Interpolation Sanity Check")
            ax_debug.set_xlabel("Strike")
            ax_debug.set_ylabel("Implied Volatility")
            ax_debug.legend()
            ax_debug.grid(True)
            
            st.pyplot(fig_debug, clear_figure=True)
