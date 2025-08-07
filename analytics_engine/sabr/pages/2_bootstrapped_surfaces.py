# pages/3_Bootstrapped_Market_Surface.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata, interp1d
import os

# Import necessary functions from your existing modules
from mdl_load import discover_snapshot_files
from mdl_run import process_snapshot_file  # We reuse the main processing function
from sabr_v2 import sabr_vol_lognormal
from mdl_rnd_utils import model_rnd

st.set_page_config(layout="wide")
st.title("Bootstrapped & Modeled Surfaces")
st.info(
    "This page constructs two surfaces: one bootstrapped directly from market data, and a second "
    "derived from an interpolation of the SABR model parameters across expiries."
)



# --- CACHE THE ENTIRE DATA PREPARATION STEP ---
@st.cache_data(show_spinner="Processing and calibrating selected files...")
def prepare_surface_data(file_paths, plot_type):
    """
    Takes a list of file paths, processes them, and returns all data needed for plotting.
    This entire block is cached, so it only runs when the list of files changes.
    """
    market_points, model_params_by_T, all_strikes = [], [], []
    for file_path in file_paths:
        res = process_snapshot_file(file_path, manual_params={})
        if res:
            df_temp = pd.read_parquet(file_path)
            snap_dt = pd.to_datetime(df_temp['snapshot_ts'].iloc[0].split(" ")[0])
            expiry_dt = pd.to_datetime(df_temp['expiry_date'].iloc[0])
            T = (expiry_dt.date() - snap_dt.date()).days / 365.0
            strikes = res['strikes']
            z_values = res['market_iv'] if "IV" in plot_type else res['rnd_market']
            valid_mask = ~np.isnan(z_values)
            for i in range(len(strikes[valid_mask])):
                market_points.append([T, strikes[valid_mask][i], z_values[valid_mask][i]])
            all_strikes.extend(strikes)
            if res.get('params_fast'):
                model_params_by_T.append({'T': T, 'F': res['forward_price'], **res['params_fast']})
    
    if not market_points:
        return None
    
    return {
        "market_points": np.array(market_points),
        "model_params_by_T": model_params_by_T,
        "all_strikes": all_strikes
    }

file_dict = discover_snapshot_files("snapshots")

# --- Sidebar for User Controls ---
with st.sidebar:
    st.markdown("### Surface Controls")
    

    def update_shared_folder():
        st.session_state.shared_folder = st.session_state.folder_selector
        # When folder changes, we should clear the multiselect for files
        if 'file_selector' in st.session_state:
            st.session_state.file_selector = []

    all_folders = list(file_dict.keys())
    default_index = 0
    if 'shared_folder' in st.session_state and st.session_state.shared_folder in all_folders:
        default_index = all_folders.index(st.session_state.shared_folder)

    selected_folder = st.selectbox(
        "Select a Snapshot Folder",
        options=all_folders,
        index=default_index,
        key='folder_selector',
        on_change=update_shared_folder
    )
    
    # Make sure the shared state is updated on the first run too
    if 'shared_folder' not in st.session_state and all_folders:
        st.session_state.shared_folder = all_folders[default_index]
    
    # Folder selection
    
    #selected_folder = st.selectbox("Select a Snapshot Folder", options=list(file_dict.keys()))

    files_to_plot = []
    if selected_folder:
        files_in_folder = file_dict.get(selected_folder, [])
        files_to_plot = st.multiselect(
            "Select Chains to Include",
            options=files_in_folder,
            default=files_in_folder,
            format_func=os.path.basename,
            key='file_selector' # Add a key here too
        )

    plot_type = st.radio("Select Surface Type", ("Market Volatility (IV)", "Risk-Neutral Density (RND)"),
        horizontal=True,
    )

    interp_method = st.selectbox("Interpolation Method", ("linear", "cubic", "nearest"),
        index=0,
        help="Method to fill gaps between market data points on the surface."
    )

    # --- Sliders for Slice Projection ---
    #st.markdown("---")
    #st.markdown("### Slice Projection Controls")
    # We will define the slider ranges after data is loaded
    #maturity_to_project = st.slider("Project Skew (at Maturity)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    #strike_to_project = st.slider("Project Term Structure (at Strike)", min_value=90.0, max_value=100.0, value=95.0, step=0.125)

# --- Main Panel for Plotting ---
#if selected_folder:
#    files_in_folder = file_dict.get(selected_folder, [])
#    st.success(f"Processing {len(files_in_folder)} files from folder: **{selected_folder}**")

    # 1. Process all files, collecting both market data and model parameters
    #market_points, model_params_by_T, all_strikes = [], [], []
    #grid_Z_market, grid_Z_model = None, None
    
    #with st.spinner("Loading and processing liquid chains..."):
    #    for file_path in files_in_folder:
    #        res = process_snapshot_file(file_path, manual_params={})
    #        if res:
    #            df_temp = pd.read_parquet(file_path)
    #            snap_dt = pd.to_datetime(df_temp['snapshot_ts'].iloc[0].split(" ")[0])
    #            expiry_dt = pd.to_datetime(df_temp['expiry_date'].iloc[0])
    #            T = (expiry_dt.date() - snap_dt.date()).days / 365.0
    #            
    #            # --- Collect data for Market Surface ---
    #            strikes = res['strikes']
    #            z_values = res['market_iv'] if plot_type == "Market Volatility (IV)" else res['rnd_market']
    #            valid_mask = ~np.isnan(z_values)
    #            for i in range(len(strikes[valid_mask])):
    #                market_points.append([T, strikes[valid_mask][i], z_values[valid_mask][i]])
    #            all_strikes.extend(strikes)
    #            
    #            # --- Collect data for Model Surface --- CHECK AGAIN 'params_fast'
    #            if res.get('params_fast'):
    #                model_params_by_T.append({'T': T, 'F': res['forward_price'], **res['params_fast']})

if files_to_plot:
    # --- Add conditional logic based on number of selected files ---
    if len(files_to_plot) == 1:
        st.info("Displaying the selected 2D curve in 3D space.")
        # We only need to process one file
        data = prepare_surface_data(tuple(files_to_plot), plot_type)
        if data:
            points = data["market_points"]
            T_val = points[0, 0] # Get the single Time-to-Maturity
            
            # Create a 3D line plot
            fig = go.Figure(data=[go.Scatter3d(
                x=[T_val] * len(points), # X is constant maturity
                y=points[:, 1],          # Y is Strike
                z=points[:, 2],          # Z is IV or RND
                mode='lines',
                line=dict(color='cyan', width=5)
            )])
            
            fig.update_layout(
                title=f"Market Curve for {os.path.basename(files_to_plot[0])}",
                height=700,
                scene=dict(
                    xaxis_title='Time to Maturity',
                    yaxis_title='Strike',
                    zaxis_title=plot_type
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
    elif len(files_to_plot) >= 2:
        st.info("Displaying the bootstrapped surface for multiple expiries.")
        surface_data = prepare_surface_data(tuple(files_to_plot), plot_type)

        if not surface_data:
            st.error("No valid data points could be extracted from the files in this folder.")
        else:
            # --- Create grid and interpolate both surfaces ---
            st.subheader("Market Vol Surface")
            points = surface_data["market_points"]
            model_params_by_T = surface_data["model_params_by_T"]
            min_T, max_T = points[:, 0].min(), points[:, 0].max()
            min_K, max_K = min(surface_data["all_strikes"]), max(surface_data["all_strikes"])


            # --- FIX #2: CREATE SLIDERS IN THE SIDEBAR (ONLY AFTER RANGES ARE KNOWN) ---
            with st.sidebar:
                st.markdown("---")
                st.markdown("### Slice Projection Controls")
                T_selected = st.slider("Project Skew (at Maturity)", min_value=min_T, max_value=max_T, value=(min_T + max_T) / 2, step=0.01)
                K_selected = st.slider("Project Term Structure (at Strike)", min_value=min_K, max_value=max_K, value=(min_K + max_K) / 2, step=0.125)

            # --- Interpolation and Plotting ---
            NUM_T_POINTS, NUM_K_POINTS = 100, 100
            grid_T, grid_K = np.mgrid[min_T:max_T:complex(NUM_T_POINTS), min_K:max_K:complex(NUM_K_POINTS)]
            grid_Z_market = griddata(points[:, :2], points[:, 2], (grid_T, grid_K), method=interp_method)

            # --- Create Traces ---
            traces = []
            # --- FIX #1: ADD showlegend=True TO SURFACE TRACES ---
            traces.append(go.Surface(x=grid_T, y=grid_K, z=grid_Z_market, colorscale='viridis', opacity=0.9, name='Market Surface', showlegend=True, showscale=False, visible='legendonly'))

            # Model Surface Logic
            if len(model_params_by_T) >= 2:
                df_params = pd.DataFrame(model_params_by_T).sort_values('T').drop_duplicates('T')
                alpha_interp = interp1d(df_params['T'], df_params['alpha'], kind=interp_method, fill_value="extrapolate")
                beta_interp = interp1d(df_params['T'], df_params['beta'], kind=interp_method, fill_value="extrapolate")
                rho_interp = interp1d(df_params['T'], df_params['rho'], kind=interp_method, fill_value="extrapolate")
                nu_interp = interp1d(df_params['T'], df_params['nu'], kind=interp_method, fill_value="extrapolate")
                F_interp = interp1d(df_params['T'], df_params['F'], kind=interp_method, fill_value="extrapolate")
                
                grid_Z_model = np.zeros_like(grid_T)
                for i in range(grid_T.shape[0]):
                    t = grid_T[i, 0]
                    params = {'alpha': alpha_interp(t), 'beta': beta_interp(t), 'rho': rho_interp(t), 'nu': nu_interp(t)}
                    F, strikes_slice = F_interp(t), grid_K[i, :]
                    grid_Z_model[i, :] = sabr_vol_lognormal(F, strikes_slice, t, **params) if "IV" in plot_type else model_rnd(strikes_slice, F, t, params)
                traces.append(go.Surface(x=grid_T, y=grid_K, z=grid_Z_model, colorscale='plasma', opacity=0.9, name='Model Surface', showlegend=True, showscale=False, visible='legendonly'))
            else:
                st.warning("Model Surface could not be generated. Requires at least 2 files with valid SABR calibrations.")

            # --- 2. CREATE TRACES FOR EACH SURFACE AND DATA TYPE ---
            #traces = []
            
            # Market Surface Trace
            #traces.append(go.Surface(x=grid_T, y=grid_K, z=grid_Z_market, colorscale='viridis', opacity=0.7, name='Market Surface', showscale=False, visible='legendonly'))
            
            # Model Surface Trace (if data exists)
            #if grid_Z_model is not None:
            #    traces.append(go.Surface(x=grid_T, y=grid_K, z=grid_Z_model, colorscale='plasma', opacity=0.7, name='Model Surface', showscale=False))
            
            # Raw Market Data Scatter Trace
            traces.append(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers', marker=dict(size=2, color='green'), name='Raw Market Data'))

            # Projection Traces (based on market data)
            #T_selected = st.session_state.get('maturity_slider', (min_T + max_T) / 2)
            #K_selected = st.session_state.get('strike_slider', (min_K + max_K) / 2)
            skew_strikes = np.linspace(min_K, max_K, NUM_K_POINTS)
            skew_z = griddata(points[:, :2], points[:, 2], (T_selected, skew_strikes), method=interp_method)
            term_maturities = np.linspace(min_T, max_T, NUM_T_POINTS)
            term_z = griddata(points[:, :2], points[:, 2], (term_maturities, K_selected), method=interp_method)
            
            traces.append(go.Scatter3d(x=np.full_like(skew_strikes, max_T), y=skew_strikes, z=skew_z, mode='lines', line=dict(color='cyan', width=5), name=f'Market Skew (T={T_selected:.2f})'))
            traces.append(go.Scatter3d(x=term_maturities, y=np.full_like(term_maturities, min_K), z=term_z, mode='lines', line=dict(color='yellow', width=5), name=f'Market Term (K={K_selected:.2f})'))

            # --- 3. COMBINE ALL TRACES INTO A SINGLE FIGURE ---
            fig = go.Figure(data=traces)

            # --- 4. UPDATE LAYOUT FOR THE COMBINED PLOT ---
            # Convert the list of points into a NumPy array to allow for slicing
            #points = np.array(market_points)
            
            z_min_all = np.nanmin(points[:, 2]) * 0.8
            z_max_all = np.nanmax(points[:, 2]) * 1.2
            
            # --- New Corrected Code ---
            fig.update_layout(
                title="Combined Market vs. Model Surface",
                height=800,
                legend=dict(x=0.8, y=0.95, bgcolor='rgba(0,0,0,0.5)', font=dict(color='white')),
                scene=dict(
                    xaxis=dict(
                        title='Time to Maturity',
                        range=[min_T, max_T * 1.1],
                        showbackground=True, # Show the side wall
                        backgroundcolor='rgba(20, 24, 33, 0.8)'
                    ),
                    yaxis=dict(
                        title='Strike',
                        range=[min_K * 0.99, max_K],
                        showbackground=True, # Show the back wall
                        backgroundcolor='rgba(20, 24, 33, 0.8)'
                    ),
                    zaxis=dict(
                        title=plot_type,
                        range=[z_min_all, z_max_all],
                        showbackground=False # Hide the floor pane
                    ),
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please select a snapshot folder from the sidebar to begin.")