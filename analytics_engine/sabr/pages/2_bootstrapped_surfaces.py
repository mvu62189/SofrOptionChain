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

# --- Sidebar for User Controls ---
with st.sidebar:
    st.markdown("### Surface Controls")

    # Folder selection
    file_dict = discover_snapshot_files("snapshots")
    selected_folder = st.selectbox(
        "Select a Snapshot Folder",
        options=list(file_dict.keys())
    )

    plot_type = st.radio(
        "Select Surface Type",
        ("Market Volatility (IV)", "Risk-Neutral Density (RND)"),
        horizontal=True,
    )

    interp_method = st.selectbox(
        "Interpolation Method",
        ("linear", "cubic", "nearest"),
        index=0,
        help="Method to fill gaps between market data points on the surface."
    )

    # --- Sliders for Slice Projection ---
    st.markdown("---")
    st.markdown("### Slice Projection Controls")
    # We will define the slider ranges after data is loaded
    maturity_to_project = st.slider("Project Skew (at Maturity)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    strike_to_project = st.slider("Project Term Structure (at Strike)", min_value=90.0, max_value=100.0, value=95.0, step=0.125)

# --- Main Panel for Plotting ---
if selected_folder:
    files_in_folder = file_dict.get(selected_folder, [])
    st.success(f"Processing {len(files_in_folder)} files from folder: **{selected_folder}**")

    # 1. Process all files, collecting both market data and model parameters
    market_points = []
    model_params_by_T = []
    all_strikes = []
    
    with st.spinner("Loading and processing all snapshot files..."):
        for file_path in files_in_folder:
            res = process_snapshot_file(file_path, manual_params={})
            if res:
                df_temp = pd.read_parquet(file_path)
                snap_dt = pd.to_datetime(df_temp['snapshot_ts'].iloc[0].split(" ")[0])
                expiry_dt = pd.to_datetime(df_temp['expiry_date'].iloc[0])
                T = (expiry_dt.date() - snap_dt.date()).days / 365.0
                
                # --- Collect data for Market Surface ---
                strikes = res['strikes']
                z_values = res['market_iv'] if plot_type == "Market Volatility (IV)" else res['rnd_market']
                valid_mask = ~np.isnan(z_values)
                for i in range(len(strikes[valid_mask])):
                    market_points.append([T, strikes[valid_mask][i], z_values[valid_mask][i]])
                all_strikes.extend(strikes)
                
                # --- Collect data for Model Surface ---
                if res.get('params_fast'):
                    model_params_by_T.append({'T': T, 'F': res['forward_price'], **res['params_fast']})

    if not market_points:
        st.error("No valid data points could be extracted from the files in this folder.")
    else:
        # --- PLOT 1: BOOTSTRAPPED MARKET SURFACE ---
        st.subheader("Bootstrapped Market Surface")
        points = np.array(market_points)
        min_T, max_T = points[:, 0].min(), points[:, 0].max()
        min_K, max_K = min(all_strikes), max(all_strikes)

        # --- Update slider ranges based on loaded data ---
        st.sidebar.slider("Project Skew (at Maturity)", min_value=min_T, max_value=max_T, value=(min_T + max_T) / 2, step=0.01, key="maturity_slider")
        st.sidebar.slider("Project Term Structure (at Strike)", min_value=min_K, max_value=max_K, value=(min_K + max_K) / 2, step=0.125, key="strike_slider")
        T_selected = st.session_state.maturity_slider
        K_selected = st.session_state.strike_slider

        NUM_T_POINTS = 100
        NUM_K_POINTS = 100

        grid_T, grid_K = np.mgrid[min_T:max_T:complex(NUM_T_POINTS), min_K:max_K:complex(NUM_K_POINTS)]
        
        with st.spinner(f"Interpolating market surface using '{interp_method}' method..."):
            grid_Z_market = griddata(points[:, :2], points[:, 2], (grid_T, grid_K), method=interp_method)
        
        z_axis_title = "Market IV" if plot_type == "Market Volatility (IV)" else "Market RND"
        title = "Bootstrapped Market Surface with Raw Data Points"

        # --- 1. CURVE PROJECTION LOGIC ---
        # Get data for the selected maturity slice (skew)
        skew_strikes = np.linspace(min_K, max_K, NUM_K_POINTS)
        skew_z = griddata(points[:, :2], points[:, 2], (T_selected, skew_strikes), method=interp_method)
        # Get data for the selected strike slice (term structure)
        term_maturities = np.linspace(min_T, max_T, NUM_T_POINTS)
        term_z = griddata(points[:, :2], points[:, 2], (term_maturities, K_selected), method=interp_method)


        # Create the smooth surface trace
        surface_trace = go.Surface(
            x=grid_T, y=grid_K, z=grid_Z_market, 
            colorscale='viridis', 
            opacity=0.7,
            colorbar=dict(title=z_axis_title), 
            cmin=np.nanmin(grid_Z_market), cmax=np.nanmax(grid_Z_market),
        
            # --- ADD INTERACTIVE CONTOUR HIGHLIGHTING ---
            contours={
                "x": {"show": True, "start": min_T, "end": max_T, "size": 0.1, "color": "white", "highlight": True, "highlightcolor": "yellow"},
                "y": {"show": True, "start": min_K, "end": max_K, "size": 1, "color": "white", "highlight": True, "highlightcolor": "cyan"}
            }
        )

        # Create the raw data scatter trace to overlay
        scatter_trace = go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
            marker=dict(size=2.5, color='red', opacity=0.6), name='Raw Market Data'
        )

        # Create traces for the projected curves on the walls
        skew_proj_trace = go.Scatter3d(x=np.full_like(skew_strikes, max_T), y=skew_strikes, z=skew_z, mode='lines', line=dict(color='cyan', width=5), name=f'Skew at T={T_selected:.2f}')
        term_proj_trace = go.Scatter3d(x=term_maturities, y=np.full_like(term_maturities, min_K), z=term_z, mode='lines', line=dict(color='yellow', width=5), name=f'Term at K={K_selected:.2f}')

        market_fig = go.Figure(data=[surface_trace, scatter_trace, skew_proj_trace, term_proj_trace])

        # --- 2. FLOATING LAYOUT ADJUSTMENTS ---
        z_min_display = np.nanmin(grid_Z_market) * 0.8 # Create a gap below the surface
        market_fig.update_layout(
            title=title, 
            height=800,
            showlegend=True,
            legend=dict(x=0.8, y=0.9),
            scene=dict(
                xaxis=dict(
                    title='Time to Maturity (Years)', range=[min_T, max_T * 1.1],
                    showbackground=True, backgroundcolor='rgba(20, 24, 33, 0.8)',
                    gridcolor='rgba(128, 128, 128, 0.5)',
                    zerolinecolor='gray'
                ),
                yaxis=dict(
                    title='Strike Price', range=[min_K * 0.99, max_K],
                    showbackground=True, backgroundcolor='rgba(20, 24, 33, 0.8)',
                    gridcolor='rgba(128, 128, 128, 0.5)',
                    zerolinecolor='gray'
                ),
                zaxis=dict(
                    title=z_axis_title, range=[z_min_display, np.nanmax(grid_Z_market)],
                    showbackground=False, backgroundcolor='rgba(20, 24, 33, 0.8)',
                    gridcolor='rgba(128, 128, 128, 0.5)',
                    zerolinecolor='gray',
                ),
                # Style the axis panes (walls)
                #xaxis_pane=dict(visible=False),
                #yaxis_pane=dict(visible=False)
            ),
            margin=dict(l=10, r=10, b=10, t=80)
        )

        st.plotly_chart(market_fig, use_container_width=True)

        st.markdown("---")

        # --- PLOT 2: INTERPOLATED MODEL SURFACE ---
        st.subheader("Interpolated SABR Model Surface")
        if len(model_params_by_T) < 2:
            st.warning("Cannot create a model surface. Need at least 2 files with valid SABR parameters to interpolate.")
        else:
            with st.spinner("Interpolating SABR parameters and building model surface..."):
                df_params = pd.DataFrame(model_params_by_T).sort_values('T').drop_duplicates('T')
                
                # Create an interpolator for each SABR parameter vs. Time
                alpha_interp = interp1d(df_params['T'], df_params['alpha'], kind=interp_method, fill_value="extrapolate")
                beta_interp = interp1d(df_params['T'], df_params['beta'], kind=interp_method, fill_value="extrapolate")
                rho_interp = interp1d(df_params['T'], df_params['rho'], kind=interp_method, fill_value="extrapolate")
                nu_interp = interp1d(df_params['T'], df_params['nu'], kind=interp_method, fill_value="extrapolate")
                F_interp = interp1d(df_params['T'], df_params['F'], kind=interp_method, fill_value="extrapolate")
                
                # Calculate Z-values on the grid using the interpolated parameters
                grid_Z_model = np.zeros_like(grid_T)
                for i in range(grid_T.shape[0]): # Iterate over maturities
                    t = grid_T[i, 0]
                    params = {'alpha': alpha_interp(t), 'beta': beta_interp(t), 'rho': rho_interp(t), 'nu': nu_interp(t)}
                    F = F_interp(t)
                    strikes_slice = grid_K[i, :]
                    
                    if plot_type == "Market Volatility (IV)":
                        grid_Z_model[i, :] = sabr_vol_lognormal(F, strikes_slice, t, **params)
                    else: # RND
                        grid_Z_model[i, :] = model_rnd(strikes_slice, F, t, params)

                # --- START of Plotting Enhancements ---
                model_surface_trace = go.Surface(
                    x=grid_T, y=grid_K, z=grid_Z_model,
                    colorscale='plasma',
                    colorbar=dict(title="Model " + z_axis_title.split(" ")[1]),
                    # --- 2. ADD INTERACTIVE CONTOUR HIGHLIGHTING ---
                    contours={
                        "x": {"show": True, "start": min_T, "end": max_T, "size": 0.1, "color": "white", "highlight": True, "highlightcolor": "yellow"},
                        "y": {"show": True, "start": min_K, "end": max_K, "size": 1, "color": "white", "highlight": True, "highlightcolor": "cyan"}
                    }
                )

                model_fig = go.Figure(data=[model_surface_trace])
                
                
                # --- 1. CONFIGURE SCENE (WALLS, GRIDS) ---
                model_fig.update_layout(
                    title="Surface from Interpolated SABR Parameters",
                    height=750,
                    scene=dict(
                        xaxis=dict(
                            title='Time to Maturity (Years)',
                            showbackground=True, backgroundcolor='rgba(20, 24, 33, 0.8)',
                            gridcolor='rgba(128, 128, 128, 0.5)'
                        ),
                        yaxis=dict(
                            title='Strike Price',
                            showbackground=True, backgroundcolor='rgba(20, 24, 33, 0.8)',
                            gridcolor='rgba(128, 128, 128, 0.5)'
                        ),
                        zaxis=dict(
                            title="Model " + z_axis_title.split(" ")[1],
                            showbackground=True, backgroundcolor='rgba(20, 24, 33, 0.8)',
                            gridcolor='rgba(128, 128, 128, 0.5)'
                        )
                    ),
                    margin=dict(l=10, r=10, b=10, t=80)
                )
                # --- END of Plotting Enhancements ---
                st.plotly_chart(model_fig, use_container_width=True)

else:
    st.info("Please select a snapshot folder from the sidebar to begin.")