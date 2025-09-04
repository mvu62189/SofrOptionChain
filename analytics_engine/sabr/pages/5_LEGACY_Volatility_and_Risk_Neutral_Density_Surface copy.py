# pages/2_Surfaces.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata, interp1d
import os


# Import necessary functions from existing modules
from mdl_load       import discover_snapshot_files
from mdl_processing import process_snapshot_file  # reuse the main processing function
from sabr_v2        import sabr_vol_lognormal, sabr_vol_normal
from mdl_rnd_utils  import model_rnd

st.set_page_config(layout="wide")
st.title("IV & RND Surfaces")
st.info(
    "IV and RND each constructs two surfaces: one derived directly from market data, and a second "
    "derived from an interpolation of the SABR model parameters across expiries."
)

# --- CACHE THE DATA PREPARATION STEP ---
@st.cache_data(show_spinner="Processing and calibrating selected files...")
def prepare_surface_data(file_paths, plot_type, model_engine):
    """
    Takes a list of file paths, processes them, and returns all data needed for plotting.
    This entire block is cached, so it only runs when the list of files changes.
    """
    market_points, model_params_by_T, all_strikes = [], [], []
    for file_path in file_paths:
        res, reason     = process_snapshot_file(file_path, manual_params={}, model_engine=model_engine)
        if reason is None and res:
            df_temp     = pd.read_parquet(file_path)
            snap_dt     = pd.to_datetime(df_temp['snapshot_ts'].iloc[0].split(" ")[0])
            expiry_dt   = pd.to_datetime(df_temp['expiry_date'].iloc[0])
            T = (expiry_dt.date() - snap_dt.date()).days / 365.0
            strikes     = res['strikes']
            z_values    = res['market_iv'] if "IV" in plot_type else res['rnd_market']
            valid_mask  = ~np.isnan(z_values)
            
            for i in range(len(strikes[valid_mask])):
                market_points.append([T, strikes[valid_mask][i], z_values[valid_mask][i]])
            all_strikes.extend(strikes)
            if res.get('params_fast'):
                model_params_by_T.append({'T': T, 'F': res['forward_price'], **res['params_fast']})
    
    if not market_points:
        return None
    
    return {
        "market_points":    np.array(market_points),
        "model_params_by_T": model_params_by_T,
        "all_strikes":      all_strikes
    }



# --- Sidebar ---
with st.sidebar:
    st.markdown("### Surface Controls")
    
    # --- MODEL ENGINE TOGGLE ---
    model_engine_display = st.radio(
        "Select Pricing Model",
        ('Black-76 (Lognormal)', 'Bachelier (Normal)'),
        horizontal=True,
        key='model_engine_selector'
    )
    model_engine = 'black76' if 'Black-76' in model_engine_display else 'bachelier'
    st.markdown("---")

    file_dict = discover_snapshot_files("snapshots")
    all_folders = list(file_dict.keys())
    
    def update_shared_folder():
        """Called when the folder selectbox changes"""
        st.session_state.shared_folder  = st.session_state.folder_selector
        # When folder changes, reset the file selection to all files in the new folder
        new_folder_files                = file_dict.get(st.session_state.folder_selector, [])
        st.session_state.file_selector  = new_folder_files

    def sync_multiselect_from_checkbox():
        """Called when the 'Select All' checkbox changes."""
        select_all      = st.session_state.get('select_all_chains', True)
        files_in_folder = file_dict.get(st.session_state.folder_selector, [])
        if select_all:
            st.session_state.file_selector = files_in_folder
        else:
            st.session_state.file_selector = []

    
    
    # Widget Creation
    default_index = 0
    if 'shared_folder' in st.session_state and st.session_state.shared_folder in all_folders:
        default_index = all_folders.index(st.session_state.shared_folder)
    # Folder selection
    selected_folder   = st.selectbox(
        "Select a Snapshot Folder",
        options =all_folders,
        index   =default_index,
        key     ='folder_selector',
        on_change=update_shared_folder
    )
    
    # Make sure the shared state is updated on the first run
    if 'shared_folder' not in st.session_state and all_folders:
        st.session_state.shared_folder = all_folders[default_index]
    

    files_to_plot = []
    if selected_folder:
        files_in_folder = file_dict.get(selected_folder, [])

        # Determine if the "Select All" box should be checked
        is_all_selected = len(st.session_state.get('file_selector', files_in_folder)) == len(files_in_folder)

        # --- "SELECT ALL" CHECKBOX ---
        st.checkbox(
            "Select All Chains",
            value   =is_all_selected,
            key     ='select_all_chains',
            on_change=sync_multiselect_from_checkbox
        )

        files_to_plot   = st.multiselect(
            "Select Chains to Include",
            options     =files_in_folder,
            format_func =os.path.basename,
            key         ='file_selector' # Add key that links it to the callbacks
        )

    st.markdown("---")

    plot_type = st.radio("Select Surface", ("Implied Volatility (IV)", "Risk-Neutral Density (RND)"),
        horizontal=True,
    )

    interp_method = st.selectbox("Interpolation Method", ("linear", "cubic", "nearest"),
        index=0,
        help="Method to fill gaps between market data points on the surface."
    )


# --- INITIALIZE CAMERA STATE ---               ### REVISIT ###  ## UI ##
# Store the camera position in session state
if 'camera' not in st.session_state:
    st.session_state.camera = None


# --- MAIN PANEL FOR PLOTS ---
if files_to_plot:
    surface_data = prepare_surface_data(tuple(files_to_plot), plot_type, model_engine)

    if not surface_data:
        st.error("No valid data points could be extracted from the selected files.")
    else:
        points          = surface_data["market_points"]
        model_params_by_T = surface_data["model_params_by_T"]

        # Build the df of successful calibrations                                                       
        df_params = pd.DataFrame(model_params_by_T).sort_values('T').drop_duplicates('T')

        # Derive the min/max T and K from the available data                                     ### REVISIT ### ## UI ##
        min_K, max_K = min(surface_data["all_strikes"]), max(surface_data["all_strikes"])
        
        # Use the min/max T from SUCCESSFUL calibrations for the grid and sliders
        if not df_params.empty:
            min_T, max_T = df_params['T'].min(), df_params['T'].max()
        else: # Fallback if no calibrations succeeded
            min_T, max_T = points[:, 0].min(), points[:, 0].max()
       


        # --- CREATE SLIDERS IN SIDEBAR AFTER DATA IS LOADED ---
        with st.sidebar:
            st.markdown("---")
            st.markdown("### Slice Projection Controls")
            T_selected = st.slider("Project Skew (at Maturity)",         min_value=min_T, max_value=max_T, value=(min_T + max_T) / 2, step=0.01)
            K_selected = st.slider("Project Term Structure (at Strike)", min_value=min_K, max_value=max_K, value=(min_K + max_K) / 2, step=0.125)

        # --- Interpolation, Trace Creation, and Plotting ---
        NUM_T_POINTS, NUM_K_POINTS = 100, 100
        grid_T, grid_K = np.mgrid[min_T:max_T:complex(NUM_T_POINTS), 
                                  min_K:max_K:complex(NUM_K_POINTS)]
        grid_Z_market  = griddata(points[:, :2], points[:, 2], 
                                  (grid_T, grid_K), 
                                  method=interp_method)
        traces = []
        
        # --- LEGEND CONFIG ---
        traces.append(go.Surface(x=grid_T, y=grid_K, z=grid_Z_market, 
                                 colorscale ='viridis', opacity=1.0, 
                                 name       ='Market Surface', 
                                 showlegend=True, showscale=False, connectgaps=True
                                 ))

        if len(model_params_by_T) >= 2:
            df_params    = pd.DataFrame(model_params_by_T).sort_values('T').drop_duplicates('T')
            alpha_interp = interp1d(df_params['T'], df_params['alpha'], kind=interp_method, bounds_error=False, fill_value=np.nan)
            beta_interp  = interp1d(df_params['T'], df_params['beta'],  kind=interp_method, bounds_error=False, fill_value=np.nan)
            rho_interp   = interp1d(df_params['T'], df_params['rho'],   kind=interp_method, bounds_error=False, fill_value=np.nan)
            nu_interp    = interp1d(df_params['T'], df_params['nu'],    kind=interp_method, bounds_error=False, fill_value=np.nan)
            F_interp     = interp1d(df_params['T'], df_params['F'],     kind=interp_method, bounds_error=False, fill_value=np.nan)
            grid_Z_model = np.zeros_like(grid_T)
            for i in range(grid_T.shape[0]):
                t = grid_T[i, 0]
                # Get extrapolated parameters and clip them to valid physical ranges
                alpha = float(np.clip(alpha_interp(t), 1e-6, 10.0)) # Alpha must be positive
                beta  = float(np.clip(beta_interp(t), 0.0, 1.0))     # Beta is between 0 and 1
                rho   = float(np.clip(rho_interp(t), -0.9999, 0.9999)) # Rho must be between -1 and 1
                nu    = float(np.clip(nu_interp(t), 1e-6, 10.0))     # Nu must be positive
                
                F     = float(np.clip(F_interp(t), 1e-6, None))
                
                params        = {'alpha': alpha, 'beta': beta, 'rho': rho, 'nu': nu}                
                strikes_slice = grid_K[i, :]
                
                # --- MAKE SABR CALCULATION MODEL-AWARE ---                          ### REVISIT ### ## LOGIC ##
                if "IV" in plot_type:
                    if model_engine == 'black76':
                        result_slice = sabr_vol_lognormal(F, strikes_slice, t, **params)
                    else: # bachelier
                        result_slice = sabr_vol_normal(F, strikes_slice, t, alpha, rho, nu)
                else: # RND
                    result_slice = model_rnd(strikes_slice, F, t, params, model_engine=model_engine)

                
                grid_Z_model[i, :] = result_slice.flatten()
            

            traces.append(go.Surface(
                x=grid_T, y=grid_K, z=grid_Z_model, 
                colorscale  ='plasma', opacity=0.7,
                name        ='Sabr Surface',                
                showlegend=True, showscale=False, connectgaps=False,
                visible     ='legendonly'
                ))
        else:
            st.warning("Sabr Surface requires at least 2 chains with valid SABR calibrations.")
        
        traces.append(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], 
                                   mode='markers', marker=dict(size=2, color='green'), 
                                   name='Raw Market Data'
                                   ))
        skew_strikes    = np.linspace(min_K, max_K, NUM_K_POINTS)
        skew_z          = griddata(points[:, :2], points[:, 2], (T_selected, skew_strikes), method=interp_method)
        term_maturities = np.linspace(min_T, max_T, NUM_T_POINTS)
        term_z          = griddata(points[:, :2], points[:, 2], (term_maturities, K_selected), method=interp_method)
        traces.append(go.Scatter3d(x=np.full_like(skew_strikes, max_T), y=skew_strikes, z=skew_z, 
                                   mode='lines', line=dict(color='cyan', width=5), 
                                   name=f'Market Skew (T={T_selected:.2f})'
                                   ))
        traces.append(go.Scatter3d(x=term_maturities, y=np.full_like(term_maturities, min_K), z=term_z, 
                                   mode='lines', line=dict(color='yellow', width=5), 
                                   name=f'Market Term (K={K_selected:.2f})'
                                   ))
        
        fig = go.Figure(data=traces)

        z_min_all = np.nanmin(points[:, 2]) * 0.8
        z_max_all = np.nanmax(points[:, 2]) * 1.2
        fig.update_layout(title="Market vs. Sabr Surface", height=800, 
                          legend=dict(x=0.8, y=0.95, bgcolor='rgba(0,0,0,0.5)', 
                                                     font   =dict(color='white')), 
                            scene=dict(xaxis=dict(title='Time to Maturity', range=[min_T, max_T * 1.1], 
                                                  showbackground=True, backgroundcolor='rgba(20, 24, 33, 0.8)'), 
                                       yaxis=dict(title='Strike',           range=[min_K * 0.99, max_K], 
                                                  showbackground=True, backgroundcolor='rgba(20, 24, 33, 0.8)'), 
                                        zaxis=dict(title=plot_type,         range=[z_min_all, z_max_all], 
                                                   showbackground=False))
                                        )
        
        # --- APPLY THE SAVED CAMERA STATE ---                                     ### REVISIT ### ## UI ##
        # Before rendering, check if we have a saved camera angle and apply it
        if st.session_state.camera is not None:
            fig.update_layout(scene_camera=st.session_state.camera)

        st.plotly_chart(fig, use_container_width=True, key="surface_plot")

else:
    st.info("Please select a folder and one or more files to begin.")