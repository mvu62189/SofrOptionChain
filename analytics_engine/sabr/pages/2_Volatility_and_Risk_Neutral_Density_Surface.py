# pages/2_Surfaces.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata, interp1d
import os
import joblib # Import joblib for loading cached files
from scipy.integrate import trapezoid

# Import necessary functions from existing modules
from mdl_load       import discover_snapshot_files
from mdl_processing import process_snapshot_file  # Keep as a fallback
from sabr_v2        import sabr_vol_lognormal, sabr_vol_normal
from mdl_rnd_utils  import model_rnd

# --- CONFIGURATION ---
CACHE_DIR = "precalibrated_cache"

st.set_page_config(layout="wide")
st.title("IV & RND Surfaces")
st.info(
    "IV and RND each constructs two surfaces: one derived directly from market data, and a second "
    "derived from an interpolation of the SABR model parameters across expiries."
)

def load_precalibrated_file(file_path, model_engine):
    """
    Loads a pre-calibrated result from the cache. If not found, it processes
    the file in real-time as a fallback.
    """
    from pathlib import Path
    
    # Construct the expected path for the cached file
    relative_path = Path(file_path).relative_to("snapshots")
    cache_file = Path(CACHE_DIR) / relative_path.parent / f"{relative_path.stem}_{model_engine}.joblib"

    if cache_file.exists():
        # If cache exists, load it
        return joblib.load(cache_file), None
    else:
        # Fallback: process the file live if no cache is found
        # st.warning(f"No pre-calibrated file found for {relative_path.name}. Processing live...", icon="⚙️")
        return process_snapshot_file(file_path, manual_params={}, model_engine=model_engine)


# --- REFACTORED DATA PREPARATION ---
# This function is now much faster as it just loads and aggregates cached data.
@st.cache_data(show_spinner="Loading and aggregating pre-calibrated data...")
def preparesurface_data(file_paths, plot_type, model_engine):
    """
    Takes a list of file paths, loads their pre-calibrated results, and aggregates
    the data needed for plotting.
    """
    market_points, model_params_by_T, all_strikes = [], [], []
    
    for file_path in file_paths:
        # Use the new loading function
        res, reason = load_precalibrated_file(file_path, model_engine)
        
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
            
            # Check for calibrated parameters in the loaded result
            if res.get('params_fast'):
                model_params_by_T.append({'T': T, 'expiry_date': expiry_dt.date(), 'F': res['forward_price'], **res['params_fast']})
    
    if not market_points:
        return None
    
    return {
        "market_points":    np.array(market_points),
        "model_params_by_T": model_params_by_T,
        "all_strikes":      all_strikes
    }

# --- NEW FUNCTION to calculate the probability matrix ---
@st.cache_data(show_spinner="Calculating probability matrix...")
def calculate_probability_matrix(surface_data, model_engine):
    """
    Calculates the probability of the underlying finishing in 0.25 point bins
    by integrating the model-derived RND for each expiry.
    """
    if not surface_data or not surface_data.get("model_params_by_T"):
        return pd.DataFrame()

    # Prepare data and define strike bins
    model_params = pd.DataFrame(surface_data['model_params_by_T']).sort_values('T').drop_duplicates('T')
    min_K = np.floor(min(surface_data['all_strikes']))
    max_K = np.ceil(max(surface_data['all_strikes']))
    bins = np.arange(min_K, max_K + 0.25, 0.25)
    
    matrix_data = []

    # Process each expiry
    for _, row in model_params.iterrows():
        T = row['T']
        F = row['F']
        params = {'alpha': row['alpha'], 'beta': row['beta'], 'rho': row['rho'], 'nu': row['nu']}
        
        # 1. Create a high-resolution strike range for the RND curve
        fine_strikes = np.linspace(min_K, max_K, 2500)
        
        # 2. Calculate the RND from the SABR model
        rnd_values = model_rnd(fine_strikes, F, T, params, model_engine=model_engine)
        rnd_values = np.maximum(rnd_values, 0) # Ensure no negative probabilities

        # 3. Integrate the RND curve over each 0.25 point bin
        probabilities = {}
        for i in range(len(bins) - 1):
            k_low, k_high = bins[i], bins[i+1]
            mask = (fine_strikes >= k_low) & (fine_strikes <= k_high)
            # Use the trapezoidal rule to find the area (probability) under the curve
            bin_prob = trapezoid(rnd_values[mask], fine_strikes[mask])

            column_name = f"{k_low:.2f}-{k_high:.2f}"
            probabilities[column_name] = bin_prob

        row_data = {'expiry': row['expiry_date'], **probabilities}
        matrix_data.append(row_data)

    if not matrix_data:
        return pd.DataFrame()

    # Format the final DataFrame
    df = pd.DataFrame(matrix_data).set_index('expiry').sort_index()
    df.index = pd.to_datetime(df.index).strftime('%-m/%d/%Y')
    return df

# --- Sidebar (No changes needed here, but included for completeness) ---
with st.sidebar:
    st.markdown("### Surface Controls")
    
    model_engine_display = st.radio(
        "Select Pricing Model",
        ('Black-76 (Lognormal)', 'Bachelier (Normal)'),
        horizontal=True,
        key='model_engine_selector'
    )
    model_engine = 'black76' if 'Black-76' in model_engine_display else 'bachelier'
    st.markdown("---")

    file_dict = discover_snapshot_files("snapshots")
    all_folders = sorted(list(file_dict.keys()))
    
    if not all_folders:
        st.warning("No snapshot folders found in the 'snapshots' directory.")
        selected_folder = None
        files_to_plot = []
    else:
        def update_from_slider():
            selected_index = st.session_state.folder_slider
            folder_name = all_folders[selected_index]
            st.session_state.shared_folder = folder_name
            files_in_new_folder = file_dict.get(folder_name, [])
            st.session_state.file_selector = files_in_new_folder
            st.session_state.select_all_chains = True

        def sync_multiselect_from_checkbox():
            select_all = st.session_state.get('select_all_chains', True)
            files_in_folder = file_dict.get(st.session_state.shared_folder, [])
            if select_all:
                st.session_state.file_selector = files_in_folder
            else:
                st.session_state.file_selector = []

        default_index = len(all_folders) - 1 
        if 'shared_folder' in st.session_state and st.session_state.shared_folder in all_folders:
            default_index = all_folders.index(st.session_state.shared_folder)
        
        selected_index = st.slider(
            "Select Snapshot Timestamp",
            min_value=0, max_value=len(all_folders) - 1,
            value=default_index,
            key='folder_slider',
            on_change=update_from_slider
        )
        
        selected_folder = all_folders[selected_index]
        
        sanitized_folder_name = selected_folder.replace('\\', '_').replace('/', '_')
        selected_folder_formatted = pd.to_datetime(sanitized_folder_name, format='%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
        st.markdown(f"**Selected:** `{selected_folder_formatted}`")

        if 'shared_folder' not in st.session_state:
            st.session_state.shared_folder = selected_folder
            st.session_state.file_selector = file_dict.get(selected_folder, [])
            st.session_state.select_all_chains = True

        files_to_plot = []
        if selected_folder:
            files_in_folder = file_dict.get(selected_folder, [])
            is_all_selected = len(st.session_state.get('file_selector', [])) == len(files_in_folder)
            st.checkbox("Select All Chains", value=is_all_selected, key='select_all_chains', on_change=sync_multiselect_from_checkbox)
            files_to_plot = st.multiselect("Select Chains to Include", options=files_in_folder, format_func=os.path.basename, key='file_selector')

    st.markdown("---")
    plot_type = st.radio("Select Surface", ("Implied Volatility (IV)", "Risk-Neutral Density (RND)"), horizontal=True)
    interp_method = st.selectbox("Interpolation Method", ("linear", "cubic", "nearest"), index=0, help="Method to fill gaps between market data points on the surface.")

# --- INITIALIZE CAMERA STATE ---
if 'camera' not in st.session_state:
    st.session_state.camera = None

# --- MAIN PANEL FOR PLOTS (No changes needed below this line) ---
if files_to_plot:
    surface_data = preparesurface_data(tuple(files_to_plot), plot_type, model_engine)

    if not surface_data:
        st.error("No valid data points could be extracted from the selected files.")
    else:
        points              = surface_data["market_points"]
        model_params_by_T = surface_data["model_params_by_T"]
        df_params = pd.DataFrame(model_params_by_T).sort_values('T').drop_duplicates('T')
        min_K, max_K = min(surface_data["all_strikes"]), max(surface_data["all_strikes"])
        
        if not df_params.empty:
            min_T, max_T = df_params['T'].min(), df_params['T'].max()
        else: 
            min_T, max_T = points[:, 0].min(), points[:, 0].max()
        
        with st.sidebar:
            st.markdown("---")
            st.markdown("### Slice Projection Controls")
            T_selected = st.slider("Project Skew (at Maturity)",        min_value=min_T, max_value=max_T, value=(min_T + max_T) / 2, step=0.01)
            K_selected = st.slider("Project Term Structure (at Strike)", min_value=min_K, max_value=max_K, value=(min_K + max_K) / 2, step=0.125)

        NUM_T_POINTS, NUM_K_POINTS = 100, 100
        grid_T, grid_K = np.mgrid[min_T:max_T:complex(NUM_T_POINTS), min_K:max_K:complex(NUM_K_POINTS)]
        grid_Z_market  = griddata(points[:, :2], points[:, 2], (grid_T, grid_K), method=interp_method)
        traces = []
        
        traces.append(go.Surface(x=grid_T, y=grid_K, z=grid_Z_market, colorscale='viridis', opacity=1.0, name='Market Surface', showlegend=True, showscale=False, connectgaps=True))

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
                alpha = float(np.clip(alpha_interp(t), 1e-6, 10.0))
                beta  = float(np.clip(beta_interp(t), 0.0, 1.0))
                rho   = float(np.clip(rho_interp(t), -0.9999, 0.9999))
                nu    = float(np.clip(nu_interp(t), 1e-6, 10.0))
                F     = float(np.clip(F_interp(t), 1e-6, None))
                params        = {'alpha': alpha, 'beta': beta, 'rho': rho, 'nu': nu}
                strikes_slice = grid_K[i, :]
                
                if "IV" in plot_type:
                    if model_engine == 'black76':
                        result_slice = sabr_vol_lognormal(F, strikes_slice, t, **params)
                    else:
                        result_slice = sabr_vol_normal(F, strikes_slice, t, alpha, rho, nu)
                else:
                    result_slice = model_rnd(strikes_slice, F, t, params, model_engine=model_engine)
                grid_Z_model[i, :] = result_slice.flatten()

            traces.append(go.Surface(x=grid_T, y=grid_K, z=grid_Z_model, colorscale='plasma', opacity=0.7, name='Sabr Surface', showlegend=True, showscale=False, connectgaps=False, visible='legendonly'))
        else:
            st.warning("Sabr Surface requires at least 2 chains with valid SABR calibrations.")
        
        traces.append(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers', marker=dict(size=2, color='green'), name='Raw Market Data'))
        skew_strikes    = np.linspace(min_K, max_K, NUM_K_POINTS)
        skew_z          = griddata(points[:, :2], points[:, 2], (T_selected, skew_strikes), method=interp_method)
        term_maturities = np.linspace(min_T, max_T, NUM_T_POINTS)
        term_z          = griddata(points[:, :2], points[:, 2], (term_maturities, K_selected), method=interp_method)
        traces.append(go.Scatter3d(x=np.full_like(skew_strikes, max_T), y=skew_strikes, z=skew_z, mode='lines', line=dict(color='cyan', width=5), name=f'Market Skew (T={T_selected:.2f})'))
        traces.append(go.Scatter3d(x=term_maturities, y=np.full_like(term_maturities, min_K), z=term_z, mode='lines', line=dict(color='yellow', width=5), name=f'Market Term (K={K_selected:.2f})'))
        
        fig = go.Figure(data=traces)

        z_min_all = np.nanmin(points[:, 2]) * 0.8
        z_max_all = np.nanmax(points[:, 2]) * 1.2
        fig.update_layout(title="Market vs. Sabr Surface", height=800, 
                          legend=dict(x=0.8, y=0.95, bgcolor='rgba(0,0,0,0.5)', font=dict(color='white')), 
                          scene=dict(xaxis=dict(title='Time to Maturity', range=[min_T, max_T * 1.1], showbackground=True, backgroundcolor='rgba(20, 24, 33, 0.8)'), 
                                     yaxis=dict(title='Strike', range=[min_K * 0.99, max_K], showbackground=True, backgroundcolor='rgba(20, 24, 33, 0.8)'), 
                                     zaxis=dict(title=plot_type, range=[z_min_all, z_max_all], showbackground=False)))
        
        if st.session_state.camera is not None:
            fig.update_layout(scene_camera=st.session_state.camera)

        st.plotly_chart(fig, use_container_width=True, key="surface_plot")

        # --- NEW SECTION FOR PROBABILITY MATRIX ---
        st.subheader("Probability of Underlying Price at Expiry")
        st.markdown(
            "The table below shows the risk-neutral probability of the underlying price finishing "
            "within a 0.25 point range at each expiry. Probabilities are derived by integrating the "
            "SABR model's Risk-Neutral Density (RND) function for the selected snapshot."
        )
        
        # Check if we have model parameters to work with
        if len(surface_data.get("model_params_by_T", [])) >= 2:
            prob_matrix = calculate_probability_matrix(surface_data, model_engine)
            if not prob_matrix.empty:
                threshold = 0.01  # 0.05% threshold to consider a probability meaningful
        
                # Find columns that have at least one value above the threshold
                non_zero_cols = (prob_matrix > threshold).any(axis=0)
                
                if non_zero_cols.any():
                    # Get the names of the first and last valid columns
                    first_col_name = non_zero_cols.idxmax()
                    last_col_name = non_zero_cols[::-1].idxmax()
                    
                    # Slice the DataFrame to the relevant range
                    trimmed_prob_matrix = prob_matrix.loc[:, first_col_name:last_col_name]
                else:
                    trimmed_prob_matrix = prob_matrix # Show original if all are zero

                # Level 1: Price Range (already created)
                price_ranges = trimmed_prob_matrix.columns.tolist()
                
                # Level 2: Rate Range (100 - Price)
                rate_ranges = []
                for price_range in price_ranges:
                    low, high = map(float, price_range.split('-'))
                    rate_high = 100 - high
                    rate_low = 100 - low
                    rate_ranges.append(f"{rate_low:.2f}%-{rate_high:.2f}%")
                
                # Create and apply the MultiIndex
                trimmed_prob_matrix.columns = pd.MultiIndex.from_arrays(
                    [price_ranges, rate_ranges],
                    names=["Underlying Price", "Implied Rate"]
                )

                # --- MODIFICATION: Update styling for dark theme ---
                st.dataframe(
                    trimmed_prob_matrix.style.format("{:.1%}")
                        .set_properties(**{
                            'background-color': '#0E1117', # Match Streamlit dark theme
                            'width': '60px',
                            'text-align': 'center'
                        })
                        .text_gradient(cmap='Greens_r', axis=1, low=0.0, high=1), 
                    use_container_width=True
                )
            else:
                st.warning("Could not generate probability matrix from the available data.")
        else:
            st.info("Requires at least two calibrated option chains to generate the probability matrix.")

else:
    st.info("Please select a snapshot and one or more chains to begin.")
