# pages/4_Greeks_Exposure.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Import necessary functions from your existing modules
from mdl_load import discover_snapshot_files
from mdl_processing import process_snapshot_file
from greeks import calculate_greeks
from sabr_v2 import sabr_vol_lognormal

st.set_page_config(layout="wide", page_title="Greeks Exposure")
st.title("Dealer Greeks Exposure Dashboard")
st.info(
    "This dashboard analyzes the estimated Greeks exposure of option dealers, "
    "assuming they are net short the open interest. All exposures are derived from "
    "the SABR model calibrated to each snapshot."
)

# --- Data Loading and Caching ---

@st.cache_data(show_spinner="Loading all snapshots and calculating Greeks...")
def load_and_process_all_snapshots(folder_paths):
    """
    Loads all parquet files, CORRECTS THE STRIKE DATA, calculates Greeks,
    and returns a comprehensive DataFrame of exposures.
    """
    all_exposures = []
    
    file_dict = discover_snapshot_files("snapshots")
    files_to_process = []
    for folder in folder_paths:
        files_to_process.extend(file_dict.get(folder, []))

    for file_path in files_to_process:
        df = pd.read_parquet(file_path)
        if df.empty:
            continue

        # --- START OF DEFINITIVE STRIKE FIX ---
        if 'ticker' not in df.columns:
            st.warning(f"Skipping file {os.path.basename(file_path)}: missing 'ticker' column.")
            continue

        strike_regex = r'[CP]\s*(\d+(?:\.\d+)?)'
        df['strike_from_ticker'] = pd.to_numeric(df['ticker'].str.extract(strike_regex, expand=False), errors='coerce')
        
        if 'opt_strike_px' in df.columns:
            df['strike_from_bbg'] = pd.to_numeric(df['opt_strike_px'], errors='coerce')
            # Prioritize the ticker parse, but fill any misses with opt_strike_px
            df['strike'] = df['strike_from_ticker'].fillna(df['strike_from_bbg'])
        else:
            df['strike'] = df['strike_from_ticker']
        
        df.drop(columns=['strike_from_ticker', 'strike_from_bbg'], errors='ignore', inplace=True)
        df.dropna(subset=['strike'], inplace=True)
        if df.empty:
            st.warning(f"Skipping file {os.path.basename(file_path)}: could not determine any valid strikes.")
            continue
        # --- END OF DEFINITIVE STRIKE FIX ---

        # Now that strikes are fixed, we can call process_snapshot_file just to get SABR params
        res = process_snapshot_file(file_path, manual_params={})
        if not res or not res.get('params_fast'):
            st.warning(f"Skipping file {os.path.basename(file_path)}: SABR calibration failed.")
            continue
        
        sabr_params = res['params_fast']
        F = res['forward_price']
        T = (pd.to_datetime(df['expiry_date'].iloc[0]).date() - 
             pd.to_datetime(df['snapshot_ts'].iloc[0].split(" ")[0]).date()).days / 365.0

        # Use the SABR params to calculate a smooth vol for ALL strikes in our now-corrected dataframe
        df['sabr_iv'] = sabr_vol_lognormal(F, df['strike'], T, **sabr_params)

        # We can now proceed, using the corrected 'strike' and new 'sabr_iv'
        df_calls = df[df['type'].str.upper() == 'C'].copy()
        df_puts = df[df['type'].str.upper() == 'P'].copy()

        greeks_c = calculate_greeks(F, df_calls['strike'], T, df_calls['sabr_iv'], 'C')
        greeks_p = calculate_greeks(F, df_puts['strike'], T, df_puts['sabr_iv'], 'P')
        
        for greek in greeks_c:
            df_calls[greek] = greeks_c[greek]
            df_puts[greek] = greeks_p[greek]
        
        df_all = pd.concat([df_calls, df_puts])
        
        df_all['rt_open_interest'] = pd.to_numeric(df_all['rt_open_interest'], errors='coerce').fillna(0)
        for greek in greeks_c:
            df_all[f'{greek}_exp'] = df_all[greek] * df_all['rt_open_interest'] * -1
        
        if 'mid' not in df_all.columns:
            df_all['mid'] = (pd.to_numeric(df_all['bid'], errors='coerce') + pd.to_numeric(df_all['ask'], errors='coerce')) / 2.0
        df_all['net_premium'] = df_all['mid'] * df_all['rt_open_interest']
        
        all_exposures.append(df_all)

    if not all_exposures:
        return pd.DataFrame()

    return pd.concat(all_exposures, ignore_index=True)



# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### Exposure Controls")
    
    file_dict = discover_snapshot_files("snapshots")
    all_folders = list(file_dict.keys())
    
    selected_folders = st.multiselect(
        "Select Snapshot Folders to Analyze",
        options=all_folders,
        default=all_folders[:min(3, len(all_folders))] # Default to first 3 folders
    )
    
    st.markdown("---")
    greek_to_show = st.selectbox(
        "Select Greek to Display",
        options=['delta', 'gamma', 'vega', 'theta', 'vanna', 'charm']
    )

# --- Main Panel ---
if not selected_folders:
    st.warning("Please select one or more snapshot folders from the sidebar.")
else:
    full_df = load_and_process_all_snapshots(tuple(selected_folders))

    if full_df.empty:
        st.error("No valid data could be processed from the selected folders.")
    else:
        # Create a time slider for navigating through snapshots
        timestamps = sorted(full_df['snapshot_ts'].unique())
        # Conditionally create the time slider
        if len(timestamps) > 1:
            # If there are multiple snapshots, show the slider for navigation
            selected_ts = st.select_slider(
                "Select Snapshot Time to View",
                options=timestamps,
                value=timestamps[-1] # Default to the latest snapshot
            )
        elif len(timestamps) == 1:
            # If there's only one snapshot, no slider is needed.
            # Automatically select the single timestamp and inform the user.
            selected_ts = timestamps[0]
            st.info(f"Displaying data for the only available snapshot: **{selected_ts}**")
        else:
            # Handle the case where no valid timestamps were found
            st.error("No valid snapshot data to display.")
            st.stop() # Stop execution if there's nothing to show
        
        # Filter the dataframe to the selected snapshot time
        df_at_time = full_df[full_df['snapshot_ts'] == selected_ts].copy()

        # --- Create Tabs for Different Views ---
        tab1, tab2, tab3, tab4 = st.tabs(["Exposure by Strike", "Exposure by Expiry", "Premium Analysis", "Debug"])

        with tab1:
            st.subheader(f"Total {greek_to_show.capitalize()} Exposure by Strike")
            
            exp_col = f'{greek_to_show}_exp'
            exposure_by_strike = df_at_time.groupby('strike')[exp_col].sum().reset_index()
            
            fig = px.bar(
                exposure_by_strike,
                x='strike',
                y=exp_col,
                title=f"Net {greek_to_show.capitalize()} Exposure at {selected_ts}",
                labels={'strike': 'Strike Price', exp_col: f'{greek_to_show.capitalize()} Exposure'}
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader(f"Total {greek_to_show.capitalize()} Exposure by Expiration")
            
            exp_col = f'{greek_to_show}_exp'
            exposure_by_expiry = df_at_time.groupby('expiry_date')[exp_col].sum().reset_index()
            
            fig = px.bar(
                exposure_by_expiry,
                x='expiry_date',
                y=exp_col,
                title=f"Net {greek_to_show.capitalize()} Exposure at {selected_ts}",
                labels={'expiry_date': 'Expiration Date', exp_col: f'{greek_to_show.capitalize()} Exposure'}
            )
            fig.update_xaxes(type='category') # Treat dates as categories for better spacing
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Net Premium Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # OpEx by Expiration
                opex = df_at_time.groupby('expiry_date')['net_premium'].sum().reset_index()
                fig_opex = px.bar(
                    opex,
                    x='expiry_date',
                    y='net_premium',
                    title="Total Premium Expiring by Date (OpEx)",
                    labels={'expiry_date': 'Expiration Date', 'net_premium': 'Total Net Premium'}
                )
                fig_opex.update_xaxes(type='category')
                st.plotly_chart(fig_opex, use_container_width=True)
            
            with col2:
                # Heatmap of Net Premium
                premium_pivot = df_at_time.pivot_table(
                    index='strike',
                    columns='expiry_date',
                    values='net_premium',
                    aggfunc='sum'
                ).fillna(0)
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=premium_pivot.values,
                    x=premium_pivot.columns,
                    y=premium_pivot.index,
                    colorscale='Viridis'
                ))
                fig_heatmap.update_layout(
                    title="Heatmap of Net Premium (Strike vs. Expiry)",
                    xaxis_title="Expiration Date",
                    yaxis_title="Strike Price"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)


        # --- NEW DEBUG TAB ---
        with tab4:
            st.subheader("Diagnostic Data Tables")
            st.info("These tables show the data at each step of the calculation for the selected snapshot time.")
            
            st.markdown("#### 1. Raw Data with Full SABR IV Curve")
            st.write("This table shows the raw option data after applying the full, smooth SABR IV curve (`sabr_iv`) to every strike.")
            st.dataframe(df_at_time[['snapshot_ts', 'expiry_date', 'strike', 'type', 'sabr_iv', 'rt_open_interest']].head(20))

            st.markdown("#### 2. Data with Calculated Greeks")
            st.write("This table shows the calculated Greek value for each option.")
            st.dataframe(df_at_time[['strike', 'type', 'sabr_iv', greek_to_show]].head(20))

            st.markdown("#### 3. Final Exposure Data")
            st.write("This table shows the final calculated exposure (Greek * Open Interest * -1).")
            st.dataframe(df_at_time[['strike', 'type', greek_to_show, 'rt_open_interest', f'{greek_to_show}_exp']].head(20))
