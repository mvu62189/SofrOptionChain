# pages/4_GEX_revised.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from collections import defaultdict

# Import necessary functions from your existing modules
from mdl_load import discover_snapshot_files
from mdl_processing import process_snapshot_file
from greeks import calculate_greeks, calculate_greeks_bachelier
from sabr_v2 import sabr_vol_lognormal, sabr_vol_normal

# --- CONFIGURATION ---
CONTRACT_NOTIONAL = 1_000_000  # Notional value for one SOFR contract

st.set_page_config(layout="wide", page_title="Greeks Exposure (Revised)")
#st.title("Advanced Greeks Exposure Dashboard")

# --- DATA LOADING & PROCESSING (REFACTORED) ---
@st.cache_data(show_spinner="Processing snapshots...")
def load_and_process_data_for_all_timestamps(model_engine):
    """
    Loads all snapshot files, groups them by timestamp, and processes each
    timestamp's data respecting the forward curve for each expiry.
    Returns a dictionary where keys are timestamps and values are processed DataFrames.
    """
    file_dict = discover_snapshot_files("snapshots")
    all_files = [f for files in file_dict.values() for f in files]
    
    if not all_files:
        return {}, {}

    # --- REQ 1: Initialize a log for skipped files ---
    skipped_files_log = defaultdict(list)

    # 1. Load all data into a single DataFrame
    df_list = []
    for f in all_files:
        df = pd.read_parquet(f)
        if not df.empty:
            df['__source_file__'] = f  # Add the source file path as a new column
            df_list.append(df)

    if not df_list:
        return {}, {}

    full_df = pd.concat(df_list, ignore_index=True)

    # 2. Clean and correct the strike column definitively
    strike_regex = r'[CP]\s*(\d+(?:\.\d+)?)'
    full_df['strike'] = pd.to_numeric(full_df['ticker'].str.extract(strike_regex, expand=False), errors='coerce')
    if 'opt_strike_px' in full_df.columns:
        opt_strike_numeric = pd.to_numeric(full_df['opt_strike_px'], errors='coerce')
        full_df['strike'] = full_df['strike'].fillna(opt_strike_numeric)

    # 3. Group by snapshot time and process each snapshot independently
    processed_data = {}
    for ts, df_snapshot in full_df.groupby('snapshot_ts'):
        expiries_data = []
        # 4. For each snapshot, group by expiry to respect the forward curve
        for expiry, df_expiry in df_snapshot.groupby('expiry_date'):
            # This slice of data represents one point on the forward curve
            df_expiry = df_expiry.copy()
            
            # --- REQ 3: Modify file source name for better logging ---
            path_parts = os.path.normpath(df_expiry['__source_file__'].iloc[0]).split(os.sep)
            source_name = os.path.join(*path_parts[-3:])

            # Use a simplified processing logic to get SABR params for this slice
            res, reason = process_snapshot_file(None, manual_params={}, df_input=df_expiry, model_engine=model_engine)
            if not res or not res.get('params_fast'):
                # --- REQ 1: Log skipped files instead of printing warnings ---
                skipped_files_log[reason].append(source_name)
                continue
            
            sabr_params = res['params_fast']
            F = res['forward_price']
            T = (pd.to_datetime(expiry).date() - pd.to_datetime(ts.split(" ")[0]).date()).days / 365.0
            
            # Calculate SABR IV and Greeks for this expiry slice
            #df_expiry['sabr_iv'] = sabr_vol_lognormal(F, df_expiry['strike'], T, **sabr_params)
            
            # --- MAKE SABR IV CALCULATION MODEL-AWARE ---
            if model_engine == 'black76':
                df_expiry['sabr_iv'] = sabr_vol_lognormal(F, df_expiry['strike'], T, **sabr_params)
                greeks_func = calculate_greeks
            else: # bachelier
                df_expiry['sabr_iv'] = sabr_vol_normal(F, df_expiry['strike'], T, sabr_params['alpha'], sabr_params['rho'], sabr_params['nu'])
                greeks_func = calculate_greeks_bachelier

            calls = df_expiry[df_expiry['type'].str.upper() == 'C'].copy()
            puts = df_expiry[df_expiry['type'].str.upper() == 'P'].copy()

            # --- MAKE GREEKS CALCULATION MODEL-AWARE ---   ### SEE IMPLEMENTED ABOVE ###
            #if model_engine == 'black76':
            #    greeks_func = calculate_greeks
            #else: # bachelier
            #    greeks_func = calculate_greeks_bachelier       


            if not calls.empty:
                greeks_c = greeks_func(F, calls['strike'], T, calls['sabr_iv'], 'C')
                for greek in greeks_c: calls[greek] = greeks_c[greek]
            
            if not puts.empty:
                greeks_p = greeks_func(F, puts['strike'], T, puts['sabr_iv'], 'P')
                for greek in greeks_p: puts[greek] = greeks_p[greek]
            
            df_processed_expiry = pd.concat([calls, puts])
            df_processed_expiry['forward_price'] = F
            expiries_data.append(df_processed_expiry)
        
        if expiries_data:
            processed_data[ts] = pd.concat(expiries_data, ignore_index=True)

    return processed_data, skipped_files_log

# --- Main Application ---

# Load all data once
#all_data = load_and_process_data_for_all_timestamps()

#if not all_data:
#    st.error("No valid snapshot data found. Please run a new snapshot.")
#    st.stop()

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### Dashboard Controls")

    # --- ADD MODEL ENGINE TOGGLE ---
    model_engine_display = st.radio(
        "Select Pricing Model",
        ('Black-76 (Lognormal)', 'Bachelier (Normal)'),
        horizontal=True,
        key='model_engine_selector'
    )
    model_engine = 'bachelier' if 'Bachelier' in model_engine_display else 'black76'

# --- Load data AFTER setting the model engine ---
all_data, skipped_files = load_and_process_data_for_all_timestamps(model_engine)

# --- Stop gracefully if no data is found ---
if not all_data:
    st.error("No valid snapshot data found. Please run a new snapshot.")
    # --- REQ 1: Still show the skipped files tab even if all data fails ---
    if skipped_files:    
        with st.expander("Skipped Files Log"):
            for reason, files in skipped_files.items():
                st.subheader(reason)
                st.json(files)
        st.stop()

# --- Continue building the sidebar with the loaded data ---
with st.sidebar:
    timestamps = sorted(all_data.keys())
    selected_ts = st.select_slider("Select Snapshot Time", options=timestamps, value=timestamps[-1])

# Get the data for the selected timestamp
df = all_data[selected_ts].copy()

# --- Calculate all exposure metrics ---
greeks_cols = ['delta', 'gamma', 'vega', 'theta', 'vanna', 'charm']
for col in greeks_cols:
    if col not in df.columns:
        df[col] = 0.0

df['rt_open_interest'] = pd.to_numeric(df['rt_open_interest'], errors='coerce').fillna(0)
df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

df['delta_exp'] = df['delta'] * df['rt_open_interest'] * -1 * CONTRACT_NOTIONAL
df['gamma_exp'] = df['gamma'] * df['rt_open_interest'] * -1 * (CONTRACT_NOTIONAL * 0.01) * 0.01
df['vega_exp'] = df['vega'] * df['rt_open_interest'] * -1 * CONTRACT_NOTIONAL
df['vanna_exp'] = df['vanna'] * df['rt_open_interest'] * -1 * CONTRACT_NOTIONAL * 0.01
df['charm_exp'] = df['charm'] * df['rt_open_interest'] * -1 * CONTRACT_NOTIONAL
df['theta_exp'] = df['theta'] * df['rt_open_interest'] * -1 * CONTRACT_NOTIONAL

# --- REQ 4 & 5: Enhanced Market State Block ---
#st.header(f"Market State at: {selected_ts}")
#st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Total GEX ($)", f"${df['gamma_exp'].sum():,.0f}",
            help="**Formula:** `Gamma * OI * -1 * (Notional * 0.01)^2` | Gamma Exposure per 1% move in the underlying, squared.")
col2.metric("Total VEX ($)", f"${df['vanna_exp'].sum():,.0f}",
            help="**Formula:** `Vanna * OI * -1 * Notional * 0.01` | Change in total Delta for a 1% change in implied volatility.")
gamma_flip_point = df.groupby('strike')['gamma_exp'].sum().cumsum()
try:
    flip_strike = gamma_flip_point[gamma_flip_point > 0].index[0]
    col3.metric("Gamma Flip Point", f"{flip_strike:.2f}",
                help="The strike level where dealer Gamma exposure flips from negative to positive.")
except IndexError:
    col3.metric("Gamma Flip Point", "N/A", help="Dealer Gamma exposure does not flip to positive in the observed range.")

col1, col2, col3 = st.columns(3)
col1.metric("Total Delta Exp ($)", f"${df['delta_exp'].sum():,.0f}",
            help="**Formula:** `Delta * OI * -1 * Notional` | Total dollar value change of dealer positions for a +1 point move in the underlying.")
col2.metric("Total Charm ($)", f"${df['charm_exp'].sum():,.0f}",
            help="**Formula:** `Charm * OI * -1 * Notional` | Change in total Delta per day (Delta decay).")
col3.metric("Total Theta ($)", f"${df['theta_exp'].sum():,.0f}",
            help="**Formula:** `Theta * OI * -1 * Notional` | PnL decay per day from being short options.")
#st.markdown("---")

# --- REQ 1, 3, 5, 6: Create Tabs for Different Views ---
tab_names = ["Forward Curve", "Exposure by Strike", "Exposure by Expiry", "Time Series Analysis", "Expiry Drill-Down", "Strike Drill-Down"]
if skipped_files:
    tab_names.append("Skipped Files Log")
tabs = st.tabs(tab_names)

with tabs[0]: # Forward Curve
    st.subheader("SOFR Forward Curve")
    forward_curve = df[['expiry_date', 'forward_price']].drop_duplicates().sort_values('expiry_date')
    st.plotly_chart(px.line(forward_curve, x='expiry_date', y='forward_price', title="Forward Price by Expiry", markers=True), use_container_width=True)

# --- REQ 1 & 2: Define a reusable function for the new plot style ---
def create_exposure_plot(data, group_by_col, greek, show_net):
    exp_col = f'{greek}_exp'
    
    # Define plotting exposure based on Greek conventions for visual separation
    if greek in ['gamma', 'vanna']:
        data['plot_exp'] = np.where(data['type'].str.upper() == 'C', -data[exp_col].abs(), data[exp_col].abs())
    elif greek in ['delta']:
        # Short Call (positive delta) -> negative exposure. Short Put (negative delta) -> positive exposure
        data['plot_exp'] = data[exp_col] * -1
    else: # Theta, Charm
        data['plot_exp'] = data[exp_col]

    if show_net:
        exposure_data = data.groupby(group_by_col)['plot_exp'].sum().reset_index()
        exposure_data['color'] = np.where(exposure_data['plot_exp'] >= 0, 'red', 'green')
        fig = px.bar(exposure_data, x=group_by_col, y='plot_exp', title=f"Net {greek.capitalize()} Exposure",
                     color='color', color_discrete_map={'green':'green', 'red':'red'})
    else:
        exposure_data = data.groupby([group_by_col, 'type'])['plot_exp'].sum().reset_index()
        fig = px.bar(exposure_data, x=group_by_col, y='plot_exp', color='type',
                     title=f"{greek.capitalize()} Exposure by Option Type")
                     #color_discrete_map={'C': 'red', 'P': 'green'}) # REQ 2: Green for Puts
        fig.for_each_trace(
            lambda t: t.update(marker_color='green') if t.name.upper() == 'P' else t.update(marker_color='red')
        )

    if group_by_col == 'expiry_date':
        fig.update_xaxes(type='category')
        
    return fig

with tabs[1]: # Exposure by Strike
    c1, c2 = st.columns([1, 3])
    # --- REQ 2: Show all 6 greeks ---
    greek_strike = c1.selectbox("Select Greek", greeks_cols, key='strike_greek')
    # --- REQ 1: New "Show Net" checkbox ---
    show_net_strike = c2.checkbox("Show Net Exposure", value=False, key='strike_net')
    
    fig = create_exposure_plot(df, 'strike', greek_strike, show_net_strike)
    
    # --- ADDED: Get the average forward for context ---
    # For a multi-expiry view, we take the average forward price.
    avg_fwd = df['forward_price'].mean()
    fig.add_vline(x=avg_fwd, line_width=2, line_dash="dash", line_color="white", 
                  annotation_text=f"Avg Fwd={avg_fwd:.2f}", annotation_position="top")
                  
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]: # Exposure by Expiry
    c1, c2 = st.columns([1, 3])
    greek_expiry = c1.selectbox("Select Greek", greeks_cols, key='expiry_greek')
    show_net_expiry = c2.checkbox("Show Net Exposure", value=False, key='expiry_net')
    
    fig = create_exposure_plot(df, 'expiry_date', greek_expiry, show_net_expiry)
    st.plotly_chart(fig, use_container_width=True)


with tabs[3]: # Time Series Analysis
    st.subheader("Time Series of Total Net Exposures")
    # --- REQ 4: Calculate time series for all greeks ---
    ts_data = []
    for ts, data in all_data.items():
        ts_point = {'timestamp': pd.to_datetime(ts, format='%Y%m%d %H%M%S')}
        oi = pd.to_numeric(data['rt_open_interest'], errors='coerce').fillna(0)
        
        for col in greeks_cols:
             if col not in data.columns: data[col] = 0.0

        ts_point['gamma_exp'] = (data['gamma'] * oi * -1 * (CONTRACT_NOTIONAL * 0.01) * 0.01).sum()
        ts_point['vanna_exp'] = (data['vanna'] * oi * -1 * CONTRACT_NOTIONAL * 0.01).sum()
        ts_point['charm_exp'] = (data['charm'] * oi * -1 * CONTRACT_NOTIONAL).sum()
        ts_point['theta_exp'] = (data['theta'] * oi * -1 * CONTRACT_NOTIONAL).sum()
        ts_point['delta_exp'] = (data['delta'] * oi * -1 * CONTRACT_NOTIONAL).sum()
        ts_point['vega_exp'] = (data['vega'] * oi * -1 * CONTRACT_NOTIONAL).sum()
        ts_data.append(ts_point)
    
    if ts_data:
        df_ts = pd.DataFrame(ts_data).sort_values('timestamp')
        for greek in greeks_cols:
            exp_col = f'{greek}_exp'
            st.plotly_chart(px.line(df_ts, x='timestamp', y=exp_col, title=f"Total Net {greek.capitalize()} Exposure ($) Over Time", markers=True), use_container_width=True)
    else:
        st.warning("Not enough data to build time series charts.")

# --- REQ 3: New Expiry Drill-Down Tab with PLOTS ---
with tabs[4]:
    st.subheader("Greeks Exposure by Strike for a Single Expiry")
    expiries = sorted(df['expiry_date'].unique())
    selected_expiry = st.selectbox("Select Expiry to Analyze", expiries)
    
    df_drill_exp = df[df['expiry_date'] == selected_expiry]
    
    c1, c2 = st.columns([1, 3])
    greek_drill_exp = c1.selectbox("Select Greek", greeks_cols, key='drill_exp_greek')
    show_net_drill_exp = c2.checkbox("Show Net Exposure", value=False, key='drill_exp_net')
    
    fig = create_exposure_plot(df_drill_exp, 'strike', greek_drill_exp, show_net_drill_exp)
    
    # --- ADDED: Get the specific forward price for the selected expiry ---
    fwd_price = df_drill_exp['forward_price'].iloc[0]
    fig.add_vline(x=fwd_price, line_width=2, line_dash="dash", line_color="white", 
                  annotation_text=f"Fwd={fwd_price:.2f}", annotation_position="top")
    
    st.plotly_chart(fig, use_container_width=True)

# --- REQ 3: New Strike Drill-Down Tab with PLOTS ---
with tabs[5]:
    st.subheader("Greeks Exposure by Expiry for a Single Strike")
    strikes = sorted(df['strike'].unique())
    selected_strike = st.select_slider("Select Strike to Analyze", options=strikes)
    
    df_drill_str = df[df['strike'] == selected_strike]

    c1, c2 = st.columns([1, 3])
    greek_drill_str = c1.selectbox("Select Greek", greeks_cols, key='drill_str_greek')
    show_net_drill_str = c2.checkbox("Show Net Exposure", value=False, key='drill_str_net')
    
    fig = create_exposure_plot(df_drill_str, 'expiry_date', greek_drill_str, show_net_drill_str)
    st.plotly_chart(fig, use_container_width=True)

# --- REQ 3: Display the skipped files log in the last tab ---
if skipped_files:
    with tabs[6]:
        st.header("Log of Skipped or Failed Snapshots")
        for reason, files in sorted(skipped_files.items()):
            with st.expander(f"{reason} ({len(files)} files)"):
                st.json(files)



# --- Create Tabs for Different Views ---
#tab1, tab2, tab3, tab4 = st.tabs(["Forward Curve", "Exposure by Strike", "Exposure by Expiry", "Time Series Analysis"])

#with tab1:
#    st.subheader("SOFR Forward Curve")
#    forward_curve = df[['expiry_date', 'forward_price']].drop_duplicates().sort_values('expiry_date')
#    fig = px.line(forward_curve, x='expiry_date', y='forward_price', title="Forward Price by Expiry", markers=True)
#    fig.update_layout(xaxis_title="Expiration Date", yaxis_title="Forward Price (100 - Rate)")
#    st.plotly_chart(fig, use_container_width=True)

#with tab2:
#    st.subheader("Exposure Profile by Strike")
#    # Prepare data for stacked bar chart (dealer short calls = negative gamma, short puts = positive gamma)
#    df['plot_gamma_exp'] = np.where(df['type'].str.upper() == 'C', -df['gamma_exp'], df['gamma_exp'])
#    
#    exposure_by_strike = df.groupby(['strike', 'type'])['plot_gamma_exp'].sum().reset_index()
#    
#    fig = px.bar(
#        exposure_by_strike,
#        x='strike',
#        y='plot_gamma_exp',
#        color='type',
#        title="Gamma Exposure (GEX) by Strike",
#        labels={'strike': 'Strike Price', 'plot_gamma_exp': 'Gamma Exposure ($)'},
#        color_discrete_map={'c': 'red', 'p': 'green'}
#    )
#    st.plotly_chart(fig, use_container_width=True)#

#with tab3:
#    st.subheader("Exposure Profile by Expiry")
#    df['plot_gamma_exp'] = np.where(df['type'].str.upper() == 'C', -df['gamma_exp'], df['gamma_exp'])
#    exposure_by_expiry = df.groupby(['expiry_date', 'type'])['plot_gamma_exp'].sum().reset_index()
#    fig = px.bar(
#        exposure_by_expiry,
#        x='expiry_date',
#        y='plot_gamma_exp',
#        color='type',
#        title="Gamma Exposure (GEX) by Expiry",
#        labels={'expiry_date': 'Expiration Date', 'plot_gamma_exp': 'Gamma Exposure ($)'},
#        color_discrete_map={'c': 'red', 'p': 'green'}
#    )
#    fig.update_xaxes(type='category')
#    st.plotly_chart(fig, use_container_width=True)

#with tab4:
#    st.subheader("Time Series of Total Net Exposure")
#    
#    # Calculate total net GEX for each timestamp
#    time_series_data = []
#    for ts, data in all_data.items():
#        gex = data['gamma'].multiply(data['rt_open_interest']).multiply(-1).sum()
#        gex_notional = gex * (CONTRACT_NOTIONAL * 0.01) * 0.01
#        time_series_data.append({'timestamp': ts, 'total_gex': gex_notional})
#    
#    df_ts = pd.DataFrame(time_series_data).sort_values('timestamp')
#    
#   fig = px.line(df_ts, x='timestamp', y='total_gex', title="Total Net Gamma Exposure (GEX) Over Time", markers=True)
#    fig.update_layout(xaxis_title="Snapshot Time", yaxis_title="Total GEX ($)")
#    st.plotly_chart(fig, use_container_width=True)
