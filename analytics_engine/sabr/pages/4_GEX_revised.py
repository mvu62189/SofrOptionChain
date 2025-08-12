# pages/5_GEX_revised.py

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

# --- CONFIGURATION ---
CONTRACT_NOTIONAL = 1_000_000  # Notional value for one SOFR contract

st.set_page_config(layout="wide", page_title="Greeks Exposure (Revised)")
st.title("Advanced Greeks Exposure Dashboard")

# --- DATA LOADING & PROCESSING (REFACTORED) ---
@st.cache_data(show_spinner="Processing snapshots...")
def load_and_process_data_for_all_timestamps():
    """
    Loads all snapshot files, groups them by timestamp, and processes each
    timestamp's data respecting the forward curve for each expiry.
    Returns a dictionary where keys are timestamps and values are processed DataFrames.
    """
    file_dict = discover_snapshot_files("snapshots")
    all_files = [f for files in file_dict.values() for f in files]
    
    if not all_files:
        return {}

    # 1. Load all data into a single DataFrame
    df_list = [pd.read_parquet(f) for f in all_files]
    full_df = pd.concat(df_list, ignore_index=True)

    # 2. Clean and correct the strike column definitively
    strike_regex = r'[CP]\s*(\d+(?:\.\d+)?)'
    full_df['strike'] = pd.to_numeric(full_df['ticker'].str.extract(strike_regex, expand=False), errors='coerce')
    if 'opt_strike_px' in full_df.columns:
        full_df['strike'].fillna(pd.to_numeric(full_df['opt_strike_px'], errors='coerce'), inplace=True)
    full_df.dropna(subset=['strike'], inplace=True)

    # 3. Group by snapshot time and process each snapshot independently
    processed_data = {}
    for ts, df_snapshot in full_df.groupby('snapshot_ts'):
        expiries_data = []
        # 4. For each snapshot, group by expiry to respect the forward curve
        for expiry, df_expiry in df_snapshot.groupby('expiry_date'):
            # This slice of data represents one point on the forward curve
            df_expiry = df_expiry.copy()
            
            # Use a simplified processing logic to get SABR params for this slice
            res = process_snapshot_file(None, manual_params={}, df_input=df_expiry)
            if not res or not res.get('params_fast'):
                continue
            
            sabr_params = res['params_fast']
            F = res['forward_price']
            T = (pd.to_datetime(expiry).date() - pd.to_datetime(ts.split(" ")[0]).date()).days / 365.0
            
            # Calculate SABR IV and Greeks for this expiry slice
            df_expiry['sabr_iv'] = sabr_vol_lognormal(F, df_expiry['strike'], T, **sabr_params)
            
            calls = df_expiry[df_expiry['type'].str.upper() == 'C'].copy()
            puts = df_expiry[df_expiry['type'].str.upper() == 'P'].copy()

            if not calls.empty:
                greeks_c = calculate_greeks(F, calls['strike'], T, calls['sabr_iv'], 'C')
                for greek in greeks_c: calls[greek] = greeks_c[greek]
            
            if not puts.empty:
                greeks_p = calculate_greeks(F, puts['strike'], T, puts['sabr_iv'], 'P')
                for greek in greeks_p: puts[greek] = greeks_p[greek]
            
            df_processed_expiry = pd.concat([calls, puts])
            df_processed_expiry['forward_price'] = F
            expiries_data.append(df_processed_expiry)
        
        if expiries_data:
            processed_data[ts] = pd.concat(expiries_data, ignore_index=True)

    return processed_data

# --- Main Application ---

# Load all data once
all_data = load_and_process_data_for_all_timestamps()

if not all_data:
    st.error("No valid snapshot data found. Please run a new snapshot.")
    st.stop()

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### Dashboard Controls")
    timestamps = sorted(all_data.keys())
    selected_ts = st.selectbox("Select Snapshot Time", options=timestamps, index=len(timestamps)-1)

# Get the data for the selected timestamp
df = all_data[selected_ts].copy()

# --- Calculate Exposure Metrics ---
df['rt_open_interest'] = pd.to_numeric(df['rt_open_interest'], errors='coerce').fillna(0)
df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

# GEX (Gamma Exposure) in $, per 1% move in underlying, per 1% underlying change
df['gamma_exp'] = df['gamma'] * df['rt_open_interest'] * -1 * (CONTRACT_NOTIONAL * 0.01) * 0.01
# VEX (Vanna Exposure) in $, per 1% vol change
df['vanna_exp'] = df['vanna'] * df['rt_open_interest'] * -1 * CONTRACT_NOTIONAL
# DEX (Delta of Volume)
df['delta_exp_flow'] = np.where(df['type'].str.upper() == 'C', df['delta'] * df['volume'], -df['delta'] * df['volume'])


# --- Key Metrics Display ---
st.header(f"Market State at: {selected_ts}")

total_gex = df['gamma_exp'].sum()
total_vex = df['vanna_exp'].sum()
total_dex = df['delta_exp_flow'].sum()

# Calculate Gamma Flip Point
gamma_by_strike = df.groupby('strike')['gamma_exp'].sum().sort_index()
cumulative_gamma = gamma_by_strike.cumsum()
try:
    gamma_flip_point = cumulative_gamma[cumulative_gamma < 0].index[0]
except IndexError:
    gamma_flip_point = "N/A"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total GEX ($)", f"${total_gex:,.0f}")
col2.metric("Total VEX ($)", f"${total_vex:,.0f}")
col3.metric("Gamma Flip Point", f"{gamma_flip_point}")
col4.metric("Net Delta Flow (DEX)", f"{total_dex:,.2f}")


# --- Create Tabs for Different Views ---
tab1, tab2, tab3, tab4 = st.tabs(["Forward Curve", "Exposure by Strike", "Exposure by Expiry", "Time Series Analysis"])

with tab1:
    st.subheader("SOFR Forward Curve")
    forward_curve = df[['expiry_date', 'forward_price']].drop_duplicates().sort_values('expiry_date')
    fig = px.line(forward_curve, x='expiry_date', y='forward_price', title="Forward Price by Expiry", markers=True)
    fig.update_layout(xaxis_title="Expiration Date", yaxis_title="Forward Price (100 - Rate)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Exposure Profile by Strike")
    # Prepare data for stacked bar chart (dealer short calls = negative gamma, short puts = positive gamma)
    df['plot_gamma_exp'] = np.where(df['type'].str.upper() == 'C', -df['gamma_exp'], df['gamma_exp'])
    
    exposure_by_strike = df.groupby(['strike', 'type'])['plot_gamma_exp'].sum().reset_index()
    
    fig = px.bar(
        exposure_by_strike,
        x='strike',
        y='plot_gamma_exp',
        color='type',
        title="Gamma Exposure (GEX) by Strike",
        labels={'strike': 'Strike Price', 'plot_gamma_exp': 'Gamma Exposure ($)'},
        color_discrete_map={'c': 'red', 'p': 'green'}
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Exposure Profile by Expiry")
    df['plot_gamma_exp'] = np.where(df['type'].str.upper() == 'C', -df['gamma_exp'], df['gamma_exp'])
    exposure_by_expiry = df.groupby(['expiry_date', 'type'])['plot_gamma_exp'].sum().reset_index()
    fig = px.bar(
        exposure_by_expiry,
        x='expiry_date',
        y='plot_gamma_exp',
        color='type',
        title="Gamma Exposure (GEX) by Expiry",
        labels={'expiry_date': 'Expiration Date', 'plot_gamma_exp': 'Gamma Exposure ($)'},
        color_discrete_map={'c': 'red', 'p': 'green'}
    )
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Time Series of Total Net Exposure")
    
    # Calculate total net GEX for each timestamp
    time_series_data = []
    for ts, data in all_data.items():
        gex = data['gamma'].multiply(data['rt_open_interest']).multiply(-1).sum()
        gex_notional = gex * (CONTRACT_NOTIONAL * 0.01) * 0.01
        time_series_data.append({'timestamp': ts, 'total_gex': gex_notional})
    
    df_ts = pd.DataFrame(time_series_data).sort_values('timestamp')
    
    fig = px.line(df_ts, x='timestamp', y='total_gex', title="Total Net Gamma Exposure (GEX) Over Time", markers=True)
    fig.update_layout(xaxis_title="Snapshot Time", yaxis_title="Total GEX ($)")
    st.plotly_chart(fig, use_container_width=True)
