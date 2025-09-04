import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from pathlib import Path
from scipy.stats import norm
from scipy.interpolate import interp1d

# Import necessary functions from your project
from mdl_load import discover_snapshot_files
from mdl_processing import process_snapshot_file

st.set_page_config(layout="wide")
st.title("Live Snapshot Overview")
st.info("This page calculates key option chain metrics on-the-fly for the selected snapshot.")

# --- HELPER FUNCTION for Delta Calculation ---
def black76_delta(F, K, T, r, sigma, option_type='call'):
    """Calculates Black-76 delta."""
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return np.exp(-r * T) * norm.cdf(d1)
    else: # put
        return np.exp(-r * T) * (norm.cdf(d1) - 1)

# --- ON-THE-FLY DATA PROCESSING FUNCTION ---
@st.cache_data(show_spinner="Calculating metrics for snapshot...")
def load_and_process_snapshot(file_paths_tuple):
    """
    Takes a tuple of file paths for a snapshot, calculates all overview metrics,
    and returns the results. The results are cached based on the input file paths.
    """
    file_paths = list(file_paths_tuple) # Convert tuple back to list

    snapshot_metrics = {
        'total_volume': 0, 'total_oi': 0,
        'call_volume': 0, 'put_volume': 0,
        'call_oi': 0, 'put_oi': 0,
        'term_structure': [], 
        'risk_reversals': [] 
    }
    
    COLUMN_MAPPING = {
        'Option Type': 'option_type',
        'Open Interest': 'open_interest',
        'Volume': 'volume',
    }

    for file_path in file_paths:
        try:
            df = pd.read_parquet(file_path)
            df.rename(columns=COLUMN_MAPPING, inplace=True)

            if 'option_type' not in df.columns:
                st.warning(f"Skipping file: 'option_type' column not found in {os.path.basename(file_path)}.")
                continue

            res, _ = process_snapshot_file(file_path, manual_params={}, model_engine='black76', df_override=df)
            if res is None:
                continue

            # Aggregate totals
            if 'volume' in df.columns:
                snapshot_metrics['total_volume'] += df['volume'].sum()
            if 'open_interest' in df.columns:
                snapshot_metrics['total_oi'] += df['open_interest'].sum()

            # Aggregate call/put specific metrics
            calls_df = df[df['option_type'] == 'C']
            puts_df = df[df['option_type'] == 'P']

            if 'volume' in df.columns:
                snapshot_metrics['call_volume'] += calls_df['volume'].sum()
                snapshot_metrics['put_volume'] += puts_df['volume'].sum()
            
            if 'open_interest' in df.columns:
                snapshot_metrics['call_oi'] += calls_df['open_interest'].sum()
                snapshot_metrics['put_oi'] += puts_df['open_interest'].sum()

            # Term Structure & Skew Metrics
            F, T = res['forward_price'], res['T']
            atm_strike_idx = np.abs(res['strikes'] - F).argmin()
            atm_iv = res['market_iv'][atm_strike_idx]
            
            if not np.isnan(atm_iv):
                snapshot_metrics['term_structure'].append({'T': T, 'atm_iv': atm_iv})

            # Calculate Risk Reversal
            deltas = [black76_delta(F, k, T, 0, iv, 'call') for k, iv in zip(res['strikes'], res['market_iv'])]
            valid_deltas = [(d, iv) for d, iv in zip(deltas, res['market_iv']) if not np.isnan(d) and not np.isnan(iv)]
            
            if len(valid_deltas) > 2:
                delta_vals, iv_vals = zip(*sorted(valid_deltas))
                interp_func = interp1d(delta_vals, iv_vals, bounds_error=False, fill_value=np.nan)
                iv_25d_call, iv_75d_call = interp_func(0.25), interp_func(0.75) # 75d call is a 25d put
                if not np.isnan(iv_25d_call) and not np.isnan(iv_75d_call):
                    rr_25d = iv_25d_call - iv_75d_call
                    snapshot_metrics['risk_reversals'].append({'T': T, 'rr_25d': rr_25d})

        except Exception as e:
            st.error(f"Failed to process file {os.path.basename(file_path)}. Error: {e}")
            continue

    # Final Calculations
    call_vol = snapshot_metrics['call_volume']
    put_vol = snapshot_metrics['put_volume']
    snapshot_metrics['volume_pc_ratio'] = put_vol / call_vol if call_vol > 0 else 0
    
    call_oi = snapshot_metrics['call_oi']
    put_oi = snapshot_metrics['put_oi']
    snapshot_metrics['oi_pc_ratio'] = put_oi / call_oi if call_oi > 0 else 0
    
    return snapshot_metrics

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Snapshot Selector")
    file_dict = discover_snapshot_files("snapshots")
    all_folders = sorted(list(file_dict.keys()))
    
    if not all_folders:
        st.warning("No snapshot folders found.")
        st.stop()

    selected_index = st.slider(
        "Select Snapshot Timestamp",
        min_value=0,
        max_value=len(all_folders) - 1,
        value=len(all_folders) - 1, # Default to the last one
    )
    
    selected_folder = all_folders[selected_index]
    sanitized_folder = selected_folder.replace('\\', '_').replace('/', '_')
    formatted_ts = pd.to_datetime(sanitized_folder, format='%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f"**Selected:** `{formatted_ts}`")


# --- MAIN PANEL ---
files_in_selected_folder = file_dict.get(selected_folder, [])
if not files_in_selected_folder:
    st.warning("Selected snapshot folder is empty.")
else:
    # Pass a tuple to the cached function as it's hashable
    snapshot_data = load_and_process_snapshot(tuple(files_in_selected_folder))

    if not snapshot_data or snapshot_data['total_volume'] == 0:
        st.error("Could not extract valid data from this snapshot. The files might be empty or have an unexpected format.")
    else:
        # --- DISPLAY METRICS ---
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Volume", f"{snapshot_data['total_volume']:,.0f}")
        col2.metric("Total Open Interest", f"{snapshot_data['total_oi']:,.0f}")
        col3.metric("Volume P/C Ratio", f"{snapshot_data['volume_pc_ratio']:.2f}")
        col4.metric("OI P/C Ratio", f"{snapshot_data['oi_pc_ratio']:.2f}")

        st.subheader("Market Sentiment & Structure")
        chart_col1, chart_col2 = st.columns(2)

        # Term Structure Chart
        with chart_col1:
            ts_df = pd.DataFrame(snapshot_data['term_structure']).sort_values('T')
            fig_ts = go.Figure()
            if not ts_df.empty:
                fig_ts.add_trace(go.Scatter(x=ts_df['T'], y=ts_df['atm_iv'], mode='lines+markers', name='ATM IV'))
            fig_ts.update_layout(title="ATM Volatility Term Structure", xaxis_title="Time to Maturity (Years)", yaxis_title="Implied Volatility", height=400)
            st.plotly_chart(fig_ts, use_container_width=True)
        
        # Risk Reversal Chart
        with chart_col2:
            rr_df = pd.DataFrame(snapshot_data['risk_reversals']).sort_values('T')
            fig_rr = go.Figure()
            if not rr_df.empty:
                 fig_rr.add_trace(go.Scatter(x=rr_df['T'], y=rr_df['rr_25d'], mode='lines+markers', name='25d Risk Reversal'))
            fig_rr.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_rr.update_layout(title="Volatility Skew (25-Delta Risk Reversal)", xaxis_title="Time to Maturity (Years)", yaxis_title="Call IV - Put IV", height=400)
            st.plotly_chart(fig_rr, use_container_width=True)
