import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from pathlib import Path

# Import discovery function to find snapshot folders
from mdl_load import discover_snapshot_files

# --- CONFIGURATION ---
OVERVIEW_CACHE_DIR = "precalibrated_cache/overview"

st.set_page_config(layout="wide")
st.title("Snapshot Health & Sentiment Overview")

# --- Data Loading ---
@st.cache_data(show_spinner="Loading snapshot overview data...")
def load_overview_data(snapshot_folder):
    """Loads a single pre-aggregated metrics file for a given snapshot."""
    sanitized_folder_name = snapshot_folder.replace('\\', '_').replace('/', '_')
    cache_file = Path(OVERVIEW_CACHE_DIR) / f"{sanitized_folder_name}.joblib"
    
    if cache_file.exists():
        return joblib.load(cache_file)
    else:
        return None

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Snapshot Selection")
    file_dict = discover_snapshot_files("snapshots")
    all_folders = sorted(list(file_dict.keys()))
    
    selected_folder = None
    if not all_folders:
        st.warning("No snapshot folders found.")
    else:
        selected_index = st.slider(
            "Select Snapshot Timestamp",
            min_value=0,
            max_value=len(all_folders) - 1,
            value=len(all_folders) - 1, # Default to the most recent
            key='overview_slider'
        )
        selected_folder = all_folders[selected_index]

        # Display the selected timestamp
        sanitized_folder_name = selected_folder.replace('\\', '_').replace('/', '_')
        selected_folder_formatted = pd.to_datetime(sanitized_folder_name, format='%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
        st.markdown(f"**Selected:** `{selected_folder_formatted}`")

# --- Main Panel ---
if selected_folder:
    data = load_overview_data(selected_folder)

    if data is None:
        st.error(f"No pre-calibrated overview data found for this snapshot. Please run `precalibrate.py`.")
    else:
        st.markdown("### Key Metrics")
        
        # --- METRIC CARDS ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Total Volume",
                value=f"{data['total_volume']:,.0f}"
            )
        with col2:
            st.metric(
                label="Total Open Interest",
                value=f"{data['total_oi']:,.0f}"
            )
        with col3:
            st.metric(
                label="Volume P/C Ratio",
                value=f"{data['volume_pc_ratio']:.2f}",
                help="A value > 1 suggests bearish sentiment (more puts traded); < 1 suggests bullish sentiment."
            )
        with col4:
            st.metric(
                label="OI P/C Ratio",
                value=f"{data['oi_pc_ratio']:.2f}",
                help="Indicates overall market positioning. > 1 is bearish; < 1 is bullish."
            )
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # --- CHARTS ---
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("ATM Volatility Term Structure")
            if data['term_structure']:
                df_term = pd.DataFrame(data['term_structure']).sort_values('T')
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_term['T'], y=df_term['atm_iv'],
                    mode='lines+markers', name='ATM IV'
                ))
                fig.update_layout(
                    xaxis_title="Time to Maturity (Years)",
                    yaxis_title="Implied Volatility",
                    yaxis_tickformat=".2%",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to plot term structure.")

        with chart_col2:
            st.subheader("25-Delta Risk Reversal")
            if data['risk_reversals']:
                df_rr = pd.DataFrame(data['risk_reversals']).sort_values('T')
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_rr['T'], y=df_rr['rr_25d'],
                    mode='lines+markers', name='Risk Reversal',
                    line=dict(color='orange')
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    xaxis_title="Time to Maturity (Years)",
                    yaxis_title="IV Spread (Call - Put)",
                    yaxis_tickformat=".2%",
                    margin=dict(l=20, r=20, t=40, b=20),
                    help="Measures the relative cost of upside vs. downside options. Negative values indicate higher demand for puts."
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to plot risk reversal.")

else:
    st.info("Please select a snapshot from the sidebar to view its metrics.")
