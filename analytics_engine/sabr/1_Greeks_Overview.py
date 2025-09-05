import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIGURATION ---
GREEKS_CACHE_DIR = 'analytics_results/greeks_exposure'
CONTRACT_NOTIONAL = 1_000_000
greeks_cols = ['gamma', 'vanna', 'delta', 'charm', 'theta', 'vega']

st.set_page_config(layout="wide", page_title="Greeks Exposure")

# --- DATA LOADING (FROM CACHE) ---
@st.cache_data(show_spinner="Loading exposure data...")
def load_data_from_cache():
    """Loads the entire greeks exposure dataset from the cache."""
    if not os.path.exists(GREEKS_CACHE_DIR):
        return pd.DataFrame()
    try:
        df = pd.read_parquet(GREEKS_CACHE_DIR, engine='pyarrow')
        # Ensure correct data types after loading
        df['snapshot_ts'] = pd.to_datetime(df['snapshot_ts'], format='%Y%m%d %H%M%S')
        df['expiry_date'] = pd.to_datetime(df['expiry_date']).dt.date
        return df
    except Exception as e:
        st.error(f"Failed to load cache from '{GREEKS_CACHE_DIR}'. Did you run the build script? Error: {e}")
        return pd.DataFrame()

# Load all data
all_data = load_data_from_cache()

# --- Main Application ---
if all_data.empty:
    st.error(f"No data found in the cache directory: '{GREEKS_CACHE_DIR}'. Please run `build_greeks_cache.py` first.")
    st.stop()

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### Dashboard Controls")
    
    timestamps = sorted(all_data['snapshot_ts'].unique())
    selected_ts = st.select_slider(
        "Select Snapshot Time", 
        options=timestamps, 
        value=timestamps[-1],
        format_func=lambda ts: pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S')
    )

# Filter data for the selected timestamp
df = all_data[all_data['snapshot_ts'] == selected_ts].copy()

# --- Display Logic (Copied from your original script, now runs instantly) ---
# ... (The rest of your display code: metrics, tabs, plots) ...
# ... I will paste the main components here for completeness ...



# --- Market State Block (Corrected) ---
st.header(f"Dealer Exposure for Snapshot: {pd.to_datetime(selected_ts).strftime('%Y-%m-%d %H:%M:%S')}")

# --- Apply the dealer's short-option perspective for summary metrics ---
# Dealers are short options, so we multiply by -1 to see their exposure.
# e.g., Shorting an option with positive gamma results in negative gamma exposure for the dealer.
total_gex = df['gamma_exp'].sum() * -1
total_vex = df['vanna_exp'].sum() * -1
total_delta_exp = df['delta_exp'].sum() * -1
total_charm_exp = df['charm_exp'].sum() * -1
# Dealers are LONG theta (profit from decay), and theta is naturally negative, so -1 makes it positive.
total_theta_exp = df['theta_exp'].sum() * -1

col1, col2, col3 = st.columns(3)

# GEX (Gamma Exposure): Note the corrected scaling and dealer perspective.
# The value represents the change in the total delta exposure for a 1% move in the underlying.
col1.metric("Total GEX ($ per 1% move)", f"${total_gex:,.0f}",
            help="**Formula:** `sum(Gamma * OI * Notional * 0.01) * -1`. Represents the change in dealer Delta exposure for a +1% move in the underlying. A negative value means dealers must sell into a rally and buy into a drop, increasing volatility.")

# VEX (Vanna Exposure)
col2.metric("Total VEX ($)", f"${total_vex:,.0f}",
            help="**Formula:** `sum(Vanna * OI * Notional * 0.01) * -1`. Change in total dealer Delta for a +1% change in implied volatility.")

# --- CORRECTED GAMMA FLIP POINT LOGIC ---
try:
    # 1. Calculate the dealer's gamma exposure per strike (raw gamma * -1).
    dealer_gamma_by_strike = df.groupby('strike')['gamma_exp'].sum() * -1
    # 2. Calculate the cumulative sum of dealer gamma, sorted by strike.
    cumulative_dealer_gamma = dealer_gamma_by_strike.sort_index().cumsum()
    # 3. Find the first strike where the cumulative dealer gamma becomes positive.
    flip_strike = cumulative_dealer_gamma[cumulative_dealer_gamma > 0].index[0]
    col3.metric("Gamma Flip Point", f"{flip_strike:.2f}",
                help="The strike level where the dealer's aggregate Gamma exposure flips from negative to positive.")
except IndexError:
    # This exception occurs if the cumulative gamma never becomes positive.
    col3.metric("Gamma Flip Point", "N/A", help="Dealer Gamma exposure does not flip to positive in the observed range.")

col1, col2, col3 = st.columns(3)

# Total Delta Exposure
col1.metric("Total Delta Exposure ($)", f"${total_delta_exp:,.0f}",
            help="**Formula:** `sum(Delta * OI * Notional) * -1`. The PnL impact on the dealer's position for a +1 point move in the underlying. A negative value means dealers are net short delta.")

# Total Charm Exposure
col2.metric("Total Charm ($ per day)", f"${total_charm_exp:,.0f}",
            help="**Formula:** `sum(Charm * OI * Notional) * -1`. The daily change in the dealer's total Delta exposure (Delta decay).")

# Total Theta Exposure
col3.metric("Total Theta ($ per day)", f"${total_theta_exp:,.0f}",
            help="**Formula:** `sum(Theta * OI * Notional) * -1`. The daily PnL decay dealers are expected to earn from being short options. Should be positive.")

# --- Create Tabs for Different Views ---
tab_names = ["Forward Curve", "Exposure by Strike", "Exposure by Expiry", 
             "Time Series Analysis", "Expiry Drill-Down", "Strike Drill-Down"]
#if skipped_files:
#    tab_names.append("Skipped Files Log")
tabs = st.tabs(tab_names)

with tabs[0]: # Forward Curve
    st.subheader("SOFR Forward Curve")
    forward_curve = df[['expiry_date', 'forward_price']].drop_duplicates().sort_values('expiry_date')
    st.plotly_chart(px.line(forward_curve, title="Forward Price by Expiry",
                            x='expiry_date', y='forward_price', markers=True), 
                            use_container_width=True)

def create_exposure_plot(data, group_by_col, greek, show_net): # Renamed to match your code
    """
    Generates exposure plots with correct logic for dealer (net short) perspective.
    - Gamma, Vega are always negative risks for dealers.
    - Theta is always a positive PnL for dealers.
    - Delta, Vanna, Charm are directional.
    """
    exp_col = f'{greek}_exp'
    df_viz = data.copy()

    # --- Apply Dealer Perspective (-1) to all exposures for visualization ---
    # This ensures we are always plotting from the market-maker's viewpoint.
    df_viz[exp_col] = df_viz[exp_col] * -1

    if show_net:
        # --- NET VIEW ---
        exposure_data = df_viz.groupby(group_by_col)[exp_col].sum().reset_index()
        exposure_data['color'] = np.where(exposure_data[exp_col] >= 0, '#2ca02c', '#d62728') # Green for positive, Red for negative
        
        fig = px.bar(exposure_data, x=group_by_col, y=exp_col,
                     title=f"Net Dealer {greek.capitalize()} Exposure",
                     color='color', 
                     color_discrete_map={'#2ca02c':'#2ca02c', '#d62728':'#d62728'}) # Pass through colors
        fig.update_layout(showlegend=False)

    else:
        # --- CALL vs PUT VIEW ---
        # Create a 'plot_exp' column for visualization purposes only.
        if greek in ['gamma', 'vega']:
            # For dealers, Gamma and Vega are always negative exposures.
            # We take the absolute value and make it negative to stack them visually on one side.
            df_viz['plot_exp'] = df_viz[exp_col].abs()
        elif greek == 'theta':
            # For dealers, Theta is always a positive PnL (they collect decay).
            # We plot the true (positive) exposure.
            df_viz['plot_exp'] = df_viz[exp_col]
        else: # Delta, Vanna, Charm are directional
            # Short Calls -> Neg Delta/Vanna. Short Puts -> Pos Delta/Vanna.
            # We plot their true directional exposure.
            df_viz['plot_exp'] = df_viz[exp_col]
        
        # --- FIX: Aggregate both plot_exp and the original exp_col ---
        # This ensures the column for hover_data is available in the final DataFrame.
        exposure_data = df_viz.groupby([group_by_col, 'type']).agg({
            'plot_exp': 'sum',
            exp_col: 'sum'  # This preserves the original exposure column
        }).reset_index()
        
        fig = px.bar(exposure_data, x=group_by_col, y='plot_exp', color='type',
                     title=f"Dealer {greek.capitalize()} Exposure by Option Type",
                     hover_data={exp_col: ':.2f'}) # This will now work correctly
        
        # --- MODIFICATION FOR OVERLAY AND TRANSPARENCY ---
        fig.update_layout(barmode='overlay')
        fig.for_each_trace(
            lambda t: t.update(marker_color='#2ca02c', opacity=0.65) if t.name.upper() == 'P' 
            else t.update(marker_color='#d62728', opacity=0.35)
        )
        # --- END MODIFICATION ---

        # Manually set colors: Puts Green, Calls Red
        fig.for_each_trace(
            lambda t: t.update(marker_color='#d62728') if t.name.upper() == 'P' else t.update(marker_color='#2ca02c')
        )
    
    # Common layout updates
    fig.update_yaxes(title_text=f"Dealer {greek.capitalize()} Exposure ($)")
    if group_by_col == 'expiry_date':
        fig.update_xaxes(type='category', tickformat='%Y-%m-%d') # Format date ticks
    return fig

# --- REQ 1 & 2: Define a reusable function for the new plot style ---
def create_exposure_plot_old(data, group_by_col, greek, show_net):
    exp_col = f'{greek}_exp'
    
    # Define plotting exposure based on Greek conventions for visual separation
    if greek in ['gamma', 'vanna']:
        data['plot_exp'] = np.where(data['type'].str.upper() == 'C', -data[exp_col].abs(), data[exp_col].abs())
    elif greek in ['delta']:
        # Short Call (positive delta) -> negative exposure. Short Put (negative delta) -> positive exposure
        data['plot_exp'] = data[exp_col] * -1
    else: # Theta, Vega, Charm
        data['plot_exp'] = data[exp_col]

    if show_net:
        exposure_data        = data.groupby(group_by_col)['plot_exp'].sum().reset_index()
        exposure_data['+/-'] = np.where(exposure_data['plot_exp'] >= 0, 'pos', 'neg')
        fig = px.bar(exposure_data, title=f"Net {greek.capitalize()} Exposure",
                     x=group_by_col, y='plot_exp', 
                     color='+/-', color_discrete_map={'neg':'green', 'pos':'red'})
        
    else:
        exposure_data = data.groupby([group_by_col, 'type'])['plot_exp'].sum().reset_index()
        fig = px.bar(exposure_data, title=f"{greek.capitalize()} Exposure by Option Type",
                     x=group_by_col, y='plot_exp', color='type'
                     )
        fig.for_each_trace(lambda t: t.update(marker_color='green') 
                            if t.name.upper() == 'P' 
                            else t.update(marker_color='red')
                            )
        
    if group_by_col == 'expiry_date':
        fig.update_xaxes(type='category')    
    return fig

# --- EXPOSURE PLOT TO CORRECTLY DISPLAY VEGA, THETA, CHARM ---                 ### REVISIT ### ## LOGIC ## ## ERROR ##
def create_exposure_plot_new_not_done(data, group_by_col, greek, show_net): 
    exp_col = f'{greek}_exp'
    
    if show_net:
        # --- CORRECT NET EXPOSURE LOGIC ---
        # 1. Group by the desired column and sum the TRUE, UNMODIFIED exposure column.
        exposure_data = data.groupby(group_by_col)[exp_col].sum().reset_index()
        
        # 2. Create a color column based on the sign of the TRUE net exposure.
        exposure_data['color'] = np.where(exposure_data[exp_col] >= 0, 'green', 'red')
        
        # 3. Plot the net exposure, using the new color column.
        fig = px.bar(exposure_data, x=group_by_col, y=exp_col,
                     title=f"Net {greek.capitalize()} Exposure",
                     color='color', 
                     color_discrete_map={'green':'#2ca02c', 'red':'#d62728'})
        fig.update_layout(showlegend=False)
    else:
        # --- SEPARATE CALL/PUT VISUALIZATION LOGIC ---
        # Use a temporary copy for visualization to avoid changing the original data
        df_viz = data.copy()
        
        # 1. This 'plot_exp_viz' column is ONLY for visual separation on the chart.
        if greek in ['gamma', 'vanna', 'delta']:
            # For these greeks, C/P have opposing exposures, so we plot them on opposite sides of zero.
            df_viz['plot_exp_viz'] = np.where(df_viz['type'].str.upper() == 'P', df_viz[exp_col].abs(), -df_viz[exp_col].abs())
        else: # Vega, Theta, Charm
            # For these greeks, short C/P have same-signed exposure. We plot their true values.
            df_viz['plot_exp_viz'] = df_viz[exp_col]
        
        # 2. Group by type to get separate C/P values for the visual plot.
        exposure_data = df_viz.groupby([group_by_col, 'type'])['plot_exp_viz'].sum().reset_index()
        
        # 3. Plot using the visual exposure column, but display the TRUE exposure value on hover.
        fig = px.bar(exposure_data, x=group_by_col, y='plot_exp_viz', color='type',
                     title=f"{greek.capitalize()} Exposure by Option Type",
                     hover_data={exp_col: ':.2f'}) # Show true value from the original column on hover
        
        # 4. Manually and robustly set the colors for the 'C' and 'P' traces.
        fig.for_each_trace(
            lambda t: t.update(marker_color='#2ca02c') if t.name.upper() == 'P' else t.update(marker_color='#d62728')
        )
    
    # Common layout updates
    fig.update_yaxes(title_text=f"{greek.capitalize()} Exposure ($)")
    if group_by_col == 'expiry_date':
        fig.update_xaxes(type='category')
    return fig

with tabs[1]: # Exposure by Strike
    c1, c2 = st.columns([1, 3])
    greek_strike = c1.selectbox("Select Greek", greeks_cols, key='strike_greek')
    # "Show Net" checkbox
    show_net_strike = c2.checkbox("Show Net Exposure", value=False, key='strike_net')
    
    fig = create_exposure_plot(df, 'strike', greek_strike, show_net_strike)
    
    # --- Get the average forward for context ---                                   ### REVISIT ### ## FEATURE ## #need to show a range of forward (curve)
    avg_fwd = df['forward_price'].mean()
    fig.add_vline(x=avg_fwd, line_width=2, line_dash="dash", line_color="white", 
                  annotation_text=f"Avg Fwd={avg_fwd:.2f}", annotation_position="top")                  
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]: # Exposure by Expiry
    c1, c2 = st.columns([1, 3])
    greek_expiry    = c1.selectbox("Select Greek", greeks_cols, key='expiry_greek')
    show_net_expiry = c2.checkbox("Show Net Exposure", value=False, key='expiry_net')
    
    fig = create_exposure_plot(df, 'expiry_date', greek_expiry, show_net_expiry)
    st.plotly_chart(fig, use_container_width=True)


with tabs[3]: # Time Series Analysis
    st.subheader("Time Series of Total Net Exposures")

    # --- New, Efficient Time Series Calculation ---
    # Group the entire dataset by the timestamp column and sum the pre-calculated exposure columns.
    df_ts = all_data.groupby('snapshot_ts')[
        ['gamma_exp', 'vanna_exp', 'charm_exp', 'theta_exp', 'delta_exp', 'vega_exp']
    ].sum().reset_index().rename(columns={'snapshot_ts': 'timestamp'})

    if not df_ts.empty:
        df_ts = df_ts.sort_values('timestamp')
        # Define greeks_cols here if it's not defined globally
        greeks_cols = ['gamma', 'vanna', 'delta', 'charm', 'theta', 'vega']
        for greek in greeks_cols:
            exp_col = f'{greek}_exp'
            st.plotly_chart(px.line(df_ts, title=f"Total Net {greek.capitalize()} Exposure ($) Over Time",
                                    x='timestamp', y=exp_col, markers=True), use_container_width=True)
    else:
        st.warning("Not enough data to build time series charts.")

# --- Expiry Drill-Down ---
with tabs[4]:
    st.subheader("Greeks Exposure by Strike for a Single Expiry")
    expiries = sorted(df['expiry_date'].unique())
    selected_expiry = st.selectbox("Select Expiry to Analyze", expiries)
    
    df_drill_exp = df[df['expiry_date'] == selected_expiry]
    
    c1, c2 = st.columns([1, 3])
    greek_drill_exp = c1.selectbox("Select Greek", greeks_cols, key='drill_exp_greek')
    show_net_drill_exp = c2.checkbox("Show Net Exposure", value=False, key='drill_exp_net')
    
    fig = create_exposure_plot(df_drill_exp, 'strike', greek_drill_exp, show_net_drill_exp)
    
    # --- Mark forward price for the selected expiry ---
    fwd_price = df_drill_exp['forward_price'].iloc[0]
    fig.add_vline(x=fwd_price, line_width=2, line_dash="dash", line_color="white", 
                  annotation_text=f"Fwd={fwd_price:.2f}", annotation_position="top")
    
    st.plotly_chart(fig, use_container_width=True)

# --- Strike Drill-Down Tab ---                                 ### REVISIT ### ## UI ## ## ERROR ##
with tabs[5]:
    st.subheader("Greeks Exposure by Expiry for a Single Strike")
    strikes         = sorted(df['strike'].unique())
    selected_strike = st.select_slider("Select Strike to Analyze", options=strikes)
    
    df_drill_str = df[df['strike'] == selected_strike]

    c1, c2 = st.columns([1, 3])
    greek_drill_str    = c1.selectbox("Select Greek",    greeks_cols, key='drill_str_greek')
    show_net_drill_str = c2.checkbox("Show Net Exposure", value=False, key='drill_str_net')
    
    fig = create_exposure_plot(df_drill_str, 'expiry_date', greek_drill_str, show_net_drill_str)
    st.plotly_chart(fig, use_container_width=True)

# --- Skipped files log showed in the last tab ---
#if skipped_files:
#    with tabs[6]:
#        st.header("Log of Skipped or Failed Snapshots")
#        for reason, files in sorted(skipped_files.items()):
#            with st.expander(f"{reason} ({len(files)} files)"):
#                st.json(files)

# ... and so on for the rest of your metrics and tabs ...
# The plotting functions like `create_exposure_plot` can be kept as they are.
# Just ensure they are defined or imported in this script.
st.subheader("Data for selected timestamp")
st.dataframe(df)

# You would continue to build out the tabs and plots here, using the `df` DataFrame.
# Since the `df` is already fully calculated, all plots will render very quickly.