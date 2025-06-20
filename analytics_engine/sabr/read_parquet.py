import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from bachelier import bachelier_iv

st.set_page_config(layout="wide", page_title="SOFR Option Chain Diagnostics")
st.title("SOFR Option Chain Diagnostics")

# --- Sidebar: Input parquet path ---
st.sidebar.header("Input Snapshot")
def select_parquet():
    default = "snapshots/20250617/122819/SFRU5_sep.parquet"
    path = st.sidebar.text_input("Parquet file path", value=default)
    if not os.path.isfile(path):
        st.sidebar.error(f"File not found: {path}")
        st.stop()
    return path

parquet_path = select_parquet()

# --- Load raw chain ---
@st.cache_data
def load_chain(path):
    df = pd.read_parquet(path)
    return df

df_raw = load_chain(parquet_path)
st.subheader("Raw Option Chain")
st.dataframe(df_raw)

# --- Extract strike from ticker ---
st.subheader("Parsed Strike Column")
df = df_raw.copy()
# Extract numbers (ints or decimals) from ticker
df['strike'] = df['ticker'].str.extract(r'\b(\d+\.\d+)\b')[0].astype(float)
st.dataframe(df[['ticker','strike']].head(10))

# --- Trim edges: drop extreme non‐liquid strikes ---
st.subheader("Trimmed Option Chain")
# Identify liquid strikes: bid>0 & ask>0
liquid = df[(df.bid>0) & (df.ask>0)]
if liquid.empty:
    st.error("No liquid strikes found.")
    st.stop()
lo, hi = liquid.strike.min(), liquid.strike.max()
df_trim = df[(df.strike>=lo) & (df.strike<=hi)].reset_index(drop=True)
st.write(f"Trimming strikes to range [{lo}, {hi}] → {len(df_trim)} rows")
st.dataframe(df_trim)

# --- Compute mid_price ---
st.subheader("Mid Prices (bid/ask mid)")
df_trim['mid_price'] = np.where(
    (df_trim.bid>0)&(df_trim.ask>0),
    0.5*(df_trim.bid + df_trim.ask),
    np.nan
)
st.dataframe(df_trim[['ticker','strike','bid','ask','mid_price']])

# --- OTM Filter (using existing type column) ---
st.subheader("OTM Filtered Chain")

# Forward price
F = float(df_trim.future_px.iloc[0])
st.write(f"Forward price: {F}")

# Make sure your type column matches exactly 'C' or 'P'
# If it's lower-case, convert:
df_trim['type'] = df_trim['type'].str.upper()

# Now filter OTM correctly
df_otm = df_trim[
    ((df_trim['type'] == 'C') & (df_trim['strike'] <= F)) |   # calls at-or-above forward
    ((df_trim['type'] == 'P') & (df_trim['strike'] >= F))     # puts  at-or-below forward
].reset_index(drop=True)

st.write(f"OTM strikes → {len(df_otm)} rows")
st.dataframe(df_otm)


# --- IV Inversion ---
st.subheader("Implied Volatility via Bachelier")
# Compute time to expiry T
snap_dt = datetime.strptime(df_otm.snapshot_ts.iloc[0], '%Y%m%d %H%M%S')
expiry = pd.to_datetime(df_otm.expiry_date.iloc[0]).date()
T = (expiry - snap_dt.date()).days / 365.0
st.write(f"Time to expiry (T): {T:.4f} yrs")
# Invert
df_otm['iv'] = df_otm.apply(
    lambda r: bachelier_iv(F, T, r.strike, r.mid_price) if not np.isnan(r.mid_price) else np.nan,
    axis=1
)
st.dataframe(df_otm[['strike','mid_price','iv']])

# --- Smile Plot ---
st.subheader("Market Volatility Smile (OTM)")
chart_df = df_otm.set_index('strike')['iv']
st.line_chart(chart_df)

# --- SABR Calibration ---
from analytics_engine.sabr.sabr_v2 import calibrate_sabr_full, sabr_vol_normal
st.subheader("SABR Full Calibration")
# Prepare arrays
strikes = df_otm.strike.values
vols = df_otm.iv.values
mask = ~np.isnan(vols)
strikes_fit = strikes[mask]
vols_fit = vols[mask]
# Calibrate
a_alpha, a_beta, a_rho, a_nu = calibrate_sabr_full(strikes_fit, vols_fit, F, T)
st.write(f"Params → alpha={a_alpha:.5f}, beta={a_beta:.3f}, rho={a_rho:.3f}, nu={a_nu:.5f}")
# Model vols
df_otm['model_iv'] = df_otm.strike.apply(lambda K: sabr_vol_normal(F, K, T, a_alpha, a_beta, a_rho, a_nu))
st.subheader("Market vs Model IV")
st.line_chart(df_otm.set_index('strike')[['iv','model_iv']])
