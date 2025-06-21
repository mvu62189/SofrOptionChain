import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from iv_utils import implied_vol
from sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast, calibrate_sabr_fast_region_weighted ,sabr_vol_normal

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
df['strike'] = (100 - df['strike'])/100
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
df_trim['future_px'] = (100 - df_trim['future_px'])/100
F = float(df_trim.future_px.iloc[0])

st.write(f"Forward price: {F}")

# Make sure your type column matches exactly 'C' or 'P'
# If it's lower-case, convert:
df_trim['type'] = df_trim['type'].str.upper()

# Now filter OTM correctly
df_otm = df_trim[
    ((df_trim['type'] == 'C') & (df_trim['strike'] >= F)) |   # calls at-or-above forward
    ((df_trim['type'] == 'P') & (df_trim['strike'] <= F))     # puts  at-or-below forward
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
    lambda r: implied_vol(
        F=    float(r.future_px),
        T=    T,
        K=    r.strike,
        price=r.mid_price,
        opt_type=r.type,          # 'C' or 'P'
        engine='bachelier'
    ) if not np.isnan(r.mid_price) else np.nan,
    axis=1
)

df_otm = df_otm[df_otm.iv < 3.0]   # drop iv >300bp

liquid = df_otm[df_otm.volume > 0]
if liquid.empty:
    raise ValueError("No OTM quotes with positive volume!")

lo = liquid.strike.min()
hi = liquid.strike.max()

df_otm = df_otm[(df_otm.strike >= lo) & (df_otm.strike <= hi)]


st.dataframe(df_otm[['strike','mid_price','iv']])

# --- Smile Plot ---
st.subheader("Market Volatility Smile (OTM)")
chart_df = df_otm.set_index('strike')['iv']
st.line_chart(chart_df)

# --- SABR Calibration ---

st.subheader("SABR Full Calibration")
# Prepare arrays
strikes = df_otm.strike.values
vols = df_otm.iv.values
mask = ~np.isnan(vols)
strikes_fit = strikes[mask]
vols_fit = vols[mask]
# Calibrate
params = calibrate_sabr_full(strikes_fit, vols_fit, F, T)
st.write(f"Params → alpha={params[0]:.5f}, beta={params[1]:.3f}, rho={params[2]:.3f}, nu={params[3]:.5f}")
m_params = calibrate_sabr_fast_region_weighted(strikes_fit, vols_fit, F, T, params, call_weight=5.0, put_weight=1.0)

# add manual sliders
st.sidebar.header("Manual SABR parameters")
alpha = st.sidebar.slider("α", min_value=1e-4, max_value=5.0, value=float(m_params[0]), step=1e-4)
beta  = st.sidebar.slider("β", min_value=0.0,  max_value=1.0, value=float(m_params[1]), step=1e-3)
rho   = st.sidebar.slider("ρ", min_value=-0.999, max_value=0.999, value=float(m_params[2]), step=1e-003)
nu    = st.sidebar.slider("ν", min_value=1e-4,  max_value=5.0, value=float(m_params[3]), step=1e-4)



# Model vols
df_otm['model_iv'] = df_otm.strike.apply(lambda K: sabr_vol_normal(F, K, T, params[0], params[1], params[2], params[3]))
df_otm['model_iv_manual'] = df_otm.strike.apply(lambda K: sabr_vol_normal(F, K, T, alpha, beta, rho, nu))

# --- Display SABR params ---    
st.subheader("SABR Parameters")
st.write(f"Full calibration → α={params[0]:.5f}, β={params[1]:.3f}, ρ={params[2]:.3f}, ν={params[3]:.5f}")
st.write(f"Fast calibration → α={m_params[0]:.5f}, β={m_params[1]:.3f}, ρ={m_params[2]:.3f}, ν={m_params[3]:.5f}")
st.write(f"Manual calibration → α={alpha:.5f}, β={beta:.3f}, ρ={rho:.3f}, ν={nu:.5f}")

st.subheader("Market vs Model IV vs Manual Model IV")
st.line_chart(df_otm.set_index('strike')[['iv','model_iv', 'model_iv_manual']])

# ─── Recalibration Button ────────────────────────────────────────────────────
if st.sidebar.button("Recalibrate around manual guess"):
    # Recalibrate using the manual params
    strikes_fit = df_otm.strike.values
    vols_fit = df_otm.iv.values
    mask = ~np.isnan(vols_fit)
    strikes_fit = strikes_fit[mask]
    vols_fit = vols_fit[mask]
    
    with st.spinner("Running fast SABR from manual seed…"):
       # Fast recalibration
        alpha, beta, rho, nu = calibrate_sabr_fast_region_weighted(
            strikes_fit, vols_fit, F, T,
            init_params=np.array([alpha, beta, rho, nu]), call_weight=5.0, put_weight=1.0
        )
    
    st.write(f"Recalibrated → α={alpha:.5f}, β={beta:.3f}, ρ={rho:.3f}, ν={nu:.5f}")
    
    # Update model IVs
    df_otm['model_iv_manual'] = df_otm.strike.apply(
        lambda K: sabr_vol_normal(F, K, T, alpha, beta, rho, nu)
    )

    st.line_chart(df_otm.set_index('strike')[['iv', 'model_iv_manual']])