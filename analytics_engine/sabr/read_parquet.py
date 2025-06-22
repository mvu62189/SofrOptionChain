#read_parquet.py

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from iv_utils import implied_vol
from sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast, calibrate_sabr_fast_region_weighted ,sabr_vol_normal
from sabr_rnd import price_from_sabr, second_derivative, compute_rnd

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
# df['strike'] = (100 - df['strike'])/100
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
# df_trim['future_px'] = (100 - df_trim['future_px'])/100
F = float(df_trim.future_px.iloc[0])

df_trim.style.format({'future_px': '{:.5f}'})

st.write(f"Forward price: {F:.5f}")


# Make sure type column matches exactly 'C' or 'P'
# If it's lower-case, convert:
df_trim['type'] = df_trim['type'].str.upper()

# Now filter OTM correctly
df_otm = df_trim[
    ((df_trim['type'] == 'C') & (df_trim['strike'] > F)) |   # calls at-or-above forward
    ((df_trim['type'] == 'P') & (df_trim['strike'] < F))     # puts  at-or-below forward
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

df_otm = df_otm[df_otm.iv < 100]   # drop iv >300bp

liquid = df_otm[df_otm.volume > 0]
if liquid.empty:
    raise ValueError("No OTM quotes with positive volume!")

lo = liquid.strike.min()
hi = liquid.strike.max()

df_otm = df_otm[(df_otm.strike >= lo-0.052) & (df_otm.strike <= hi+0.052)]

df_otm['spread'] = df_otm['ask'] - df_otm['bid']
df_otm = df_otm[df_otm.spread <= 0.012]  # filter out zero spreads

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
alpha_f, beta_f, rho_f, nu_f = calibrate_sabr_full(strikes_fit, vols_fit, F, T)
st.write(f"Params → alpha={alpha_f:.5f}, rho={rho_f:.3f}, nu={nu_f:.5f}")
alpha_fast, beta_fast, rho_fast, nu_fast = calibrate_sabr_fast(strikes_fit, vols_fit, F, T, init_params=(alpha_f, beta_f, rho_f, nu_f))

# add manual sliders
st.sidebar.header("Manual SABR parameters")
alpha = st.sidebar.slider("α", min_value=1e-4, max_value=5.0, value=float(alpha_fast), step=1e-4)
rho   = st.sidebar.slider("ρ", min_value=-0.999, max_value=0.999, value=float(rho_fast), step=1e-3)
nu    = st.sidebar.slider("ν", min_value=1e-4, max_value=5.0, value=float(nu_fast), step=1e-4)


# Model vols
df_otm['model_iv']        = sabr_vol_normal(F, df_otm.strike.values, T, alpha_f, rho_f, nu_f)
df_otm['model_iv_manual'] = sabr_vol_normal(F, df_otm.strike.values, T, alpha, rho, nu)

# --- Display SABR params ---    
st.subheader("SABR Parameters")
st.write(f"Full calibration → α={alpha_f:.5f}, ρ={rho_f:.3f}, ν={nu_f:.5f}")
st.write(f"Fast calibration → α={alpha_fast:.5f}, ρ={rho_fast:.3f}, ν={nu_fast:.5f}")
st.write(f"Manual calibration → α={alpha:.5f}, ρ={rho:.3f}, ν={nu:.5f}")

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
        alpha, beta, rho, nu = calibrate_sabr_fast(
            strikes_fit, vols_fit, F, T,
            init_params=np.array([alpha, beta_f, rho, nu])
        )
    
    st.write(f"Recalibrated → α={alpha:.5f}, ρ={rho:.3f}, ν={nu:.5f}")
    
    # Update model IVs
    df_otm['model_iv_manual'] = df_otm.strike.apply(
        lambda K: sabr_vol_normal(F, K, T, alpha, rho, nu)
    )

    st.line_chart(df_otm.set_index('strike')[['iv', 'model_iv_manual']])


# --- RND: SABR vs Market ----------------------------------------
st.subheader("Risk-Neutral Density (RND) Comparison")

# 1. RND from SABR (fast calibration)
rnd_strikes = df_otm.strike.values
rnd_sabr = compute_rnd(
    strikes=rnd_strikes, F=F, T=T,
    alpha=alpha_fast, rho=rho_fast, nu=nu_fast
)

# 2. RND from market (using interpolated Bachelier prices from mid_price)
# We need to interpolate a smooth price function from market data for finite-diff
from scipy.interpolate import interp1d

market_strikes = df_otm.strike.values
market_prices = df_otm.mid_price.values

# Interpolate mid_price to function for finite diff
market_price_func = interp1d(
    market_strikes, market_prices,
    kind='cubic', fill_value="extrapolate", bounds_error=False
)

rnd_market = np.array([max(0, second_derivative(market_price_func, K)) for K in market_strikes])

# Normalize densities for plotting (optional, since only relative shape matters)
if np.sum(rnd_sabr) > 0:
    rnd_sabr = rnd_sabr / np.trapezoid(rnd_sabr, rnd_strikes)
if np.sum(rnd_market) > 0:
    rnd_market = rnd_market / np.trapezoid(rnd_market, market_strikes)

# Add to DataFrame for plotting
df_otm['rnd_sabr'] = rnd_sabr
df_otm['rnd_market'] = rnd_market


# --- Plot RNDs ---
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(rnd_strikes, rnd_market, label="Market-derived RND", lw=2)
ax.plot(rnd_strikes, rnd_sabr, label="SABR-derived RND", lw=2, linestyle="--")
ax.set_xlabel("Strike")
ax.set_ylabel("RND (normalized)")
ax.legend()
ax.set_title("Risk-Neutral Density (RND): Market vs SABR")
st.pyplot(fig)
