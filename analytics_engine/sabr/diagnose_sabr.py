import os
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from bachelier import bachelier_price, bachelier_vega
from sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast, sabr_vol_normal
import os
from pathlib import Path
from .iv_utils import implied_vol

# ─── Sidebar: Snapshot Selection ────────────────────────────────────────────
st.title("SABR Calibration Diagnostics")

path = "snapshots/20250617/122819/SFRU5_sep.parquet"

# ─── Load Raw Snapshot ──────────────────────────────────────────────────────
df = pd.read_parquet(path)
st.subheader("Raw Snapshot Data")
st.dataframe(df)

# ─── Filter Liquid Strikes & Compute Mid ────────────────────────────────────
df_liq = df[(df.bid>0)&(df.ask>0)].copy()
def trim_volume_edges(group):
    # Find first and last index where volume is notna
    valid = group['volume'].notna()
    if not valid.any():
        return pd.DataFrame(columns=group.columns)  # No valid volume at all
    first = valid.idxmax()
    last = valid[::-1].idxmax()
    return group.loc[first:last]

df = df.groupby('type', group_keys=False).apply(trim_volume_edges).reset_index(drop=True)

df['strike'] = df['ticker'].str.extract(r'(\d+\.\d+|\d+)').astype(float)

df_liq['mid_price'] = (df.bid + df.ask) / 2.0
st.subheader("Filtered Liquid Data")
st.dataframe(df_liq[['ticker','strike','bid','ask','mid_price','volume']])

# ─── Compute Tenor & Forward ─────────────────────────────────────────────────
F = float(df_liq.future_px.iloc[0])
snap_dt = datetime.strptime(df_liq.snapshot_ts.iloc[0], '%Y%m%d %H%M%S').date()
expiry = pd.to_datetime(df_liq.expiry_date.iloc[0]).date()
T = (expiry - snap_dt).days / 365.0
st.write(f"**Forward (F):** {F:.4f}   &nbsp;&nbsp; **Tenor (T):** {T:.4f} yrs")

# ─── Show IV Inversion (Mid vs Ask) ─────────────────────────────────────────
st.subheader("Implied Vol Comparison (Mid vs Ask)")
demo = df_liq.head(5).copy()
demo['iv_mid'] = demo.apply(
    lambda r: implied_vol(
        F=F,
        T=T,
        K=r.strike,
        price=0.5*(r.bid + r.ask),
        opt_type=r.type,
        engine='bachelier'
    ) if (r.bid>0 and r.ask>0) else np.nan,
    axis=1
)

demo['iv_ask'] = demo.apply(
    lambda r: implied_vol(
        F=F,
        T=T,
        K=r.strike,
        price=r.ask,
        opt_type=r.type,
        engine='bachelier'
    ) if r.ask>0 else np.nan,
    axis=1
)

st.table(demo[['strike','bid','ask','iv_mid','iv_ask']])

# ─── Plot Market Vol Smiles ──────────────────────────────────────────────────
st.subheader("Market Volatility Smiles")
df_liq['iv_mid'] = df_liq.apply(
    lambda r: implied_vol(F, T, r.strike, 0.5*(r.bid + r.ask), r.type, engine='bachelier')
                  if (r.bid>0 and r.ask>0) else np.nan,
    axis=1
)

df_liq['iv_ask'] = df_liq.apply(
    lambda r: implied_vol(F, T, r.strike, r.ask, r.type, engine='bachelier')
                  if r.ask>0 else np.nan,
    axis=1
)

df_plot = df_liq.set_index('strike')[['iv_mid','iv_ask']]
st.line_chart(df_plot)

# ─── Calibration Mode & Parameters ──────────────────────────────────────────
mode = st.sidebar.selectbox("Calibration Mode", ['full','fast','auto'])
params_dir = os.path.join('analytics_results','model_params', snap_dt.split('_')[0])
os.makedirs(params_dir, exist_ok=True)
existing = sorted([f for f in os.listdir(params_dir) if f.endswith('.json')])

use_full = (mode=='full') or (mode=='auto' and not existing)

st.subheader("Calibration")
if use_full:
    st.write("**Running full calibration**")
    params = calibrate_sabr_full(df_liq.strike.values, df_liq.iv_ask.values, F, T)
else:
    st.write("**Running fast calibration**")
    prev = json.load(open(os.path.join(params_dir, existing[-1])))
    params = calibrate_sabr_fast(df_liq.strike.values, df_liq.iv_ask.values, F, T, np.array(prev))
st.write("**Calibrated Parameters:**", params)

# ─── Plot Model vs Market Vol ────────────────────────────────────────────────
st.subheader("Model vs Market Volatility")
model_iv = [sabr_vol_normal(F, K, T, *params) for K in df_liq.strike]
df_liq['model_iv'] = model_iv
st.line_chart(df_liq.set_index('strike')[['iv_ask','model_iv']])

# ─── Compute & Plot RND ──────────────────────────────────────────────────────
st.subheader("Risk-Neutral Density")
strikes = df_liq.strike.values
market_rnd = []
model_rnd = []
h = 1e-2
for i,K in enumerate(strikes):
    vol = df_liq.iv_ask.iloc[i]
    p = bachelier_price(F, K, T, vol)
    p_up = bachelier_price(F, K+h, T, np.interp(K+h, strikes, df_liq.iv_ask))
    p_dn = bachelier_price(F, K-h, T, np.interp(K-h, strikes, df_liq.iv_ask))
    market_rnd.append(max(0,(p_up - 2*p + p_dn)/h**2))

    p_m = df_liq.model_iv.iloc[i]
    p_up_m = bachelier_price(F, K+h, T, np.interp(K+h, strikes, df_liq.model_iv))
    p_dn_m = bachelier_price(F, K-h, T, np.interp(K-h, strikes, df_liq.model_iv))
    model_rnd.append(max(0,(p_up_m - 2*p_m + p_dn_m)/h**2))

df_rnd = pd.DataFrame({'strike':strikes,'market_rnd':market_rnd,'model_rnd':model_rnd}).set_index('strike')
st.line_chart(df_rnd)
