# mdl_precalib.py
import pandas as pd
import numpy as np
from datetime import datetime
from iv_utils import implied_vol

def prepare_calibration(path: str):
    """
    Loads a parquet snapshot, filters for liquid OTM options,
    and calculates implied volatility from mid-prices.
    """
    # 1) Read & basic filters
    df = pd.read_parquet(path)
    df = df[(df.bid > 0) & (df.ask > 0)].copy()
    if df.empty:
        return None, None, None

    # 1.5) Extract strike from ticker
    df['strike'] = df['ticker'].str.extract(r'\b(\d+\.\d+)\b')[0].astype(float)

    # 2) Future price, Time to Expiry (T), and type
    F = float(df['future_px'].iloc[0])
    snap_dt = datetime.strptime(df['snapshot_ts'].iloc[0], '%Y%m%d %H%M%S')
    expiry = pd.to_datetime(df['expiry_date'].iloc[0]).date()
    T = (expiry - snap_dt.date()).days / 365.0
    df['type'] = df['type'].str.upper()

    # 3) OTM selection and mid-price calculation
    df_otm = df[((df['type'] == 'C') & (df['strike'] >= F)) |
                ((df['type'] == 'P') & (df['strike'] < F))].reset_index(drop=True)
    df_otm['mid_price'] = (df_otm['bid'] + df_otm['ask']) / 2.0
    if df_otm.empty:
        return None, None, None

    # 4) Invert to get Implied Volatility (IV)
    df_otm['iv'] = df_otm.apply(
        lambda r: implied_vol(
            F=F, T=T, K=r['strike'], price=r['mid_price'],
            opt_type=r['type'], engine='black76'
        ) if not np.isnan(r['mid_price']) else np.nan,
        axis=1
    )
    
    # Return the processed DataFrame and key parameters
    return df_otm, F, T