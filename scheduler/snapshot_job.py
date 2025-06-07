# scheduler/snapshot_job.py

import os
from datetime import datetime
from xbbg import blp
import pandas as pd

def get_option_tickers(root_ticker='SFRM5 Comdty'):
    try:
        chain = blp.bds(root_ticker, 'OPT_CHAIN')
        col = [c for c in chain.columns if 'security' in c.lower()]
        if not col:
            return []
        raw = chain[col[0]].dropna().astype(str)
        return [' '.join(s.split()) for s in raw.tolist()]
    except Exception as e:
        print(f"[ERROR] Fetching tickers: {e}")
        return []

def fetch_snapshot():
    tickers = get_option_tickers()
    if not tickers:
        print("[WARN] No tickers found.")
        return None

    fields = [
        'OPT_STRIKE_PX', 'MONEYNESS', 'BID', 'ASK',
        'LAST_PRICE', 'VOLUME', 'OPEN_INT',
        'DELTA', 'GAMMA', 'VEGA', 'THETA', 'RHO'
    ]

    try:
        df = blp.bdp(tickers, fields).reset_index().rename(columns={"index": "Ticker"})
        df.insert(0, 'timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return df
    except Exception as e:
        print(f"[ERROR] Snapshot pull failed: {e}")
        return None

def save_snapshot(df: pd.DataFrame):
    if df is None or df.empty:
        return
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('snapshots', exist_ok=True)
    path = f'snapshots/snapshot_{timestamp}.csv'
    df.to_csv(path, index=False)
    print(f"[INFO] Snapshot saved: {path}")

def run_snapshot_job():
    print("[INFO] Running snapshot job...")
    df = fetch_snapshot()
    save_snapshot(df)