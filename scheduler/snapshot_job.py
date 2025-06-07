import os
import logging
from datetime import datetime
from xbbg import blp
import pandas as pd

# ---------------- Setup logging ----------------
os.makedirs('logs', exist_ok=True)
log_path = f'logs/snapshot_log_{datetime.now().strftime("%Y%m%d")}.log'

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
# Also log to terminal
logging.getLogger().addHandler(logging.StreamHandler())

# ---------------- Ticker loader ----------------
def get_option_tickers(root_ticker='SFRM5 Comdty'):
    try:
        chain = blp.bds(root_ticker, 'OPT_CHAIN')
        col = [c for c in chain.columns if 'security' in c.lower()]
        if not col:
            logging.warning("No valid 'security' column found in OPT_CHAIN.")
            return []
        raw = chain[col[0]].dropna().astype(str)
        tickers = [' '.join(s.split()) for s in raw.tolist()]
        logging.info(f"Fetched {len(tickers)} tickers from OPT_CHAIN.")
        return tickers
    except Exception as e:
        logging.error(f"Fetching tickers failed: {e}")
        return []

# ---------------- Snapshot loader ----------------
def fetch_snapshot():
    tickers = get_option_tickers()
    if not tickers:
        logging.warning("No tickers found. Skipping snapshot.")
        return None

    fields = [
        'OPT_STRIKE_PX', 'MONEYNESS', 'BID', 'ASK',
        'LAST_PRICE', 'VOLUME', 'OPEN_INT',
        'DELTA', 'GAMMA', 'VEGA', 'THETA', 'RHO'
    ]

    try:
        df = blp.bdp(tickers, fields).reset_index().rename(columns={"index": "Ticker"})
        df.insert(0, 'timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logging.info(f"Snapshot fetched: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except Exception as e:
        logging.error(f"Snapshot pull failed: {e}")
        return None

# ---------------- Snapshot saver ----------------
def save_snapshot(df: pd.DataFrame):
    if df is None or df.empty:
        logging.warning("Snapshot empty. Nothing to save.")
        return None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('snapshots', exist_ok=True)
    path = f'snapshots/snapshot_{timestamp}.csv'
    df.to_csv(path, index=False)
    logging.info(f"Snapshot saved: {path}")
    return path

# ---------------- Job trigger ----------------
def run_snapshot_job():
    logging.info("Running snapshot job...")
    df = fetch_snapshot()
    path = save_snapshot(df)
    if path:
        logging.info(f"Snapshot available for downstream engine: {path}")


    # Optional: Pass to analytics engine
    # from analytics_engine import run_analysis
    # run_analysis(snapshot_path)
