import os
import sys
import logging
from datetime import datetime
from xbbg import blp
import pandas as pd

# ---------------- Setup logging ----------------
os.makedirs('logs', exist_ok=True)
log_path = f'logs/snapshot_log_{datetime.now().strftime("%Y%m%d")}.log'

# Clear existing handlers if re-running in some environments (like Jupyter or reruns)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)


# ---------------- Config ----------------
root_futures = [
    'SFRM5', 'SFRU5', 'SFRZ5', 
    'SFRH6', 'SFRM6', 'SFRU6', 'SFRZ6',
    'SFRH7', 'SFRM7', 'SFRU7', 'SFRZ7',
    'SFRH8', 'SFRM8', 'SFRU28', 'SFRZ28',
    'SFRH29'
]
if '--test' in sys.argv:
    root_futures = ['SFRM5']

fields = ['opt_strike_px', 'bid', 'ask', 'last_price', 'volume', 'ivol_ask']

# ---------------- Ticker loader ----------------
def get_option_tickers(root_ticker):
    try:
        chain = blp.bds(f'{root_ticker} Comdty', 'OPT_CHAIN')
        col = [c for c in chain.columns if 'security' in c.lower()]
        if not col:
            logging.warning(f"No security column for {root_ticker}")
            return []
        raw = chain[col[0]].dropna().astype(str)
        return [' '.join(s.split()) for s in raw.tolist()]
    except Exception as e:
        logging.error(f"Fetching tickers failed for {root_ticker}: {e}")
        return []

# ---------------- Snapshot fetch + merge ----------------
def fetch_snapshot():
    all_data = []
    smile_band_data = []

    for root in root_futures:
        tickers = get_option_tickers(root)
        if not tickers:
            continue

        calls = [t for t in tickers if 'C ' in t]
        puts  = [t for t in tickers if 'P ' in t]

        try:
            df_calls = blp.bdp(calls, fields).reset_index().rename(columns={'index': 'ticker'})
            df_calls['type'] = 'c'
            df_calls.columns = [c.lower() for c in df_calls.columns]

            df_puts = blp.bdp(puts, fields).reset_index().rename(columns={'index': 'ticker'})
            df_puts['type'] = 'p'
            df_puts.columns = [c.lower() for c in df_puts.columns]

            for col in ['bid', 'ask', 'last_price', 'volume', 'ivol_ask']:
                if col not in df_calls.columns:
                    df_calls[col] = pd.NA
                if col not in df_puts.columns:
                    df_puts[col] = pd.NA

        except Exception as e:
            logging.error(f"Snapshot pull failed for {root}: {e}")
            continue

        df_all = pd.concat([df_calls, df_puts], ignore_index=True)

        for col in ['bid', 'ask']:
            if col in df_all.columns:
                count = df_all[col].notna().sum()
                logging.info(f"[COVERAGE] {root} - {col.upper()} present in {count} / {len(df_all)} rows")
            else:
                logging.warning(f"[COVERAGE] {root} - {col.upper()} column missing")

        df_all = df_all[df_all['bid'].notna() | df_all['ask'].notna()]
        df_all[['bid', 'ask']] = df_all[['bid', 'ask']].fillna(0)

        df_all['expiry'] = root
        df_all['strike'] = pd.to_numeric(
            df_all['ticker'].str.extract(r'(\d{2,3}\.\d{1,2})')[0], errors='coerce'
        )

        df_all['call_ivm'] = df_all.apply(lambda x: x['ivol_ask'] if x['type'] == 'c' else None, axis=1)
        df_all['put_ivm']  = df_all.apply(lambda x: x['ivol_ask'] if x['type'] == 'p' else None, axis=1)

        # Merged IVM snapshot
        merged = df_all.groupby(['expiry', 'strike']).agg({
            'call_ivm': 'max',
            'put_ivm': 'max',
            'bid': 'max',
            'ask': 'max',
            'last_price': 'max',
            'volume': 'sum',
        }).reset_index()

        merged['ivm_mid'] = merged[['call_ivm', 'put_ivm']].mean(axis=1)
        merged['ivm_lower'] = merged[['call_ivm', 'put_ivm']].min(axis=1)
        merged['ivm_upper'] = merged[['call_ivm', 'put_ivm']].max(axis=1)
        merged.insert(0, 'timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        all_data.append(merged)

        # Smile bands
        smile_data = df_all.copy()
        smile_data['call_iv_bid'] = smile_data['ivol_ask'].where(smile_data['type'] == 'c', None)
        smile_data['call_iv_ask'] = smile_data['ivol_ask'].where(smile_data['type'] == 'c', None)
        smile_data['put_iv_bid']  = smile_data['ivol_ask'].where(smile_data['type'] == 'p', None)
        smile_data['put_iv_ask']  = smile_data['ivol_ask'].where(smile_data['type'] == 'p', None)

        smile_bands = smile_data.groupby(['expiry', 'strike']).agg({
            'call_iv_bid': 'min',
            'call_iv_ask': 'max',
            'put_iv_bid': 'min',
            'put_iv_ask': 'max',
        }).reset_index()

        smile_bands['call_iv_mid'] = smile_bands[['call_iv_bid', 'call_iv_ask']].mean(axis=1)
        smile_bands['put_iv_mid']  = smile_bands[['put_iv_bid',  'put_iv_ask']].mean(axis=1)
        smile_bands['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        smile_band_data.append(smile_bands)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        smile_df = pd.concat(smile_band_data, ignore_index=True)
        return final_df, smile_df
    else:
        logging.warning("No snapshot data gathered.")
        return None, None

# ---------------- Save to Parquet ----------------
def save_snapshot(df, smile_df):
    if df is None or df.empty:
        logging.warning("No data to save.")
        return
    today = datetime.now().strftime('%Y%m%d')
    os.makedirs(f'snapshots/{today}', exist_ok=True)
    df.to_parquet(f'snapshots/{today}/combined.parquet', index=False)
    logging.info(f"Saved combined snapshot to snapshots/{today}/combined.parquet")
    if smile_df is not None and not smile_df.empty:
        smile_df.to_parquet(f'snapshots/{today}/smile_bands.parquet', index=False)
        logging.info(f"Saved smile bands to snapshots/{today}/smile_bands.parquet")

# ---------------- Run job ----------------
def run_snapshot_job():
    logging.info("[START] Snapshot job running...")
    df, smile_df = fetch_snapshot()
    save_snapshot(df, smile_df)
    if df is not None:
        logging.info("[READY] Snapshot available for analytics engine.")

if __name__ == '__main__':
    run_snapshot_job()
