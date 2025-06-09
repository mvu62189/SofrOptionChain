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
logging.getLogger().addHandler(logging.StreamHandler())

# ---------------- Config ----------------
root_futures = ['SFRM5', 'SFRU5', 'SFRZ5', 
                'SFRH6', 'SFRM6', 'SFRU6', 'SFRZ6',
                'SFRH7', 'SFRM7', 'SFRU7', 'SFRZ7',
                'SFRH8', 'SFRM8', 'SFRU28', 'SFRZ28',
                'SFRH29', 'SFRM6', 'SFRU6', 'SFRZ6']  # Extend this list as needed
fields = ['Ticker', 'type', 'opt_strike_px', 'bid', 'ask', 'last_price', 'volume', 'ivol_ask']

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
    for root in root_futures:
        tickers = get_option_tickers(root)
        if not tickers:
            continue

        calls = [t for t in tickers if 'C ' in t]
        puts  = [t for t in tickers if 'P ' in t]

        try:
            df_calls = blp.bdp(calls, fields).reset_index().rename(columns={'index': 'Ticker'})
            df_calls['type'] = 'C'
            print(f"[DEBUG] CALL columns: {df_calls.columns.tolist()}")
            
            print("[DEBUG] Requested fields:", fields)
            df_debug = blp.bdp(calls[:1], fields)
            print("[DEBUG] Returned fields:", df_debug.columns.tolist())

            
            df_puts  = blp.bdp(puts, fields).reset_index().rename(columns={'index': 'Ticker'})
            df_puts['type'] = 'P'
            print(f"[DEBUG] PUT columns: {df_puts.columns.tolist()}")

            print("[DEBUG] Requested fields:", fields)
            df_debug = blp.bdp(puts[:1], fields)
            print("[DEBUG] Returned fields:", df_debug.columns.tolist())

            print(blp.flds('SFRM5C 95.0 Comdty', show='description'))

            
        except Exception as e:
            logging.error(f"Snapshot pull failed for {root}: {e}")
            continue

        df_all = pd.concat([df_calls, df_puts], ignore_index=True)
        df_all = df_all[df_all['bid'].notna() | df_all['ask'].notna()]  # Filter by liquidity

        # Extract strike and expiry
        df_all['Expiry'] = root
        df_all['Strike'] = df_all['Ticker'].str.extract(r'(\d{2,3}\.\d{1,2})').astype(float)
        df_all['Call_IVM'] = df_all.apply(lambda x: x['ivol_ask'] if x['type'] == 'C' else None, axis=1)
        df_all['Put_IVM'] = df_all.apply(lambda x: x['ivol_ask'] if x['type'] == 'P' else None, axis=1)

        # Pivot to merge calls and puts by strike
        merged = df_all.groupby(['Expiry', 'Strike']).agg({
            'Call_IVM': 'max',
            'Put_IVM': 'max',
            'bid': 'max',
            'ask': 'max',
            'last_price': 'max',
            'volume': 'sum',
        }).reset_index()

        merged['IVM_MID'] = merged[['Call_IVM', 'Put_IVM']].mean(axis=1)
        merged['IVM_LOWER'] = merged[['Call_IVM', 'Put_IVM']].min(axis=1)
        merged['IVM_UPPER'] = merged[['Call_IVM', 'Put_IVM']].max(axis=1)

        merged.insert(0, 'timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        all_data.append(merged)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        logging.info(f"Fetched combined snapshot: {final_df.shape[0]} rows")
        return final_df
    else:
        logging.warning("No snapshot data gathered.")
        return None

# ---------------- Save to Parquet ----------------
def save_snapshot(df):
    if df is None or df.empty:
        logging.warning("No data to save.")
        return
    today = datetime.now().strftime('%Y%m%d')
    os.makedirs(f'snapshots/{today}', exist_ok=True)
    path = f'snapshots/{today}/combined.parquet'
    df.to_parquet(path, index=False)
    logging.info(f"Saved snapshot to {path}")

# ---------------- Run job ----------------
def run_snapshot_job():
    logging.info("[START] Snapshot job running...")
    df = fetch_snapshot()
    save_snapshot(df)
    if df is not None:
        logging.info("[READY] Snapshot available for analytics engine.")

if __name__ == '__main__':
    run_snapshot_job()
