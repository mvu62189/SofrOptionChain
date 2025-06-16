import os
import sys
import calendar
import logging
from datetime import datetime
from xbbg import blp
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
# ---------------- Logging Setup ----------------
os.makedirs('logs', exist_ok=True)
log_path = f'logs/snapshot_log_{datetime.now().strftime("%Y%m%d")}.log'

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logging.getLogger().addHandler(logging.StreamHandler())

# ---------------- Config ----------------
root_futures = [
    'SFRM5', 'SFRU5', 'SFRZ5', 
    'SFRH6', 'SFRM6', 'SFRU6', 'SFRZ6',
    'SFRH7', 'SFRM7', 'SFRU7', 'SFRZ7',
    'SFRH8', 'SFRM8', 'SFRU28', 'SFRZ28',
    'SFRH29'
]

MONTH_CODE_MAP = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}


fields = ['opt_strike_px', 'bid', 'ask', 'last_price', 'volume', 'ivol_ask']

# ---------------- Helper Functions ----------------

def third_wed(year, month):
    c = calendar.Calendar()
    wednesdays = [d for d in c.itermonthdates(year, month) if d.weekday() == 2 and d.month == month]
    return wednesdays[2]

def sofr_option_expiry(year, month):
    third_wednesday = third_wed(year, month)
    expiry = third_wednesday - pd.Timedelta(days=5)  # Prior Friday
    holidays = USFederalHolidayCalendar().holidays(start=f'{year}-01-01', end=f'{year}-12-31')
    while expiry in holidays or expiry.weekday() != 4:  # must be Friday and not holiday
        expiry -= pd.Timedelta(days=1)
    return expiry


# ---------------- Ticker Loader ----------------
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

# ---------------- Snapshot Fetch ----------------
def extract_option_code(ticker):
    # Extracts the SFRxxx part from the ticker, e.g. SFRN5 from SFRN5P 90.25 Comdty
    import re
    m = re.match(r'(SFR\w+\d)[CP] ', ticker)
    if m:
        return m.group(1)
    return None

def fetch_snapshot():
    all_data, smile_band_data = [], []

    for root in root_futures:
        tickers = get_option_tickers(root)
        print(f"[DEBUG] {root} tickers: {tickers[:5]} ... total: {len(tickers)}")

        if not tickers:
            continue

        # List all unique SFRxxx codes found in this option chain
        code_map = {}
        for t in tickers:
            code = extract_option_code(t)
            if code:
                code_map.setdefault(code, []).append(t)

        calls = [t for t in tickers if 'C ' in t]
        puts  = [t for t in tickers if 'P ' in t]

        try:
            future_px = blp.bdp([f"{root} Comdty"], ['PX_LAST']).iloc[0, 0]
            print(f"[DEBUG] {root} future price: {future_px}")
        except Exception as e:
            print(f"[ERROR] Could not fetch future price for {root}: {e}")

        try:
            df_calls = blp.bdp(calls, fields).reset_index().rename(columns={'index': 'Ticker'})
            df_calls['type'] = 'C'
            df_calls.columns = [c.lower() for c in df_calls.columns]

            df_puts = blp.bdp(puts, fields).reset_index().rename(columns={'index': 'Ticker'})
            df_puts['type'] = 'P'
            df_puts.columns = [c.lower() for c in df_puts.columns]

            # List all fields returned by Bloomberg for this chain
            print(f"[FIELDS] {root} option chain fields: {list(df_calls.columns)}")

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
        df_all['strike'] = df_all['ticker'].str.extract(r'(\d{2,3}\.\d{1,2})').astype(float)
        df_all['call_ivm'] = df_all.apply(lambda x: x['ivol_ask'] if x['type'] == 'c' else None, axis=1)
        df_all['put_ivm'] = df_all.apply(lambda x: x['ivol_ask'] if x['type'] == 'p' else None, axis=1)
        print(f"[DEBUG] df_all for {root}:\n", df_all.head())

    
        merged = df_all.groupby(['expiry', 'strike']).agg({
            'call_ivm': 'max', 'put_ivm': 'max', 'bid': 'max', 'ask': 'max',
            'last_price': 'max', 'volume': 'sum'
        }).reset_index()

        merged['ivm_mid'] = merged[['call_ivm', 'put_ivm']].mean(axis=1)
        merged['ivm_lower'] = merged[['call_ivm', 'put_ivm']].min(axis=1)
        merged['ivm_upper'] = merged[['call_ivm', 'put_ivm']].max(axis=1)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        merged.insert(0, 'timestamp', timestamp)
        all_data.append(merged)

        # Smile bands
        smile = df_all.copy()
        smile['iv_bid'] = smile['ivol_ask'].where(smile['type'] == 'c')
        smile['iv_ask'] = smile['ivol_ask'].where(smile['type'] == 'c')
        smile['put_iv_bid'] = smile['ivol_ask'].where(smile['type'] == 'p')
        smile['put_iv_ask'] = smile['ivol_ask'].where(smile['type'] == 'p')

        smile_bands = smile.groupby(['expiry', 'strike']).agg({
            'iv_bid': 'min', 'iv_ask': 'max',
            'put_iv_bid': 'min', 'put_iv_ask': 'max',
        }).reset_index()

        smile_bands['call_iv_mid'] = smile_bands[['iv_bid', 'iv_ask']].mean(axis=1)
        smile_bands['put_iv_mid'] = smile_bands[['put_iv_bid', 'put_iv_ask']].mean(axis=1)
        smile_bands.insert(0, 'timestamp', timestamp)
        smile_band_data.append(smile_bands)

        print(f"[DEBUG] merged for {root}:\n", merged.head())
        print(f"[DEBUG] smile_bands for {root}:\n", smile_bands.head())

    if all_data:
        return pd.concat(all_data, ignore_index=True), pd.concat(smile_band_data, ignore_index=True)
    else:
        logging.warning("No snapshot data gathered.")
        return None, None

# ---------------- Save to Cache ----------------
def save_to_cache(df1, df2):
    os.makedirs('cache', exist_ok=True)
    df1.to_parquet('cache/snapshot_latest.parquet', index=False)
    df2.to_parquet('cache/smile_bands_latest.parquet', index=False)
    logging.info("Saved to cache.")

# ---------------- Save to Archive ----------------
def save_to_archive(df1, df2):
    today = datetime.now().strftime('%Y%m%d')
    timestamp = datetime.now().strftime('%H%M%S')
    path = f'snapshots/{today}'
    os.makedirs(path, exist_ok=True)
    df1.to_parquet(f'{path}/snapshot_{timestamp}.parquet', index=False)
    df2.to_parquet(f'{path}/smile_{timestamp}.parquet', index=False)
    logging.info(f"Archived snapshots at {timestamp}.")

# ---------------- Run Job ----------------
def run_snapshot_job():
    logging.info("[START] Snapshot job running...")
    df_snap, df_smile = fetch_snapshot()
    if df_snap is not None and df_smile is not None:
        save_to_cache(df_snap, df_smile)
        save_to_archive(df_snap, df_smile)
        logging.info("[READY] Snapshot available for analytics engine.")

if __name__ == '__main__':
    run_snapshot_job()
