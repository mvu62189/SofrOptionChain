# analytics_engine/sabr/mdl_snapshot_v2.py

import os
import logging
import calendar
from datetime import datetime
import pandas as pd

from pandas.tseries.holiday import USFederalHolidayCalendar
try:
    # Attempt to import the libraries
    from xbbg import blp
    import blpapi
    BLOOMBERG_AVAILABLE = True
except ImportError:
    # If the import fails, set a flag
    BLOOMBERG_AVAILABLE = False
    print("Bloomberg libraries not found. Running in local mode with historical data.")

# ---------------- Config ----------------

MONTH_CODE_MAP = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,

                'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}


FIELDS = ['opt_strike_px', 'bid', 'ask', 'mid', 'ivol_mid_rt', 'last_price', 'volume', 'rt_open_interest', 'rt_open_int_dt', 'open_int_change']

ROOT_FUTURES = [

    'SFRU5', 'SFRZ5', 'SFRH6', 'SFRM6', 'SFRU6', 'SFRZ6',

    'SFRH7', 'SFRM7', 'SFRU7', 'SFRZ7', 'SFRH8', 'SFRM8',

    'SFRU28', 'SFRZ28', 'SFRH29', 'SFRM29', 'SFRU29', 'SFRZ29'
]

# ---------------- Logging ----------------
os.makedirs('logs', exist_ok=True)
log_path = f'logs/snapshot_log_{datetime.now().strftime("%Y%m%d")}.log'

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Ensure logger level is set

# Define a standard formatter
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')



# 1. Check for and add a FileHandler if none exists
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# 2. Check for and add a StreamHandler if none exists
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

# ---------------- Helpers ----------------
def third_wed(year, month):
    weds = [d for d in calendar.Calendar().itermonthdates(year, month)
        if d.weekday() == 2 and d.month == month]
    return weds[2]

def sofr_expiry(year, month):
    expiry = third_wed(year, month) - pd.Timedelta(days=5)
    holidays = USFederalHolidayCalendar().holidays(start=f'{year}-01-01', end=f'{year}-12-31')
    while expiry.weekday() != 4 or expiry in holidays:
        expiry -= pd.Timedelta(days=1)
    return expiry

def extract_option_code(ticker):
    import re
    m = re.match(r'(SFR\w+\d)[CP] ', ticker)
    return m.group(1) if m else None

def get_option_tickers(root_ticker):
    try:
        chain = blp.bds(f'{root_ticker} Comdty', 'OPT_CHAIN')
        if chain.empty:
            logging.warning(f"{root_ticker}: OPT_CHAIN is empty.")
            return []
        col = [c for c in chain.columns if 'security' in c.lower()]
        if not col:
            logging.warning(f"{root_ticker}: No 'security' column in OPT_CHAIN.")
            return []
        raw = chain[col[0]].dropna().astype(str)
        return [' '.join(s.split()) for s in raw.tolist()]
    except Exception as e:
        logging.error(f"Ticker fetch failed for {root_ticker}: {e}")
        return []

# ---------------- Filtering Logic ----------------

def filter_liquid_options(df):

    """Filter to liquid options based on bid/ask presence."""

    if 'bid' not in df.columns or 'ask' not in df.columns:
        return pd.DataFrame() # Drop this chain

    df = df[(df['bid'].notna()) & (df['ask'].notna())]
    return df

def trim_volume_edges(df):

    """Trim to strikes between first and last with non-NaN volume, per type."""

    def _trim(group):
        # The 'type' for the current group ('c' or 'p') is stored in its name
        group_type = group.name

        if 'rt_open_interest' not in group.columns or group['rt_open_interest'].isna().all():
            return pd.DataFrame(columns=group.columns)
        valid = group['rt_open_interest'].notna()
        first = valid.idxmax()
        last = valid[::-1].idxmax()

        # Get the trimmed DataFrame slice
        trimmed_df = group.loc[first:last].copy()

        # Re-add the 'type' column before returning
        trimmed_df['type'] = group_type

        return trimmed_df

    return df.groupby('type', group_keys=False).apply(_trim, include_groups=False).reset_index(drop=True)

# ---------------- Snapshot Main Function ----------------

def run_snapshot(output_dir=None, futures_list=None):
    futures_list = futures_list or ROOT_FUTURES
    now = datetime.now()
    today = now.strftime('%Y%m%d')
    timestamp = now.strftime('%H%M%S')
    out_dir = output_dir or f'snapshots/{today}/{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    for root in futures_list:
        tickers = get_option_tickers(root)
        if not tickers:
            continue

        logging.info(f"[{root}] total tickers: {len(tickers)}")
        try:
            future_px = blp.bdp([f"{root} Comdty"], ['PX_LAST']).iloc[0, 0]
        except Exception as e:
            logging.warning(f"{root}: Failed to get future price: {e}")
            future_px = None

        code_map = {}
        for t in tickers:
            code = extract_option_code(t)
            if code:
                code_map.setdefault(code, []).append(t)

        for code, group in code_map.items():
            calls = [t for t in group if 'C ' in t]
            puts = [t for t in group if 'P ' in t]

            df_calls = blp.bdp(calls, FIELDS).reset_index().rename(columns={'index': 'ticker'})
            df_calls['type'] = 'c'
            df_puts = blp.bdp(puts, FIELDS).reset_index().rename(columns={'index': 'ticker'})
            df_puts['type'] = 'p'

            df = pd.concat([df_calls, df_puts], ignore_index=True)

            # Apply modular filters
            df = filter_liquid_options(df)

            # --- ADD CHECK ---
            # If the chain was dropped for missing bid/ask, skip to the next.
            if df.empty:
                logging.warning(f"[SKIP] {code}: Chain dropped, missing bid/ask columns.")
                continue
            # --------------------

            df = trim_volume_edges(df)

            if df.empty:
                logging.info(f"[SKIP] {code}: No valid options after filtering.")
                continue

            # Get strike
            #df['strike'] = df['ticker'].str.extract(r'(\d+\.\d+|\d+)').astype(float)

            try:
                m_code = code[3]
                year_digit = code[4:]
                year = 2020 + int(year_digit) if len(year_digit) == 1 else 2000 + int(year_digit)
                month = MONTH_CODE_MAP.get(m_code.upper(), 1)
                expiry = sofr_expiry(year, month)

            except Exception as e:
                logging.warning(f"Could not determine expiry for {code}: {e}")
                expiry = pd.NaT

            df['expiry_code'] = code
            df['expiry_date'] = expiry
            df['snapshot_ts'] = f"{today} {timestamp}"
            df['future_px'] = future_px

            file_name = f"{code}_{expiry.strftime('%b').lower() if pd.notna(expiry) else 'na'}.parquet"
            df.to_parquet(f"{out_dir}/{file_name}", index=False)
            logging.info(f"[SAVE] {code} -> {file_name}")

    logging.info(f"[SUMMARY] Total futures scanned: {len(futures_list)}")
    logging.info(f"[SUMMARY] Output written to: {out_dir}")
    logging.info("[DONE] Snapshot job complete.")



if __name__ == '__main__':
    import sys

    # Check for a '--test' command-line argument
    if '--test' in sys.argv:
        logging.info("Running in --test mode with a single future.")
        # Run with just one future for a quick test
        test_futures = ['SFRU5']
        run_snapshot(futures_list=test_futures)
    else:
        # Run with the full list of futures defined in the script
        run_snapshot()