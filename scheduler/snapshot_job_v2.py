# Save snapshots as parquet to snapshots/yyyymmdd/hhmmss/SFRmy-mmm.parquet

import os
import sys
import calendar
import logging
from datetime import datetime
import pandas as pd
from xbbg import blp
from pandas.tseries.holiday import USFederalHolidayCalendar

# ---------------- Config ----------------
MONTH_CODE_MAP = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                  'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}


# ---------------- Config ----------------
fields = ['opt_strike_px', 'bid', 'ask', 'last_price', 'volume']
root_futures = [
    'SFRU5', 'SFRZ5',
    'SFRH6', 'SFRM6', 'SFRU6', 'SFRZ6',
    'SFRH7', 'SFRM7', 'SFRU7', 'SFRZ7',
    'SFRH8', 'SFRM8', 'SFRU28', 'SFRZ28',
    'SFRH29', 'SFRM29', 'SFRU29', 'SFRZ29'
]
if '--test' in sys.argv:
    root_futures = ['SFRU5']

# ---------------- Logging ----------------
os.makedirs('logs', exist_ok=True)
log_path = f'logs/snapshot_log_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logging.getLogger().addHandler(logging.StreamHandler())

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


# ---------------- Main Snapshot ----------------
def run_snapshot():
    today = datetime.now().strftime('%Y%m%d')
    timestamp = datetime.now().strftime('%H%M%S')
    out_dir = f'snapshots/{today}/{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    for root in root_futures:
        tickers = get_option_tickers(root)
        if not tickers:
            continue

        logging.info(f"[{root}] total tickers: {len(tickers)}")
        future_px = blp.bdp([f"{root} Comdty"], ['PX_LAST']).iloc[0, 0]

        code_map = {}
        for t in tickers:
            code = extract_option_code(t)
            if code:
                code_map.setdefault(code, []).append(t)

        for code, group in code_map.items():
            calls = [t for t in group if 'C ' in t]
            puts  = [t for t in group if 'P ' in t]

            df_calls = blp.bdp(calls, fields).reset_index().rename(columns={'index': 'ticker'})
            df_calls['type'] = 'c'
            df_puts = blp.bdp(puts, fields).reset_index().rename(columns={'index': 'ticker'})
            df_puts['type'] = 'p'

            df = pd.concat([df_calls, df_puts], ignore_index=True)

            # if either bid or ask is entirely missing, drop this chain
            missing = [c for c in ('bid','ask') if c not in df.columns]
            if missing:
                logging.warning(f"{code}: missing columns {missing}, dropping this chain.")
                continue

            # now filter to truly liquid strikes (both bid and ask present)
            df = df[(df['bid'].notna()) & (df['ask'].notna())]

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
            logging.info(f"[SAVE] {code} → {file_name}")

    logging.info(f"[SUMMARY] Total futures scanned: {len(root_futures)}")
    logging.info(f"[SUMMARY] Output written to: {out_dir}")

    logging.info("[DONE] Snapshot job complete.")

if __name__ == '__main__':
    run_snapshot()
