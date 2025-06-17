# sabr_run.py

import os
import json
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast
from bachelier import bachelier_iv

def setup_logger():
    logger = logging.getLogger("sabr_run")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    return logger

def load_and_prepare(path, logger=None):
    df = pd.read_parquet(path)
    df = df[(df.bid>0)&(df.ask>0)]
    
    logger.info(f"Loaded {len(df)} rows from {path}")
    
    df['mid_price'] = (df.bid + df.ask)/2
    strikes = df.strike.values
    F = df.future_px.iloc[0]
    T = (pd.to_datetime(df.expiry_date.iloc[0]).date() -
         datetime.strptime(df.snapshot_ts.iloc[0], '%Y%m%d %H%M%S').date()).days/365
    vols = np.array([
        bachelier_iv(p, F, K, T) for p,K in zip(df.mid_price, strikes)
    ])
    mask = ~np.isnan(vols)
    return strikes[mask], vols[mask], F, T

def main():
    parser = argparse.ArgumentParser(
        description="Run SABR calibration on a snapshot parquet file"
    )
    parser.add_argument("parquet", help="Path to snapshot parquet file")
    parser.add_argument(
        "--params-dir",
        default="analytics_results/model_params",
        help="Base directory to store SABR parameter JSONs"
    )
    parser.add_argument(
        "--mode", choices=["auto","full","fast"], default="auto",
        help="Calibration mode: auto (first=full, then fast), or force full/fast"
    )
    args = parser.parse_args()

    logger = setup_logger()
    strikes, vols, F, T = load_and_prepare(args.parquet, logger)
 
    # Build model → expiry directory under analytics_results
    code = os.path.basename(args.parquet).split("_")[0]
    base_dir = os.path.join(args.params_dir, "sabr")
    code_dir = os.path.join(base_dir, code)
    os.makedirs(code_dir, exist_ok=True)

    existing = sorted([f for f in os.listdir(code_dir) if f.endswith(".json")])
    use_full = (args.mode == "full") or (args.mode == "auto" and not existing)

    if use_full:
        logger.info("Running full SABR calibration")
        params = calibrate_sabr_full(strikes, vols, F, T)
    else:
        if args.mode == "fast":
            logger.info("Running fast SABR calibration")
        else:
            logger.info("Existing params found; running fast SABR calibration")
        prev = json.load(open(os.path.join(code_dir, existing[-1])))
        params = calibrate_sabr_fast(strikes, vols, F, T, np.array(prev))

    # Use the snapshot’s own timestamp (YYYYMMDD/HHMMSS from the path)
    parts = os.path.normpath(args.parquet).split(os.sep)
    # expect .../snapshots/YYYYMMDD/HHMMSS/Foo.parquet
    date_part, time_part = parts[-3], parts[-2]
    ts = date_part + time_part

    out_file = os.path.join(code_dir, f"{ts}.json")
    with open(out_file, "w") as f:
        json.dump(params.tolist(), f)
    logger.info(f"Saved SABR parameters to {out_file}")


if __name__ == "__main__":
    main()



# python analytics_engine/sabr/sabr_run.py snapshots/20250616/135106/SFRN5_jul.parquet --params-dir analytics_results/model_params --mode auto/ or force full/fast