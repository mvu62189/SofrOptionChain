# sabr_run.py

import os
import json
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from analytics_engine.sabr.sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast
from analytics_engine.sabr.bachelier import bachelier_iv, bachelier_vega

def setup_logger():
    logger = logging.getLogger("sabr_run")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    return logger


def load_and_prepare(path, logger=None, min_iv=1e-4, min_vega=1e-6):
    # 1) Read & basic bid/ask filter
    df = pd.read_parquet(path)
    df = df[(df.bid > 0) & (df.ask > 0)]
    if logger:
        logger.info(f"Raw rows after bid/ask filter: {len(df)}")

    # 2) Mid‐price and intrinsic
    df['mid_price'] = (df.bid + df.ask) / 2
    intrinsic = np.maximum(0.0, df.future_px - df.strike)
    df = df[df.mid_price > intrinsic]
    if logger:
        logger.info(f"Rows with positive time‐value: {len(df)}")

    # 3) Compute T
    expiry_dt = pd.to_datetime(df.expiry_date.iloc[0]).date()
#    raw = df.snapshot_ts.iloc[0]
#    try:
#        snap_dt = datetime.strptime(raw, '%Y%m%d %H%M%S').date()
#    except ValueError:
#        snap_dt = datetime.strptime(raw, '%Y%m%d%H%M%S').date()

    raw   = df.snapshot_ts.iloc[0]
    clean = raw.replace(" ", "")
    snap_dt = datetime.strptime(clean, '%Y%m%d%H%M%S').date()
    days    = (expiry_dt - snap_dt).days
    # ensure at least one day so tests don’t drop everything
    T = max(days, 1) / 365.0


#    T = max((expiry_dt - snap_dt).days, 1) / 365

    # 4) Invert to IV, floor it, compute vega, and filter
    strikes, ivs = [], []
    for _, row in df.iterrows():
        K = row.strike
        p = row.mid_price
        iv = bachelier_iv(row.future_px, T, K, p)
        iv = max(iv, min_iv)
        vega = bachelier_vega(row.future_px, K, T, iv)
        if vega >= min_vega:
            strikes.append(K)
            ivs.append(iv)

    if logger:
        logger.info(f"After min_iv & min_vega filter: {len(strikes)} strikes")

    return np.array(strikes), np.array(ivs), df.future_px.iloc[0], T


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
    code = os.path.basename(args.parquet).split("_")[0]     # extract the option‐chain code
    base_dir = os.path.join(args.params_dir, "sabr")        # locate “sabr” subfolder under the chosen params‐root
    code_dir = os.path.join(base_dir, code)                 # per expiry folders
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
    names = ["alpha", "beta", "rho", "nu"]
    param_dict = dict(zip(names, params.tolist()))
    with open(out_file, "w") as f:
        json.dump(param_dict, f, indent=2)
    logger.info(f"Saved SABR parameters to {out_file}")


if __name__ == "__main__":
    main()



# python analytics_engine/sabr/sabr_run.py snapshots/20250616/135106/SFRN5_jul.parquet 
# --params-dir analytics_results/model_params 
# --mode auto/ or force full/fast

