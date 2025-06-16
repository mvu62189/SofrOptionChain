# main.py

from scheduler.snapshot_job import run_snapshot_job
import schedule
import time
import sys

def main():
    if 'once' in sys.argv:
        print("[INFO] Manual trigger: running snapshot job once")
        run_snapshot_job()
        return

    # Scheduled mode
    print("[INFO] Starting scheduler loop (every 10 minutes)...")
    schedule.every(10).minutes.do(run_snapshot_job)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    main()



### ---------------------------------------
### JSON-lookup logic SABR
### ---------------------------------------

from sabr_v4 import calibrate_full_parquet, fast_recalibrate_parquet
import glob, os, json

def choose_and_run(code, parquet_path):
    param_dir = f"sabr_params/{code}"
    os.makedirs(param_dir, exist_ok=True)
    existing = sorted(glob.glob(f"{param_dir}/*.json"))
    if not existing:
        params = calibrate_full_parquet(parquet_path)
    else:
        latest = existing[-1]
        prev = json.load(open(latest))
        params = fast_recalibrate_parquet(parquet_path, prev)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    out = f"{param_dir}/{ts}.json"
    json.dump(params, open(out,"w"))
    return params
