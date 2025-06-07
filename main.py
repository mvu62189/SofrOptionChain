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
