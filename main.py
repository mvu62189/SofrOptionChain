# main.py

from scheduler.snapshot_job import run_snapshot_job
import schedule
import time

# Schedule every 10 minutes (change as needed)
schedule.every(10).minutes.do(run_snapshot_job)

print("[INFO] Starting scheduler loop...")
while True:
    schedule.run_pending()
    time.sleep(1)