# main.py
"""
The main entry point for the Bloomberg SOFR Options ETL pipeline.

This script:
1. Initializes the database and tables.
2. Starts the Bloomberg API service.
3. Sets up a daily scheduler to run the EOD snapshot (Phase 1).
4. Runs an *initial* EOD snapshot on startup to get today's data.
5. Initializes the live data cache and table from that snapshot (Phase 2).
6. Subscribes to real-time 'VOLUME' for all active tickers.
7. Enters the main event loop to process real-time trade events (Phase 3).
"""

import blpapi
import time
import sys
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Import our custom modules
import config
from database_manager import engine, create_db_and_tables
from bloomberg_service import BloombergService
from pipeline_logic import (
    run_initial_snapshot,
    initialize_live_data,
    process_subscription_event
)

def main():
    """
    Main application function.
    """
    print("--- Starting SOFR Options ETL Pipeline ---")

    # --- 1. System Initialization ---
    print("Initializing database...")
    create_db_and_tables()
    db_engine = engine

    print("Initializing Bloomberg Service...")
    blp_service = BloombergService()
    if not blp_service.start():
        print("CRITICAL: Failed to start Bloomberg session. Exiting.")
        return # Exit if we can't connect

    # --- 2. Scheduler Setup (Phase 1) ---
    # This scheduler will run the EOD snapshot *every day* at the
    # configured time to update the 'initial_snapshot' table for *tomorrow*.
    scheduler = BackgroundScheduler(daemon=True)
    try:
        hour, minute = config.SCHEDULED_JOB_TIME_HHMM.split(':')
        scheduler.add_job(
            run_initial_snapshot,
            CronTrigger(hour=hour, minute=minute),
            args=[blp_service, db_engine],
            name="Daily EOD Snapshot"
        )
        scheduler.start()
        print(f"Successfully scheduled daily EOD job for {hour}:{minute}.")
    except Exception as e:
        print(f"Warning: Could not start scheduler. {e}")
        print("The daily EOD snapshot will not run automatically.")

    # --- 3. Initial Data Load (Phase 1 & 2 on Startup) ---
    # We must run the snapshot *now* to get the data for *today's* session.
    # This uses yesterday's EOD data to build our starting point.
    print("Running initial EOD snapshot on startup...")
    active_tickers = run_initial_snapshot(blp_service, db_engine)
    if not active_tickers:
        print("CRITICAL: No active tickers found. Exiting.")
        blp_service.stop()
        return

    print("Initializing live data cache from initial snapshot...")
    live_snapshot_cache = initialize_live_data(db_engine)
    if not live_snapshot_cache:
        print("CRITICAL: Failed to initialize live data cache. Exiting.")
        blp_service.stop()
        return

    # --- 4. Real-time Subscription ---
    print(f"Subscribing to {config.SUBSCRIPTION_FIELD} for {len(active_tickers)} tickers...")
    blp_service.subscribe(active_tickers, [config.SUBSCRIPTION_FIELD])
    print("Subscription request sent. Ready for market data.")

    # --- 5. Main Event Loop (Phase 3) ---
    print("--- Entering Main Event Loop (Press Ctrl+C to stop) ---")
    try:
        while True:
            # next_event() will block for a short time (e.g., 100ms)
            event = blp_service.next_event(timeout_ms=100)
            
            # If event is None, it means the timeout was hit.
            # This is normal, just loop again.
            if event is None:
                continue

            # We are only interested in SUBSCRIPTION_DATA events
            if event.eventType() == blpapi.Event.SUBSCRIPTION_DATA:
                process_subscription_event(
                    event,
                    blp_service,
                    db_engine,
                    live_snapshot_cache
                )
            
            # Note: A production system would also handle other event types,
            # like SubscriptionFailure, Admin, SessionTerminated, etc.

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Shutting down gracefully...")
    except Exception as e:
        print(f"CRITICAL ERROR in main loop: {e}")
    finally:
        # --- 6. Cleanup ---
        print("Shutting down scheduler...")
        if scheduler.running:
            scheduler.shutdown()
        
        print("Stopping Bloomberg session...")
        blp_service.stop()
        
        print("--- Pipeline Stopped ---")
        sys.exit(0)


if __name__ == "__main__":
    main()