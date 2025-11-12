# main.py
"""
The main entry point for the Bloomberg SOFR Options ETL pipeline.

*** MANUAL BYPASS MODE ***
This version is modified to SKIP all refdata (Phase 1) and
database (Phase 2) initialization. It uses a hardcoded
list of tickers to test the subscription (Phase 3) logic.
"""

import blpapi
import time
import sys
# from apscheduler.schedulers.background import BackgroundScheduler
# from apscheduler.triggers.cron import CronTrigger

# Import our custom modules
import config
# from database_manager import engine, create_db_and_tables # No DB
from bloomberg_service import BloombergService
from pipeline_logic import (
    # run_initial_snapshot, # Bypassed
    # initialize_live_data, # Bypassed
    process_subscription_event,
    build_manual_cache # <-- Our new function
)

def main():
    """
    Main application function (MANUAL BYPASS).
    """
    print("--- Starting SOFR Options ETL Pipeline (MANUAL MODE) ---")

    # --- 1. System Initialization ---
    # print("Initializing database...")
    # create_db_and_tables() # Bypassed
    # db_engine = engine # Bypassed
    
    # We still need a db_engine object for process_subscription_event
    # It just won't be used if we don't save to DB.
    # Let's import it but comment out the save logic.
    from database_manager import engine as db_engine

    print("Initializing Bloomberg Service...")
    blp_service = BloombergService()
    if not blp_service.start():
        print("CRITICAL: Failed to start Bloomberg session. Exiting.")
        return

    # --- 2. Scheduler Setup (Phase 1) ---
    # print("Scheduler bypassed.")
    
    # --- 3. Initial Data Load (Phase 1 & 2 on Startup) ---
    print("Bypassing refdata and database load...")
    
    (live_snapshot_cache, active_tickers) = build_manual_cache()

    if not active_tickers:
        print("CRITICAL: No active tickers found in manual_tickers.py. Exiting.")
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
            event = blp_service.next_event(timeout_ms=100)
            
            if event is None:
                continue

            if event.eventType() == blpapi.Event.SUBSCRIPTION_DATA:
                # NOTE: This will try to write to the database.
                # You can comment out the DB save logic in 
                # process_subscription_event if you want to avoid it.
                process_subscription_event(
                    event,
                    blp_service,
                    db_engine,
                    live_snapshot_cache
                )
            elif event.eventType() == blpapi.Event.SUBSCRIPTION_STATUS:
                # This is a message *about* our subscription (e.g., a failure)
                for msg in event:
                    # We defined SUBSCRIPTION_FAILURE in bloomberg_service.py
                    if msg.messageType() == blpapi.Name("SubscriptionFailure"):
                        
                        # Get the ticker we tried to subscribe to
                        ticker = msg.correlationIds()[0].value()
                        
                        # Get the error reason
                        reason = msg.getElement("reason")
                        error_code = reason.getElementAsString("errorCode")
                        description = reason.getElementAsString("description")
                        
                        print("---" * 10)
                        print(f"!!! SUBSCRIPTION FAILED !!!")
                        print(f"  Ticker:  {ticker}")
                        print(f"  Error:   {error_code}")
                        print(f"  Reason:  {description}")
                        print("---" * 10)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Shutting down gracefully...")
    except Exception as e:
        print(f"CRITICAL ERROR in main loop: {e}")
    finally:
        # --- 6. Cleanup ---
        print("Stopping Bloomberg session...")
        blp_service.stop()
        print("--- Pipeline Stopped ---")
        sys.exit(0)


if __name__ == "__main__":
    main()