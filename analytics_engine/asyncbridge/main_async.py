# /async_pipeline/main_async.py
"""
Main entry point for the ASYNC Bloomberg pipeline.

This file is responsible for:
1. Starting the AsyncBloombergService.
2. Starting the BLPAPI poller thread.
3. Building the initial data cache.
4. Sending the initial subscription requests.
5. Starting the main asynchronous event processing loop.
"""

import asyncio
import blpapi
import sys
import os

# --- Add parent directory to path to import shared modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from manual_tickers import TICKER_LIST
import config # We need this for the config.SUBSCRIPTION_FIELD

# --- Import from our async pipeline folder ---
from async_blp_service import AsyncBloombergService
from pipeline_logic_async import (
    build_manual_cache,
    process_bridge_queue
)


async def main():
    """
    Run the main async application.
    """
    service = None
    try:
        # --- 1. System Initialization ---
        service = AsyncBloombergService()
        if not await service.start():
            print("Failed to start service.")
            return
            
        service.start_poller_thread()

        # --- 2. Build Initial Cache ---
        # (This uses the manual_tickers.py file)
        live_snapshot_cache = build_manual_cache()
        active_tickers = list(live_snapshot_cache.keys())
        
        if not active_tickers:
            print("CRITICAL: No active tickers found. Exiting.")
            return

        # --- 3. Send Subscriptions ---
        print(f"Subscribing to {config.SUBSCRIPTION_FIELD} for {len(active_tickers)} tickers...")
        subList = blpapi.SubscriptionList()
        
        for ticker in active_tickers:
            cid = blpapi.CorrelationId(ticker)
            subList.add(
                ticker,
                config.SUBSCRIPTION_FIELD, # Use field from config
                correlationId=cid
            )
        
        service.session.subscribe(subList)
        print("Subscription request sent.")

        # --- 4. Start Main Event Loop ---
        # This function will now run forever, processing events
        # from the bridge queue as they arrive.
        await process_bridge_queue(
            service=service,
            bridge_queue=service.bridge_queue,
            cache=live_snapshot_cache
        )
            
    except KeyboardInterrupt:
        print("Caught interrupt, shutting down...")
    except Exception as e:
        print(f"CRITICAL error in main: {e}")
    finally:
        # --- 5. Cleanup ---
        if service:
            service.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program terminated.")