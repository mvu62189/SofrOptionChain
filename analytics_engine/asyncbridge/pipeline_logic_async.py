# /async_pipeline/pipeline_logic_async.py
"""
Contains the core asynchronous business logic for processing events
from the async_blp_service bridge.
"""

import asyncio
import blpapi
from datetime import datetime, date
from typing import Dict, List, Any

# Import shared modules from the parent directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from manual_tickers import TICKER_LIST
from data_models import LiveSnapshotModel, TradeOccursModel
from database_manager import engine as db_engine

# Import the service class for type hinting
from .async_blp_service import (
    AsyncBloombergService,
    SUBSCRIPTION_DATA,
    SUBSCRIPTION_FAILURE,
    SUBSCRIPTION_STATUS,
)

# Import SQLModel for the database session
from sqlmodel import Session


def build_manual_cache() -> Dict[str, LiveSnapshotModel]:
    """
    Creates a minimal in-memory cache from the hardcoded TICKER_LIST.
    This completely skips all refdata calls.
    
    Returns the in-memory cache: {ticker: LiveSnapshotModel_object}
    """
    print("--- RUNNING IN MANUAL BYPASS MODE ---")
    
    live_snapshot_cache: Dict[str, LiveSnapshotModel] = {}

    for ticker in TICKER_LIST:
        if not ticker or ticker.isspace():
            continue
            
        # Create a DUMMY LiveSnapshotModel.
        # The only critical field for Phase 3 is 'volume = 0'.
        dummy_model = LiveSnapshotModel(
            ticker=ticker,
            underlying="MANUAL",
            strike=99.0, # Dummy data
            cp_flag="C",  # Dummy data
            maturity=date.today(), # Dummy data
            open_interest=1, # Dummy data
            settle_price=0.0, # Dummy data
            volume=0  # <-- This is the only important part
        )
        live_snapshot_cache[ticker] = dummy_model

    print(f"Built manual cache with {len(live_snapshot_cache)} tickers.")
    return live_snapshot_cache


async def process_bridge_queue(service: AsyncBloombergService, 
                               bridge_queue: asyncio.Queue,
                               cache: Dict[str, LiveSnapshotModel]):
    """
    The main asynchronous event processing loop.
    It waits for events from the bridge_queue and routes them.
    """
    print("Starting main event consumer loop...")
    while True:
        # 1. Wait for an event from the poller thread
        event = await bridge_queue.get()

        try:
            # 2. Route the event to the correct processor
            await _route_event(event, service, cache)
            
        except Exception as e:
            print(f"Error processing event: {e}")
        finally:
            # Tell the queue this task is done
            bridge_queue.task_done()


async def _route_event(event: blpapi.Event, 
                       service: AsyncBloombergService, 
                       cache: Dict[str, LiveSnapshotModel]):
    """
    Routes a single event to the appropriate handler function.
    """
    event_type = event.eventType()

    if event_type == SUBSCRIPTION_DATA:
        for msg in event:
            await _process_subscription_data(msg, service, cache)
    
    elif event_type == SUBSCRIPTION_STATUS:
        for msg in event:
            _process_subscription_status(msg)
    
    # Can add other handlers here (e.g., SESSION_STATUS)
    # else:
    #    print(f"Received unhandled event type: {event_type}")


def _save_trade_to_db_sync(trade_record: TradeOccursModel, 
                           live_record: LiveSnapshotModel):
    """
    A SYNCHRONOUS function that performs the database write.
    This is designed to be run in a separate thread via `asyncio.to_thread`.
    """
    try:
        with Session(db_engine) as session:
            # Add the new trade to the 'trade_occurs' log
            session.add(trade_record)
            
            # Update the existing row in the 'live_snapshot' table
            # .merge() finds the row by its primary key (ticker) and updates it
            session.merge(live_record)
            
            session.commit()
            print(f"  DB: Successfully saved trade for {live_record.ticker}.")
    except Exception as e:
        print(f"  DB Error: Failed to save trade for {live_record.ticker}: {e}")


async def _process_subscription_data(msg: blpapi.Message, 
                                     service: AsyncBloombergService, 
                                     cache: Dict[str, LiveSnapshotModel]):
    """
    Handles a single subscription data message (the "trigger-pull" logic).
    """
    # We only care about VOLUME messages
    if not msg.hasElement("VOLUME"):
        return

    try:
        ticker = msg.correlationIds()[0].value()
        new_volume = msg.getElementAsInteger("VOLUME")
        
        # Get the previous state from our cache
        cached_state = cache[ticker]
        previous_volume = cached_state.volume

        # 3. Check for a new trade
        if new_volume != previous_volume:
            print("---" * 10)
            print(f"TRADE DETECTED: {ticker}")
            print(f"  Volume change: {previous_volume} -> {new_volume}")
            
            # 4. Pull on-demand snapshot (non-blocking)
            snapshot = await service.get_snapshot_data(
                tickers=[ticker],
                fields=["LAST_PRICE", "BID", "ASK", "BID_IMPLIED_VOL", "ASK_IMPLIED_VOL"]
            )
            
            if not snapshot or ticker not in snapshot:
                print(f"  Error: Failed to get snapshot for {ticker}.")
                return
            
            snapshot_data = snapshot[ticker]
            print(f"  Snapshot data: {snapshot_data}")

            now = datetime.now()
            trade_size = new_volume - previous_volume

            # 5. Create DB models
            trade_record = TradeOccursModel(
                timestamp=now,
                ticker=ticker,
                trade_size=trade_size,
                trade_price=snapshot_data.get("LAST_PRICE"),
                market_bid=snapshot_data.get("BID"),
                market_ask=snapshot_data.get("ASK"),
                bid_iv=snapshot_data.get("BID_IMPLIED_VOL"),
                ask_iv=snapshot_data.get("ASK_IMPLIED_VOL")
            )

            # 6. Update the cached LiveSnapshotModel object IN MEMORY
            cached_state.volume = new_volume
            cached_state.last_price = snapshot_data.get("LAST_PRICE")
            cached_state.bid = snapshot_data.get("BID")
            cached_state.ask = snapshot_data.get("ASK")
            cached_state.bid_iv = snapshot_data.get("BID_IMPLIED_VOL")
            cached_state.ask_iv = snapshot_data.get("ASK_IMPLIED_VOL")
            cached_state.last_update_timestamp = now

            # 7. Save to DB (asynchronously)
            # We run the synchronous, blocking DB code in a worker thread.
            await asyncio.to_thread(
                _save_trade_to_db_sync,
                trade_record,
                cached_state
            )
            print("---" * 10)

    except KeyError:
        print(f"Error: Received tick for untracked ticker: {msg.correlationIds()[0].value()}")
    except Exception as e:
        print(f"Error processing subscription data: {e}")


def _process_subscription_status(msg: blpapi.Message):
    """
    Handles a subscription status message (e.g., failures).
    """
    if msg.messageType() == SUBSCRIPTION_FAILURE:
        try:
            ticker = msg.correlationIds()[0].value()
            reason = msg.getElement("reason")
            error_code = reason.getElementAsString("errorCode")
            description = reason.getElementAsString("description")
            
            print("---" * 10)
            print(f"!!! SUBSCRIPTION FAILED !!!")
            print(f"  Ticker:  {ticker}")
            print(f"  Error:   {error_code}")
            print(f"  Reason:  {description}")
            print("---" * 10)
        except Exception as e:
            print(f"Error parsing subscription failure: {e}")