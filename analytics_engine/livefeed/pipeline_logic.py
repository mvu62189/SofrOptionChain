# pipeline_logic.py
"""
Contains the core ETL business logic for all three phases:
1. Scheduled EOD Snapshot
2. Live Data Initialization
3. Real-time Event Processing
"""

import blpapi
import config
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

# Database imports
from sqlmodel import Session, select
from sqlalchemy.engine import Engine
from database_manager import engine as db_engine # Import our main engine

# Service and model imports
from bloomberg_service import BloombergService, SUBSCRIPTION_DATA
from data_models import InitialSnapshotModel, LiveSnapshotModel, TradeOccursModel
from manual_tickers import TICKER_LIST

# --- Phase 1: Scheduled EOD Snapshot ---

def run_initial_snapshot(blp_service: BloombergService, 
                         db_engine: Engine) -> List[str]:
    """
    Runs the full EOD snapshot process.
    1. Loops through all MAIN_UNDERLYING_TICKER from config.
    2. Fetches the option chain for each underlying.
    3. Fetches EOD data for ALL options found.
    4. Filters for strikes with Open Interest > 0.
    5. Wipes and repopulates the 'initial_snapshot' table.
    
    Returns:
        A list of active tickers (those with OI > 0) to be subscribed to.
    """
    print("--- Starting Phase 1: Initial EOD Snapshot ---")
    
    # 1. Fetch option chains for all underlyings
    option_to_underlying_map: Dict[str, str] = {}
    all_option_tickers: List[str] = []

    try:
        # Loop over the new list from config
        for underlying in config.MAIN_UNDERLYING_TICKER:
            print(f"Fetching option chain for {underlying}...")
            chain = blp_service.get_option_chain(underlying)
            if not chain:
                print(f"No option chain found for {underlying}. Skipping.")
                continue
            
            # Add to our master lists
            for option_ticker in chain:
                if option_ticker not in option_to_underlying_map:
                    option_to_underlying_map[option_ticker] = underlying
                    all_option_tickers.append(option_ticker)
            
        if not all_option_tickers:
            print("No option tickers found for any underlyings. Aborting.")
            return []
            
    except Exception as e:
        print(f"Error fetching option chain: {e}")
        return []

    # 2. Fetch EOD data for ALL options at once
    try:
        print(f"Fetching EOD data for {len(all_option_tickers)} total strikes...")
        eod_data = blp_service.get_reference_data(
            tickers=all_option_tickers,
            fields=config.EOD_SNAPSHOT_FIELDS
        )
    except Exception as e:
        print(f"Error fetching EOD reference data: {e}")
        return []

    # 3. Filter for active strikes and build model objects
    initial_records: List[InitialSnapshotModel] = []
    active_tickers: List[str] = []

    for ticker, fields in eod_data.items():
        try:
            oi = fields.get("OPEN_INTEREST")
            
            # 4. Filter for Open Interest > 0
            if oi is not None and oi > 0:
                
                # Use our map to find the correct underlying
                underlying = option_to_underlying_map.get(ticker, "UNKNOWN")
                if underlying == "UNKNOWN":
                    print(f"Warning: Could not find parent for {ticker}. Skipping.")
                    continue

                record = InitialSnapshotModel(
                    ticker=ticker,
                    underlying=underlying,  # <-- This now correctly uses the mapped underlying
                    strike=fields.get("OPT_STRIKE_PX"),
                    cp_flag=fields.get("OPT_PUT_CALL"),
                    maturity=fields.get("OPT_EXPIRE_DT"),
                    open_interest=oi,
                    settle_price=fields.get("PX_SETTLE")
                )
                initial_records.append(record)
                active_tickers.append(ticker)
        except Exception as e:
            print(f"Error parsing EOD data for {ticker}: {e}. Skipping.")

    print(f"Found {len(active_tickers)} active strikes with OI > 0 across all underlyings.")
    
    # 5. Wipe and repopulate the 'initial_snapshot' table
    with Session(db_engine) as session:
        try:
            print("Wiping 'initial_snapshot' table...")
            statement = select(InitialSnapshotModel)
            results = session.exec(statement)
            for row in results:
                session.delete(row)
            
            print(f"Populating 'initial_snapshot' table with {len(initial_records)} records...")
            session.add_all(initial_records)
            
            session.commit()
            print("Initial snapshot saved successfully.")
        except Exception as e:
            print(f"Database error during initial snapshot: {e}")
            session.rollback()
            return [] # Return empty list on DB error

    print("--- Phase 1: Completed ---")
    return active_tickers

# --- Phase 2: Live Data Initialization ---

def initialize_live_data(db_engine: Engine) -> Dict[str, LiveSnapshotModel]:
    """
    Prepares the live environment for the trading day.
    1. Wipes the 'live_snapshot' table.
    2. Copies all data from 'initial_snapshot' to 'live_snapshot'.
    3. Builds and returns an in-memory cache of the live state.
    
    Returns:
        A dictionary cache: {ticker: LiveSnapshotModel_object}
    """
    print("--- Starting Phase 2: Live Data Initialization ---")
    live_snapshot_cache: Dict[str, LiveSnapshotModel] = {}
    
    with Session(db_engine) as session:
        try:
            # 1. Wipe the 'live_snapshot' table
            print("Wiping 'live_snapshot' table...")
            statement = select(LiveSnapshotModel)
            results = session.exec(statement)
            for row in results:
                session.delete(row)
            
            # 2. Get all records from 'initial_snapshot'
            initial_records = session.exec(select(InitialSnapshotModel)).all()
            
            if not initial_records:
                print("No records found in 'initial_snapshot'. Live data will be empty.")
                return {}

            print(f"Copying {len(initial_records)} records to 'live_snapshot' table...")
            new_live_records = []
            for initial_record in initial_records:
                # Create a LiveSnapshotModel from the InitialSnapshotModel
                live_record = LiveSnapshotModel.model_validate(initial_record)
                
                # 3. Build the in-memory cache
                live_snapshot_cache[live_record.ticker] = live_record
                new_live_records.append(live_record)

            session.add_all(new_live_records)
            session.commit()
            
            print(f"Created in-memory cache with {len(live_snapshot_cache)} items.")
            
        except Exception as e:
            print(f"Database error during live data initialization: {e}")
            session.rollback()
            return {} # Return empty cache on error

    print("--- Phase 2: Completed ---")
    return live_snapshot_cache


# --- Phase 3: Real-time Event Processing ---

def _parse_subscription_event(event: blpapi.Event) -> Optional[Tuple[str, int]]:
    """Helper to parse a subscription event and return (ticker, new_volume)."""
    for msg in event:
        # Check if it's subscription data
        if msg.messageType() != SUBSCRIPTION_DATA:
            continue
        
        # Check if it contains our VOLUME field
        if not msg.hasElement(config.SUBSCRIPTION_FIELD):
            continue
            
        try:
            # Get ticker from the CorrelationId we set during subscription
            ticker = msg.correlationIds()[0].value()
            new_volume = msg.getElementAsInteger(config.SUBSCRIPTION_FIELD)
            return ticker, new_volume
        except Exception as e:
            print(f"Error parsing subscription message: {e}")
            
    return None

def process_subscription_event(
    event: blpapi.Event,
    blp_service: BloombergService,
    db_engine: Engine,
    live_snapshot_cache: Dict[str, LiveSnapshotModel]
):
    """
    Processes a single real-time event from the subscription.
    This is the core "trigger-pull" logic.
    """
    # 1. Parse the event to get ticker and new volume
    parsed_data = _parse_subscription_event(event)
    if not parsed_data:
        return # Not a volume event, ignore

    ticker, new_volume = parsed_data

    # 2. Get previous state from in-memory cache
    try:
        cached_state = live_snapshot_cache[ticker]
        previous_volume = cached_state.volume
    except KeyError:
        print(f"Received event for untracked ticker: {ticker}. Ignoring.")
        return

    # 3. The Logic Gate: Check if volume has *actually* changed
    if new_volume == previous_volume:
        # This was just a quote update or other event, not a trade. Ignore.
        return
        
    # --- A Trade Has Been Detected! ---
    print(f"Trade detected for {ticker}: Vol {previous_volume} -> {new_volume}")

    # 4. Calculate trade size
    trade_size = new_volume - previous_volume # This handles new trades and bust trades

    # 5. Pull the on-demand snapshot for this single strike
    try:
        snapshot_result = blp_service.get_reference_data(
            tickers=[ticker],
            fields=config.ON_DEMAND_SNAPSHOT_FIELDS
        )
        snapshot_data = snapshot_result.get(ticker)
        if not snapshot_data:
            print(f"Failed to get on-demand snapshot for {ticker}. Aborting trade log.")
            return
    except Exception as e:
        print(f"Error on on-demand snapshot for {ticker}: {e}")
        return

    # 6. Create the TradeOccursModel
    now = datetime.now()
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

    # 7. Update the cached LiveSnapshotModel object
    cached_state.volume = new_volume
    cached_state.last_price = snapshot_data.get("LAST_PRICE")
    cached_state.bid = snapshot_data.get("BID")
    cached_state.ask = snapshot_data.get("ASK")
    cached_state.bid_iv = snapshot_data.get("BID_IMPLIED_VOL")
    cached_state.ask_iv = snapshot_data.get("ASK_IMPLIED_VOL")
    cached_state.last_update_timestamp = now

    # 8. Save changes to the database
    with Session(db_engine) as session:
        try:
            # Add the new trade to the 'trade_occurs' log
            session.add(trade_record)
            
            # Update the existing row in the 'live_snapshot' table
            session.merge(cached_state)
            
            session.commit()
            print(f"Successfully processed and saved trade for {ticker}.")
        except Exception as e:
            print(f"Database error saving trade for {ticker}: {e}")
            session.rollback()
            # CRITICAL: If DB save fails, we must revert the in-memory cache
            # to prevent data drift.
            cached_state.volume = previous_volume # Revert to old volume
            # (Note: Reverting other fields is complex;
            # a more robust system might reload from DB)


# --- NEW BYPASS FUNCTION ---

def build_manual_cache() -> Tuple[Dict[str, LiveSnapshotModel], List[str]]:
    """
    BYPASS FUNCTION: Creates a minimal in-memory cache from the
    hardcoded TICKER_LIST. This completely skips all refdata calls.
    
    Returns a tuple of: (live_snapshot_cache, active_tickers)
    """
    print("--- RUNNING IN MANUAL BYPASS MODE ---")
    print("--- Bypassing all database and refdata calls ---")
    
    live_snapshot_cache: Dict[str, LiveSnapshotModel] = {}
    active_tickers: List[str] = []

    for ticker in TICKER_LIST:
        if not ticker or ticker.isspace():
            continue
            
        # Create a DUMMY LiveSnapshotModel.
        # The only critical field for Phase 3 is 'volume = 0'.
        # The other fields are just to prevent errors.
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
        active_tickers.append(ticker)

    print(f"Built manual cache with {len(active_tickers)} tickers.")
    return live_snapshot_cache, active_tickers