# config.py
"""
Central configuration file for the Bloomberg ETL pipeline.
"""

# --- Bloomberg Tickers & Fields ---

# The main underlying security to scan for its option chain.
# Example is for 3M SOFR Futures (SR3 Comdty).
# You will need to find the specific ticker for the future you want to track.
# E.g., "SR3M5 ComdTCY" for the June 2025 contract's options.
# Using a generic "SR3 Comdty" might be too broad. Let's assume you have a specific future contract.
# For this example, let's pretend "SR3Z5 Comdty" is our target (Dec 2025).
# A better approach is to get the active contract and then its option chain.
# For now, let's set a placeholder.
MAIN_UNDERLYING_TICKER = [

    'SFRU5 Comdty', 'SFRZ5 Comdty', 'SFRH6 Comdty', 'SFRM6 Comdty', 'SFRU6 Comdty', 'SFRZ6 Comdty',

    'SFRH7 Comdty', 'SFRM7 Comdty', 'SFRU7 Comdty', 'SFRZ7 Comdty', 'SFRH8 Comdty', 'SFRM8 Comdty',

    'SFRU28 Comdty', 'SFRZ28 Comdty', 'SFRH29 Comdty', 'SFRM29 Comdty', 'SFRU29 Comdty', 'SFRZ29 Comdty'
]
 # Example: Dec 2025 SOFR Future

# Fields for the initial EOD snapshot (Phase 1)
# We get the chain and then poll these fields for each option in the chain.
EOD_SNAPSHOT_FIELDS = [
    "OPT_STRIKE_PX",         # Strike Price
    "OPT_PUT_CALL",          # Put or Call (C/P)
    "OPT_EXPIRE_DT",         # Expiration Date
    "OPEN_INTEREST",         # Official EOD Open Interest
    "PX_SETTLE"              # Official EOD Settlement Price
]

# Fields for the on-demand snapshot (Phase 3)
# These are pulled *after* a volume trigger.
ON_DEMAND_SNAPSHOT_FIELDS = [
    "LAST_PRICE",            # Last trade price
    "BID",                   # Current bid
    "ASK",                   # Current ask
    "VOLUME",                # New total cumulative volume
    "BID_IMPLIED_VOL",       # Bid Implied Vol
    "ASK_IMPLIED_VOL",       # Ask Implied Vol
    "LAST_TRADE_TIME_RT"     # Timestamp of the last trade
]

# Field for real-time subscription (Phase 2)
# We subscribe ONLY to this field.
SUBSCRIPTION_FIELD = "VOLUME"


# --- Scheduling ---

# Time to run the EOD snapshot job.
# Should be after market close and settlement.
# Format: HH:MM in 24-hour time (e.g., "17:30" for 5:30 PM)
# This time is in the local timezone of the server running the script.
SCHEDULED_JOB_TIME_HHMM = "17:30"


# --- Database ---

# File path for the local SQLite database.
# Using SQLite is simple, requires no server setup.
DATABASE_FILE = "sofr_options_etl.db"