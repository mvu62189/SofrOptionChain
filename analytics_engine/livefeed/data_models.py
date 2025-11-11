# data_models.py
"""
SQLModel models (Pydantic + SQLAlchemy) to define the data structures
and database tables for our ETL pipeline.
"""

from sqlmodel import Field, SQLModel
from datetime import datetime, date
from typing import Optional

class BaseStrikeModel(SQLModel):
    """
    Common fields that identify a specific option strike.
    This is a base class and not a table itself.
    """
    ticker: str = Field(..., primary_key=True, description="The unique Bloomberg option ticker")
    underlying: str = Field(..., description="The underlying future ticker (e.g., SR3Z5 Comdty)")
    strike: float = Field(..., description="Strike price")
    cp_flag: str = Field(..., description="Contract type: 'C' for Call, 'P' for Put")
    maturity: date = Field(..., description="Option expiration date")

class InitialSnapshotModel(BaseStrikeModel, table=True):
    """
    Defines the data captured in the scheduled EOD snapshot.
    This maps to the 'initial_snapshot' table.
    """
    open_interest: int = Field(..., description="Previous day's settlement Open Interest")
    settle_price: float = Field(..., description="Previous day's settlement price")
    
class LiveSnapshotModel(BaseStrikeModel, table=True):
    """
    Defines the "live" state of a strike.
    This maps to the 'live_snapshot' table.
    """
    # EOD fields
    open_interest: int
    settle_price: float
    
    # Real-time fields (initialized to None/0)
    volume: int = 0
    last_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_iv: Optional[float] = None
    ask_iv: Optional[float] = None
    last_update_timestamp: Optional[datetime] = None

class TradeOccursModel(SQLModel, table=True):
    """
    Defines the data we store when a trade is detected.
    This maps to the 'trade_occurs' table.
    """
    # Auto-incrementing primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Event data
    timestamp: datetime = Field(..., description="Timestamp when the event was processed")
    ticker: str = Field(..., foreign_key="livesnapshotmodel.ticker", description="The option ticker")
    trade_size: int = Field(..., description="Calculated size of the trade (new_vol - prev_vol)")
    
    # Data from the on-demand snapshot
    trade_price: Optional[float] = Field(..., description="LAST_PRICE from the snapshot")
    market_bid: Optional[float] = Field(..., description="BID from the snapshot")
    market_ask: Optional[float] = Field(..., description="ASK from the snapshot")
    bid_iv: Optional[float] = Field(..., description="BID_IMPLIED_VOL from snapshot")
    ask_iv: Optional[float] = Field(..., description="ASK_IMPLIED_VOL from snapshot")