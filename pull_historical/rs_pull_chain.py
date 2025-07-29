from xbbg import blp
import pandas as pd

# Example: get SOFRU5 option chain as of June 1, 2025
chain = blp.bds(
    'SFRZ5 Comdty', 
    'OPT_CHAIN', 
    ovrds=[('CHAIN_DATE', '20250728')]  # override is key for history
)

# Use the security column from Bloomberg
ticker_col = [c for c in chain.columns if 'security' in c.lower()][0]
tickers = chain[ticker_col].dropna().tolist()

print(tickers[:5])

