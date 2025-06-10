### THIS TEST PLOT AN INTERACTIVE BID/ASK IVM FOR BOTH CALL AND PUT

### ---------------------------------------------------
#   SHOWING IVM BID = IVM ASK. NEED TO EXAMINE THIS
### ---------------------------------------------------

import pandas as pd
import plotly.graph_objects as go

# Load the parquet file
df = pd.read_parquet('snapshots/20250609/smile_bands.parquet')

# Focus on a single expiry
expiry = 'SFRM5'
subset = df[df['expiry'] == expiry].copy()
subset = subset[subset['strike'].notna()]

fig = go.Figure()

# Add all four lines
fig.add_trace(go.Scatter(x=subset['strike'], y=subset['call_iv_bid'], mode='lines',
                         name='Call IVM Bid', line=dict(color='blue', dash='solid')))
fig.add_trace(go.Scatter(x=subset['strike'], y=subset['call_iv_ask'], mode='lines',
                         name='Call IVM Ask', line=dict(color='blue', dash='dot')))
fig.add_trace(go.Scatter(x=subset['strike'], y=subset['put_iv_bid'], mode='lines',
                         name='Put IVM Bid', line=dict(color='red', dash='solid')))
fig.add_trace(go.Scatter(x=subset['strike'], y=subset['put_iv_ask'], mode='lines',
                         name='Put IVM Ask', line=dict(color='red', dash='dot')))

fig.update_layout(
    title=f"Interactive IVM Smile for {expiry}",
    xaxis_title="Strike",
    yaxis_title="Implied Volatility",
    hovermode="x unified",
    template="plotly_white"
)

fig.show()
