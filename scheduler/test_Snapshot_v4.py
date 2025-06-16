import pandas as pd
from analytics_engine.sabr.sabr_v4 import run_sabr_on_snapshot

# Modify as needed
snapshot_path = 'snapshots/20250609/smile_bands.parquet'
output_dir = 'snapshots/20250609/sabr_fits'
forward = 95.0  # You may want to extract this from somewhere else
tau_dict = {
    'SFRM5': 0.05,
    'SFRU5': 0.25,
    'SFRZ5': 0.5,
    'SFRH6': 0.75,
    # ... add more as needed
}

df = pd.read_parquet(snapshot_path)

for expiry in df['expiry'].unique():
    if expiry not in tau_dict:
        print(f"Skipping {expiry} — no tau info")
        continue

    run_sabr_on_snapshot(df, expiry, forward, tau_dict[expiry], output_dir)
