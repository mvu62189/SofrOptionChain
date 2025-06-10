### THIS TEST PLOT AND SAVE BID/ASK IVM FOR BOTH CALL AND PUT 
#   FOR ALL TICKERS IN 1 PARQUET FILE. AND STORE THEM IN THE SAME DIR


import os
import pandas as pd
import matplotlib.pyplot as plt

# Reload required file after kernel reset
sample_path = 'snapshots\\20250609\smile_bands.parquet'  # adjust as needed
save_dir = 'snapshots\\20250609'  # adjust as needed
df = pd.read_parquet(sample_path)

# ----------- Plot per expiry -----------
for expiry in df['expiry'].unique():
    subset = df[df['expiry'] == expiry]

    plt.figure(figsize=(10, 6))
    plt.plot(subset['strike'], subset['call_iv_bid'], label='Call IVM Bid', linestyle='--', marker='.')
    plt.plot(subset['strike'], subset['call_iv_ask'], label='Call IVM Ask', linestyle='--', marker='.')
    plt.plot(subset['strike'], subset['put_iv_bid'], label='Put IVM Bid', linestyle='-', marker='x')
    plt.plot(subset['strike'], subset['put_iv_ask'], label='Put IVM Ask', linestyle='-', marker='x')

    plt.title(f"IVM Smile Bands - {expiry}")
    plt.xlabel('Strike')
    plt.ylabel('IVM (Implied Vol)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"plot_{expiry}.png")
    plt.savefig(save_path)
    plt.close()