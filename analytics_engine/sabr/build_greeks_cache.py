import os
import pandas as pd
import numpy as np
from collections import defaultdict

# --- Import your existing calculation and processing modules ---
# Note: Ensure these paths are correct for your project structure
from mdl_load import discover_snapshot_files
from mdl_processing import process_snapshot_file
from greeks import calculate_greeks, calculate_greeks_bachelier
from sabr_v2 import sabr_vol_lognormal, sabr_vol_normal

# --- CONFIGURATION ---
RAW_SNAPSHOTS_DIR = 'snapshots'
GREEKS_CACHE_DIR = 'analytics_results/greeks_exposure'
CONTRACT_NOTIONAL = 1_000_000

def build_cache_from_raw_snapshots(model_engine='black76'):
    """
    Processes all raw snapshot files, calculates greeks and exposures,
    and saves the results to a query-optimized partitioned Parquet dataset.
    """
    print(f"--- Starting cache build for model: {model_engine} ---")
    
    file_dict = discover_snapshot_files(RAW_SNAPSHOTS_DIR)
    all_files = [f for files in file_dict.values() for f in files]
    if not all_files:
        print("No snapshot files found. Exiting.")
        return

    skipped_files_log = defaultdict(list)
    all_processed_data = []

    # 1. Load all data into a single DataFrame first for efficiency
    df_list = []
    for f in all_files:
        df_list.append(pd.read_parquet(f))
    full_df = pd.concat(df_list, ignore_index=True)

    # 2. Clean and correct the strike column
    strike_regex = r'[CP]\s*(\d+(?:\.\d+)?)'
    full_df['strike'] = pd.to_numeric(full_df['ticker'].str.extract(strike_regex, expand=False), errors='coerce')
    if 'opt_strike_px' in full_df.columns:
        opt_strike_numeric = pd.to_numeric(full_df['opt_strike_px'], errors='coerce')
        full_df['strike'] = full_df['strike'].fillna(opt_strike_numeric)
        
    # 3. Process each snapshot and expiry individually
    for ts, df_snapshot in full_df.groupby('snapshot_ts'):
        print(f"Processing snapshot: {ts}")
        for expiry, df_expiry in df_snapshot.groupby('expiry_date'):
            df_expiry = df_expiry.copy()
            
            # --- Calibration ---
            res, reason = process_snapshot_file(None, manual_params={}, df_input=df_expiry, model_engine=model_engine)
            if not res or not res.get('params_fast'):
                source_name = f"{ts}_{expiry}"
                skipped_files_log[reason].append(source_name)
                continue 
            
            sabr_params = res['params_fast']
            F = res['forward_price']
            T = (pd.to_datetime(expiry).date() - pd.to_datetime(ts.split(" ")[0]).date()).days / 365.0
            
            # --- Greeks Calculation ---
            if model_engine == 'black76':
                df_expiry['sabr_iv'] = sabr_vol_lognormal(F, df_expiry['strike'], T, **sabr_params)
                greeks_func = calculate_greeks
            else: # bachelier
                df_expiry['sabr_iv'] = sabr_vol_normal(F, df_expiry['strike'], T, sabr_params['alpha'], sabr_params['rho'], sabr_params['nu'])
                greeks_func = calculate_greeks_bachelier

            calls = df_expiry[df_expiry['type'].str.upper() == 'C'].copy()
            puts = df_expiry[df_expiry['type'].str.upper() == 'P'].copy()
            if not calls.empty:
                greeks_c = greeks_func(F, calls['strike'], T, calls['sabr_iv'], 'C')
                for greek in greeks_c: calls[greek] = greeks_c[greek]
            if not puts.empty:
                greeks_p = greeks_func(F, puts['strike'], T, puts['sabr_iv'], 'P')
                for greek in greeks_p: puts[greek] = greeks_p[greek]
            
            df_processed_expiry = pd.concat([calls, puts])
            df_processed_expiry['forward_price'] = F
            all_processed_data.append(df_processed_expiry)

    if not all_processed_data:
        print("No data was successfully processed.")
        return

    # 4. Combine all results and calculate exposure metrics
    final_df = pd.concat(all_processed_data, ignore_index=True)
    
    # Fill missing greeks columns for safety
    greeks_cols = ['delta', 'gamma', 'vega', 'theta', 'vanna', 'charm']
    for col in greeks_cols:
        if col not in final_df.columns:
            final_df[col] = 0.0

    final_df['rt_open_interest'] = pd.to_numeric(final_df['rt_open_interest'], errors='coerce').fillna(0)
    
    # --- Exposure Calculations ---
    final_df['delta_exp'] = final_df['delta'] * final_df['rt_open_interest'] * CONTRACT_NOTIONAL
    final_df['gamma_exp'] = final_df['gamma'] * final_df['rt_open_interest'] * (CONTRACT_NOTIONAL * 0.01) * 0.01
    final_df['vega_exp']  = final_df['vega']  * final_df['rt_open_interest'] * CONTRACT_NOTIONAL
    final_df['vanna_exp'] = final_df['vanna'] * final_df['rt_open_interest'] * CONTRACT_NOTIONAL * 0.01
    final_df['charm_exp'] = final_df['charm'] * final_df['rt_open_interest'] * CONTRACT_NOTIONAL
    final_df['theta_exp'] = final_df['theta'] * final_df['rt_open_interest'] * CONTRACT_NOTIONAL
    
    # --- Add Partitioning Columns ---
    final_df['snapshot_date'] = pd.to_datetime(final_df['snapshot_ts'].str.split(" ").str[0], format='%Y%m%d').dt.date
    if 'underlying_ticker' not in final_df.columns:
         final_df['underlying_ticker'] = final_df['ticker'].str.extract(r'^([A-Z0-9]+)')[0]


    # 5. Save the final results to the partitioned cache
    print(f"Saving final dataset to partitioned cache at: {GREEKS_CACHE_DIR}")
    final_df.to_parquet(
        GREEKS_CACHE_DIR,
        partition_cols=['snapshot_date', 'underlying_ticker'],
        engine='pyarrow'
    )
    
    print("\n--- Cache build complete! ---")
    if skipped_files_log:
        print("Some expiries could not be processed:")
        for reason, sources in skipped_files_log.items():
            print(f"  Reason: {reason} ({len(sources)} instances)")

if __name__ == "__main__":
    # You can choose which model engine's results to cache
    # build_cache_from_raw_snapshots(model_engine='black76')
    # Or run it for bachelier as well if needed
    build_cache_from_raw_snapshots(model_engine='bachelier')