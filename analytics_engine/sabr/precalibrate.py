"""
precalibrate.py

This script discovers all snapshot files, runs the SABR model calibration for
both 'black76' and 'bachelier' models, and saves the results to a cache directory.

It now performs two caching operations:
1.  Saves individual chain calibration results for the 'Surfaces' page.
2.  Calculates and saves aggregated snapshot metrics for the 'Snapshot Overview' page.

Usage:
    python precalibrate.py
"""
import os
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from scipy.interpolate import interp1d

# Ensure the script can find the necessary modules from your project
from mdl_load import discover_snapshot_files
from mdl_processing import process_snapshot_file

# --- CONFIGURATION ---
SNAPSHOTS_DIR = "snapshots"
CACHE_DIR = "precalibrated_cache"
OVERVIEW_CACHE_DIR = os.path.join(CACHE_DIR, "overview")
MODEL_ENGINES = ['black76', 'bachelier']

# --- HELPER FUNCTION for Delta Calculation ---
def black76_delta(F, K, T, r, sigma, option_type='call'):
    """Calculates Black-76 delta."""
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return np.exp(-r * T) * norm.cdf(d1)
    else: # put
        return np.exp(-r * T) * (norm.cdf(d1) - 1)

def run_precalibration():
    """
    Finds all snapshot files, processes them, and caches the results for
    both the surfaces page and the new overview page.
    """
    print("--- Starting SABR & Metrics Pre-calibration ---")
    
    # Create the cache directories
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OVERVIEW_CACHE_DIR, exist_ok=True)

    file_dict = discover_snapshot_files(SNAPSHOTS_DIR)
    if not file_dict:
        print(f"Error: No snapshot folders found in the '{SNAPSHOTS_DIR}' directory.")
        return

    print(f"Found {len(file_dict)} snapshot folders to process.")
    
    # Loop 1: Calibrate individual files (for Surfaces page)
    all_files = [file for files in file_dict.values() for file in files]
    for file_path_str in tqdm(all_files, desc="Calibrating Individual Chains"):
        file_path = Path(file_path_str)
        relative_path = file_path.relative_to(SNAPSHOTS_DIR)
        cache_subdir = Path(CACHE_DIR) / relative_path.parent
        os.makedirs(cache_subdir, exist_ok=True)
        
        for engine in MODEL_ENGINES:
            cache_file_path = cache_subdir / f"{file_path.stem}_{engine}.joblib"
            res, reason = process_snapshot_file(str(file_path), manual_params={}, model_engine=engine)
            if reason is None and res:
                joblib.dump(res, cache_file_path)

    # --- NEW: Column name mapping for standardization ---
    COLUMN_MAPPING = {
        'Option Type': 'option_type',
        'Open Interest': 'open_interest',
        'Volume': 'volume',
        # Add other potential variations here if needed
    }

    # Loop 2: Aggregate metrics for each snapshot folder (for Overview page)
    for folder, files in tqdm(file_dict.items(), desc="Aggregating Snapshot Metrics"):
        snapshot_metrics = {
            'total_volume': 0, 'total_oi': 0,
            'call_volume': 0, 'put_volume': 0,
            'call_oi': 0, 'put_oi': 0,
            'term_structure': [], # List of {'T', 'atm_iv'}
            'risk_reversals': [] # List of {'T', 'rr_25d'}
        }
        
        for file_path in files:
            df = pd.read_parquet(file_path)
            
            # --- FIX: Standardize column names ---
            df.rename(columns=COLUMN_MAPPING, inplace=True)
            
            # Now, the check for 'option_type' will work correctly
            if 'option_type' not in df.columns:
                print(f"\n[Warning] Skipping file: 'option_type' column not found in {file_path} after renaming. Cannot aggregate.")
                continue
            
            res, _ = process_snapshot_file(file_path, manual_params={}, model_engine='black76', df_override=df)

            if res is None: continue

            # Aggregate totals
            if 'volume' in df.columns:
                snapshot_metrics['total_volume'] += df['volume'].sum()
            if 'open_interest' in df.columns:
                snapshot_metrics['total_oi'] += df['open_interest'].sum()

            # Aggregate call/put specific metrics
            calls_df = df[df['option_type'] == 'C']
            puts_df = df[df['option_type'] == 'P']

            if 'volume' in df.columns:
                snapshot_metrics['call_volume'] += calls_df['volume'].sum()
                snapshot_metrics['put_volume'] += puts_df['volume'].sum()
            
            if 'open_interest' in df.columns:
                snapshot_metrics['call_oi'] += calls_df['open_interest'].sum()
                snapshot_metrics['put_oi'] += puts_df['open_interest'].sum()

            # Term Structure & Skew Metrics
            F = res['forward_price']
            T = res['T']
            atm_strike_idx = np.abs(res['strikes'] - F).argmin()
            atm_iv = res['market_iv'][atm_strike_idx]
            
            if not np.isnan(atm_iv):
                snapshot_metrics['term_structure'].append({'T': T, 'atm_iv': atm_iv})

            # Calculate Risk Reversal
            try:
                deltas = [black76_delta(F, k, T, 0, iv, 'call') for k, iv in zip(res['strikes'], res['market_iv'])]
                valid_deltas = [(d, iv) for d, iv in zip(deltas, res['market_iv']) if not np.isnan(d) and not np.isnan(iv)]
                if len(valid_deltas) > 2:
                    delta_vals, iv_vals = zip(*sorted(valid_deltas))
                    interp_func = interp1d(delta_vals, iv_vals, bounds_error=False, fill_value=np.nan)
                    
                    iv_25d_call = interp_func(0.25)
                    iv_25d_put = interp_func(1 - 0.25)
                    
                    if not np.isnan(iv_25d_call) and not np.isnan(iv_25d_put):
                        rr_25d = iv_25d_call - iv_25d_put
                        snapshot_metrics['risk_reversals'].append({'T': T, 'rr_25d': rr_25d})
            except Exception:
                continue # Skip if interpolation fails

        # Final Calculations
        call_vol = snapshot_metrics['call_volume']
        put_vol = snapshot_metrics['put_volume']
        snapshot_metrics['volume_pc_ratio'] = put_vol / call_vol if call_vol > 0 else 0
        
        call_oi = snapshot_metrics['call_oi']
        put_oi = snapshot_metrics['put_oi']
        snapshot_metrics['oi_pc_ratio'] = put_oi / call_oi if call_oi > 0 else 0


        # Save the aggregated metrics for this snapshot
        sanitized_folder_name = folder.replace('\\', '_').replace('/', '_')
        overview_cache_file = Path(OVERVIEW_CACHE_DIR) / f"{sanitized_folder_name}.joblib"
        joblib.dump(snapshot_metrics, overview_cache_file)

    print("\n--- Pre-calibration and Aggregation Complete ---")

if __name__ == "__main__":
    start_time = time.time()
    run_precalibration()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

