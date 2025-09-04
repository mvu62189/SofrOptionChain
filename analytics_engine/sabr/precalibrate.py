"""
precalibrate.py

### FOR PAGE 2_VOL_RND_SURFACE ###

This script discovers all snapshot files, runs the SABR model calibration for
both 'black76' and 'bachelier' models, and saves the results to a cache directory.

Run this script from your terminal before launching the Streamlit app to ensure
a fast, responsive user experience.

Usage:
    python precalibrate.py
"""
import os
import time
from pathlib import Path
import joblib
from tqdm import tqdm

# Ensure the script can find the necessary modules from your project
from mdl_processing import process_snapshot_file

# --- CONFIGURATION ---
SNAPSHOTS_DIR = "snapshots"
CACHE_DIR = "precalibrated_cache"
MODEL_ENGINES = ['black76', 'bachelier']

def run_precalibration():
    """
    Finds all snapshot files, processes them, and caches the results.
    """
    print("--- Starting SABR Model Pre-calibration ---")
    
    # Create the main cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Discover all .parquet files in the snapshots directory
    snapshot_files = list(Path(SNAPSHOTS_DIR).rglob("*.parquet"))
    
    if not snapshot_files:
        print(f"Error: No .parquet files found in the '{SNAPSHOTS_DIR}' directory.")
        return

    print(f"Found {len(snapshot_files)} snapshot files to process.")
    
    # Use tqdm for a progress bar
    for file_path in tqdm(snapshot_files, desc="Calibrating Snapshots"):
        
        # Determine the relative path to mirror the directory structure
        relative_path = file_path.relative_to(SNAPSHOTS_DIR)
        cache_subdir = Path(CACHE_DIR) / relative_path.parent
        os.makedirs(cache_subdir, exist_ok=True)
        
        for engine in MODEL_ENGINES:
            # Define the output path for the cached result
            file_stem = file_path.stem
            cache_file_path = cache_subdir / f"{file_stem}_{engine}.joblib"
            
            # --- CORE CALIBRATION ---
            # Process the file using the existing logic
            # The 'res' dictionary contains all calibrated parameters, market data, etc.
            res, reason = process_snapshot_file(str(file_path), manual_params={}, model_engine=engine)

            if reason is None and res:
                # Save the entire result dictionary using joblib for efficiency
                joblib.dump(res, cache_file_path)
            # else:
                # Optional: log failures if needed
                # print(f"Skipping {file_path} for {engine} model. Reason: {reason}")
                
    print("\n--- Pre-calibration Complete ---")
    print(f"Cached results are saved in the '{CACHE_DIR}' directory.")


if __name__ == "__main__":
    start_time = time.time()
    run_precalibration()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
