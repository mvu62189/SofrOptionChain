import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Imports needed by the process_snapshot_file function
from iv_utils import implied_vol
from mdl_calibration import fit_sabr_de, fit_sabr
from mdl_rnd_utils import market_rnd, model_rnd

@st.cache_data(show_spinner="Calibrating...", persist=True)
def process_snapshot_file(parquet_path, manual_params, df_input=None, model_engine='black76'):
    # Logic to accept df and parquet as input
    source_name = ""
    if parquet_path:
        source_name = os.path.basename(parquet_path)
    elif df_input is not None and not df_input.empty:
        # If we get a dataframe, create a name from its expiry code or date
        if 'expiry_code' in df_input.columns:
            source_name = f"data for {df_input['expiry_code'].iloc[0]}"
        elif 'expiry_date' in df_input.columns:
            source_name = f"data for {pd.to_datetime(df_input['expiry_date'].iloc[0]).date()}"
        else:
            source_name = "an input DataFrame"
            
    if df_input is not None:
        df_raw = df_input
    elif parquet_path:
        df_raw = pd.read_parquet(parquet_path)
    else:
        st.error("process_snapshot_file requires either a parquet_path or a df_input.")
        return None
    
    df = df_raw.copy()

    ####--- Extract strike from ticker ---#####
    # This logic prioritizes parsing the ticker, which is the most reliable source.

    if 'ticker' not in df.columns:
        st.error(f"Critical 'ticker' column missing in {source_name}. Cannot process.")
        return None

    # 1. Define a highly specific regex to capture the strike price.
    # It looks for a 'C' or 'P' followed by the number.
    # Example: "SFRZ7C 94.625" -> "94.625"
    strike_regex = r'[CP]\s*(\d+(?:\.\d+)?)'

    # 2. Extract strikes from the reliable 'ticker' column.
    df['strike'] = df['ticker'].str.extract(strike_regex, expand=False)
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')

    # 3. As a fallback, fill any remaining missing strikes with 'opt_strike_px'.
    if 'opt_strike_px' in df.columns:
        # Create a temporary numeric version of opt_strike_px
        opt_strike_numeric = pd.to_numeric(df['opt_strike_px'], errors='coerce')
        df['strike'] = df['strike'].fillna(opt_strike_numeric)

    # 4. Ensure the final strike column is clean and numeric.
    df.dropna(subset=['strike'], inplace=True)
    if df.empty:
        #st.warning(f"Could not determine any valid strikes for {source_name}.")
        return None, "Could not determine any valid strikes."
    # --- END OF DEFINITIVE STRIKE FIX ---


    # df['strike'] = df['ticker'].str.extract(r'\b(\d+\.\d+)\b')[0].astype(float)
    
    # Filter for strikes with active bid/ask
    liquid = df[(df['bid']>0)&(df['ask']>0)]
    if liquid.empty: return None, "No options with valid bid/ask prices."
    lo, hi = liquid['strike'].min(), liquid['strike'].max()
    df_trim = df[(df['strike']>=lo)&(df['strike']<=hi)].reset_index(drop=True)

    # --- START OF FIX: Robustly ensure 'type' column exists ---
    if 'type' not in df_trim.columns:
        if 'ticker' in df_trim.columns:
            #st.warning(f"File {source_name} is missing 'type' column. Recreating from ticker.")
            df_trim['type'] = df_trim['ticker'].str.extract(r'([CP])\b', expand=False)
        else:
            #st.error(f"Cannot determine option type for {source_name}. Missing 'type' and 'ticker' columns.")
            return None, "No options with valid bid/ask prices."
    
    df_trim.dropna(subset=['type'], inplace=True)

    # --- START OF FIX: Add a guard clause here ---
    # If cleaning the 'type' column removed all rows, the DataFrame is invalid.
    if df_trim.empty:
        #st.warning(f"No rows with a valid option type found for {source_name}.")
        return None, "No rows with a valid option type found."
    # --- END OF FIX ---

    df_trim['type'] = df_trim['type'].str.upper()
    # --- END OF FIX ---

    # Get mid price
    df_trim['mid_price'] = np.where((df_trim['bid']>0)&(df_trim['ask']>0), 0.5*(df_trim['bid']+df_trim['ask']), np.nan)
    
    
    F = float(df_trim['future_px'].iloc[0])
    df_otm = df_trim[((df_trim['type']=='C')&(df_trim['strike']>F))|((df_trim['type']=='P')&(df_trim['strike']<F))].reset_index(drop=True)
    df_otm = df_otm.sort_values(by='strike').reset_index(drop=True)
    
    if df_otm.empty: return None
    snap_dt = datetime.strptime(df_otm['snapshot_ts'].iloc[0], '%Y%m%d %H%M%S')
    expiry = pd.to_datetime(df_otm['expiry_date'].iloc[0]).date()
    T = (expiry - snap_dt.date()).days/365.0
    
    # Get Black-76 vol
    #df_otm['iv'] = df_otm.apply(lambda r: implied_vol(F=float(r['future_px']), T=T, K=r['strike'], price=r['mid_price'], opt_type=r['type'], engine='black76') if not np.isnan(r['mid_price']) else np.nan, axis=1)
    
    # --- Get Black-76 / Bachelier vol based on toggle ---
    df_otm['iv'] = df_otm.apply(
        lambda r: implied_vol(
            F=float(r['future_px']), T=T, K=r['strike'], 
            price=r['mid_price'], opt_type=r['type'], 
            engine=model_engine  # Pass the selected engine
        ) if not np.isnan(r['mid_price']) else np.nan, 
        axis=1
    )

    #df_otm = df_otm[df_otm['iv']<200]
    
    #liquid2 = df_otm[df_otm['volume']>0]
    #if liquid2.empty: return None
    #lo2, hi2 = liquid2['strike'].min(), liquid2['strike'].max()
    #df_otm = df_otm[(df_otm['strike']>=lo2-0.52)&(df_otm['strike']<=hi2+0.52)]
    #df_otm['spread'] = df_otm['ask'] - df_otm['bid']
    #df_otm = df_otm[df_otm['spread']<=0.012]

    # --- New OINT trim  ---

    if 'rt_open_interest' in df_otm.columns:
        liquid2 = df_otm[df_otm['rt_open_interest'] > 0]
        
        # If no options with volume exist, we cannot determine the liquid strike range.
        # It's safest to skip processing for this file, as per original logic.
        if liquid2.empty:
            #st.warning(f"No options with open interest > 0 found in {source_name}. Skipping file.")
            return None, "No options with open interest > 0 found."
        
        # Trim the dataframe to the range of strikes that have trading volume.
        lo2, hi2 = liquid2['strike'].min(), liquid2['strike'].max()
        df_otm = df_otm[(df_otm['strike'] >= lo2 - 0.52) & (df_otm['strike'] <= hi2 + 0.52)]
    else:
        # If the rt_open_interest column doesn't exist, we can't perform this filter.
        # We'll issue a warning and proceed without this specific trimming step.
        #st.warning(f"'rt_open_interest' column not found in {source_name}. Skipping file.")
        return None
    # --- END OF FIX ---

    df_otm['spread'] = df_otm['ask'] - df_otm['bid']
    #df_otm = df_otm[df_otm['spread']<=0.012]    

    strikes = df_otm['strike'].values
    market_iv = df_otm['iv'].values
    mask = ~np.isnan(market_iv)
    fit_order = np.argsort(strikes[mask])
    strikes_fit = strikes[mask][fit_order]
    vols_fit = market_iv[mask][fit_order]
    
    # --- START OF FIX --- DROPPING EMPTY EXPIRIES --------------
    # Add a check here. If no valid options remain after filtering, skip this file.
    if len(strikes_fit) == 0:
        #st.warning(f"No valid data points found to calibrate for {source_name}. Skipping.")
        return None, "Not enough valid data points for calibration (<4)."
    # --- END OF FIX ---

    # Automatic calibration
    # params_fast, iv_model_fit, debug_data = fit_sabr(strikes_fit, F, T, vols_fit, method='fast')
    params_fast, iv_model_fit, debug_data = fit_sabr_de(strikes_fit, F, T, vols_fit, model_engine=model_engine)
    
    # This check is also good practice, in case calibration fails
    if len(iv_model_fit) == 0:
        #st.warning(f"SABR calibration failed for {source_name}. Skipping.")
        return None, "SABR calibration failed to converge."


    recalibrate = manual_params.get('recalibrate', False) # Safety: get recalibrate flag
    # ---  MANUAL CALIBRATION ---
    params_man, iv_manual = (None, None) # Initialize iv_manual as None for plotting
    if recalibrate and st.session_state.get('manual_file') == parquet_path:
        # Correctly unpack the 3-item tuple returned by fit_sabr
        manual_results = fit_sabr(strikes_fit, F, T, vols_fit, method='fast', manual_params=manual_params)

        if manual_results and len(manual_results) == 3:
            params_man, iv_manual_fit, debug_data = manual_results
            
            # Interpolate the manual IV from the fit grid back to the main strike grid
            if iv_manual_fit is not None and len(iv_manual_fit) > 0:
                 iv_manual = np.interp(strikes, strikes_fit, iv_manual_fit)
    
    # Interpolate the automatic model's IV back to the main grid for consistent plotting
    model_iv_on_market_strikes = np.interp(strikes, strikes_fit, iv_model_fit)
    # --- END OF CORRECTED SECTION ---

    mid_prices = df_otm['mid_price'].values
    rnd_mkt = market_rnd(strikes, mid_prices, F, T, model_engine)
    rnd_sabr = model_rnd(strikes, F, T, params_fast, model_engine=model_engine)
    rnd_man  = model_rnd(strikes, F, T, params_man, model_engine=model_engine) if params_man else None

    area_market = float(np.trapezoid(rnd_mkt, strikes))
    area_model  = float(np.trapezoid(rnd_sabr, strikes))

    return {'strikes': strikes, 'market_iv': market_iv,
            'model_iv': model_iv_on_market_strikes, 
            'iv_manual': iv_manual, 
            'rnd_market': rnd_mkt, 'rnd_sabr': rnd_sabr,
            'rnd_manual': rnd_man, 'params_fast': params_fast,
            'params_manual': params_man, 'mid_prices': mid_prices,
            'area_model': area_model, 'area_market': area_market,
            'forward_price': F,
            'debug_data': debug_data
            }, None