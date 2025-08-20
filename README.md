# SOFR SABR Options Analysis Pipeline

This project provides a comprehensive Streamlit-based toolkit for the analysis of SOFR (Secured Overnight Financing Rate) options. It leverages the SABR volatility model to calibrate to market data, derive risk-neutral densities (RNDs), and estimate dealer exposure to various Greeks.

The application can connect directly to the Bloomberg API for live data snapshots or run in a local mode using historical parquet files.

## Features

-   **Live Data Snapshotting:** Pull real-time options chain data from the Bloomberg Terminal.
-   **SABR Model Calibration:** Calibrate SABR parameters ($\alpha, \rho, \nu$) to market volatility smiles using robust optimization techniques. Supports both **Black-76 (Lognormal)** and **Bachelier (Normal)** pricing models.
-   **Global Beta ($\beta$) Calibration:** Perform a historical calibration across multiple snapshots to determine a stable, long-term beta parameter.
-   **Volatility Smile Visualization:** Plot market-implied vs. SABR model-implied volatility smiles. Includes interactive manual calibration tools.
-   **Risk-Neutral Density (RND) Analysis:** Compute and plot market-implied and SABR-based RNDs.
-   **3D Volatility & RND Surfaces:** Visualize the entire volatility or RND surface across strikes and maturities. Compare the market-interpolated surface with a smooth SABR-parameter-interpolated surface.
-   **Greeks Exposure Dashboard:** An advanced dashboard to analyze dealer positioning based on open interest. It calculates and visualizes key metrics like:
    -   Gamma Exposure (GEX) and the "Gamma Flip" point.
    -   Vanna Exposure (VEX).
    -   Charm, Delta, Vega, and Theta exposure.
    -   Time-series analysis of total net exposures.

## Prerequisites

1.  **Python:** Version 3.8 or higher.
2.  **Bloomberg Terminal:** Required for live data snapshots.
    -   A valid Bloomberg subscription with API access.
    -   Bloomberg Desktop API installed.
    -   The `blpapi3_64.dll` file must be accessible. The easiest way is to place it in the root directory of this project.
3.  **C++ Redistributable:** The Bloomberg API often requires the Microsoft Visual C++ Redistributable for Visual Studio.

## Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Set up Bloomberg API**
    -   Download the C++ `blpapi` library from the Bloomberg API downloads page.
    -   Locate the `blpapi3_64.dll` file within the downloaded package.
    -   **Copy `blpapi3_64.dll` into the root directory of this project.**

3.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Requirements**
    -   A Python wheel for the `blpapi` library is required. Place the `.whl` file in the project directory.
    -   Create a `requirements.txt` file with the following content:
        ```
        streamlit
        pandas
        numpy
        scipy
        matplotlib
        plotly
        xbbg
        pyarrow
        ```
    -   Install the `blpapi` wheel first, followed by the other packages:
        ```bash
        # Replace the filename with your specific wheel version
        pip install blpapi-3.19.1-cp39-cp39-win_amd64.whl
        pip install -r requirements.txt
        ```

## Running the Application

Once the installation is complete, you can launch the Streamlit application:

```bash
streamlit run 1_RND.py
```


The application will open in your default web browser.

Project Structure
.
├── 1_RND.py                # Main app entry point (Vol Smile & RND)
├── pages/
│   ├── 2_surfaces.py       # IV & RND Surfaces page
│   └── 3_Greeks_Exposure.py# Greeks Exposure Dashboard page
├── snapshots/              # Default directory for saved market data
├── analytics_results/      # Default directory for saved model parameters
├── mdl_*.py                # Core application modules (loading, processing, etc.)
├── sabr_v2.py              # SABR model implementation
├── greeks.py               # Greeks calculation functions
├── blpapi-X.X.X.whl        # Bloomberg API Python wheel (user must provide)
├── blpapi3_64.dll          # Bloomberg API DLL (user must provide)
└── readme.md               # This file

## Local Mode (Without Bloomberg)
If you do not have the Bloomberg Terminal or API set up, the application will start in "local mode." It will disable the "Pull New Snapshot" functionality but will be fully capable of analyzing any pre-existing .parquet files located in the snapshots/ directory.

