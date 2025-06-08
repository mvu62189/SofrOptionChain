# SOFR Option Chain Snapshot Framework

A modular system for pulling, caching, and analyzing the full SOFR option chain across expiries.

## Features

- **Snapshot Scheduler**: Automates periodic Bloomberg pulls for option chain data
- **Local Caching**: Stores snapshot data with timestamped metadata
- **Analytics Engine**: Pluggable modules for model calibration and surface fitting
- **Log & Feed Panel**: Monitors changes, cache events, and refresh diagnostics
- **Rate Monitor**: Tracks Bloomberg API usage to avoid daily limit caps

## Folder Structure
scheduler/ # Scheduled pulls and manual refresh logic
snapshots/ # Raw option chain data (timestamped)
cache_engine/ # Data loaders, filters, and merging logic
analytics_engine/ # Calibrators, fitters, and volatility engines
logs/ # System logs, feed logs, error tracking
utils/ # Common functions (Bloomberg helpers, file ops, etc.)


## Setup

- Requires: `blpapi`, `xbbg`, `streamlit`, `pandas`, `schedule`, `apscheduler`
- Configure Bloomberg session before running



## ⚙️ Setup Instructions (Windows)

### 1. Clone the Repo

Open PowerShell and run:

```powershell
git clone https://github.com/mvu62189/SofrOptionChain.git
cd SofrOptionChain



