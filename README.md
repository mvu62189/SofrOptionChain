# SOFR Option Chain Snapshot Framework

A modular system for pulling, caching, and analyzing the full SOFR option chain across expiries.

## Features

- **Local Caching**: Stores snapshot data with timestamped metadata
- **Analytics Engine**: Pluggable modules for model calibration and surface fitting
- **Log & Feed Panel**: Monitors changes, cache events, and refresh diagnostics

## Feutures
- **Snapshot Scheduler**: Automates periodic Bloomberg pulls for option chain data
- **Rate Monitor**: Tracks Bloomberg API usage to avoid daily limit caps
