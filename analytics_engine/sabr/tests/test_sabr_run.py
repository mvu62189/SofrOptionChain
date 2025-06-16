import os
import json
import sys
import numpy as np
import pandas as pd
import pytest
from datetime import datetime
import sabr_run

@pytest.fixture
def fake_snapshot(tmp_path):
    # build a minimal DataFrame matching the schema
    df = pd.DataFrame({
        'ticker': ['SFRX5C 100.0', 'SFRX5P 100.0'],
        'strike': [100.0, 100.0],
        'bid':    [2.0, 1.8],
        'ask':    [2.2, 2.0],
        'future_px': [100.0, 100.0],
        # expiry_date as date
        'expiry_date': [pd.Timestamp('2025-11-21'), pd.Timestamp('2025-11-21')],
        # snapshot_ts in required format
        'snapshot_ts': ['20251121 120000', '20251121 120000']
    })
    p = tmp_path / "test_snapshot.parquet"
    df.to_parquet(p, index=False)
    return str(p)

def test_sabr_run_creates_params(tmp_path, fake_snapshot, monkeypatch, caplog):
    # create a params dir
    params_dir = tmp_path / "params"
    # simulate CLI invocation
    monkeypatch.setattr(sys, 'argv', [
        'sabr_run.py', fake_snapshot,
        '--params-dir', str(params_dir),
        '--mode', 'auto'
    ])
    caplog.set_level('INFO')
    sabr_run.main()
    # after run, we should have one subfolder SFRX5 and at least one json inside
    code = 'SFRX5'
    code_dir = params_dir / code
    files = list(code_dir.glob("*.json"))
    assert len(files) == 1
    # load and check it’s a list of 4 numbers
    params = json.load(open(files[0]))
    assert isinstance(params, list) and len(params) == 4
    # check log message
    assert "Saved SABR parameters to" in caplog.text
