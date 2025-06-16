import numpy as np
import pytest
from sabr_v2 import sabr_vol_normal, calibrate_sabr_full, calibrate_sabr_fast

# pick a “true” parameter set
TRUE_PARAMS = dict(alpha=0.03, beta=0.5, rho=-0.2, nu=0.4)
F, T = 100.0, 0.5

@pytest.fixture
def synthetic_smile():
    strikes = np.linspace(80, 120, 9)
    vols = np.array([
        sabr_vol_normal(F, K, T,
                        TRUE_PARAMS['alpha'],
                        TRUE_PARAMS['beta'],
                        TRUE_PARAMS['rho'],
                        TRUE_PARAMS['nu'])
        for K in strikes
    ])
    return strikes, vols

def test_sabr_vol_consistency(synthetic_smile):
    strikes, vols = synthetic_smile
    # at least ensure vol is finite and >0
    assert np.all(vols > 0) and np.all(np.isfinite(vols))

def test_full_calibration_recovers_params(synthetic_smile):
    strikes, vols = synthetic_smile
    est = calibrate_sabr_full(strikes, vols, F, T)
    # allow some tolerance
    assert pytest.approx(TRUE_PARAMS['alpha'], rel=0.1) == est[0]
    assert pytest.approx(TRUE_PARAMS['beta'], rel=0.1) == est[1]
    assert pytest.approx(TRUE_PARAMS['rho'],  rel=0.2) == est[2]
    assert pytest.approx(TRUE_PARAMS['nu'],   rel=0.1) == est[3]

def test_fast_calibration_warm_start(synthetic_smile):
    strikes, vols = synthetic_smile
    # start from a slightly perturbed true params
    init = np.array([0.8*TRUE_PARAMS['alpha'],
                     TRUE_PARAMS['beta'],
                     TRUE_PARAMS['rho']*1.1,
                     TRUE_PARAMS['nu']*0.9])
    est = calibrate_sabr_fast(strikes, vols, F, T, init)
    # should converge back near TRUE_PARAMS
    assert pytest.approx(TRUE_PARAMS['alpha'], rel=0.1) == est[0]
    assert pytest.approx(TRUE_PARAMS['rho'],   rel=0.2) == est[2]
