# calibration.py
import numpy as np
from typing import Tuple, Dict
from sabr_v2 import calibrate_sabr_full, calibrate_sabr_fast, sabr_vol_normal
import json, os


def fit_sabr(strikes: np.ndarray, F: float, T: float,
             vols: np.ndarray, method: str = 'fast',
             manual_params: Dict[str, float] = None) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Calibrate SABR to `vols` at strikes, return params and model IV curve.
    If `manual_params` is provided, overwrite the calibrated params.
    """

    # → choose initial beta: manual override or historical global
    if manual_params is not None:
        init_beta = 0
    else:
        init_beta = load_global_beta()

    # build a 4-tuple for the SABR routines
    if manual_params is not None:
        init_seq = (
            float(manual_params['alpha']),
            init_beta,
            float(manual_params['rho']),
            float(manual_params['nu'])
        )
    else:
        # defaults from last known calibration
        init_seq = (0.66745, init_beta, 0.79241, 2.46749)

    # choose calibration engine
    if method == 'fast':
        # pass a tuple (alpha0,beta0,rho0,nu0)
        alpha, beta, rho, nu = calibrate_sabr_fast(strikes, vols, F, T, init_seq)
    else:
        alpha, beta, rho, nu = calibrate_sabr_full(strikes, vols, F, T)
    # override with manual inputs if given
    if manual_params:
        alpha = manual_params.get('alpha', alpha)
        beta  = manual_params.get('beta', beta)
        rho   = manual_params.get('rho', rho)
        nu    = manual_params.get('nu', nu)
    params: Dict[str,float] = {'alpha': alpha, 'beta': beta, 'rho': rho, 'nu': nu}
    iv_model = sabr_vol_normal(F, strikes, T, alpha, rho, nu)
    return params, iv_model
