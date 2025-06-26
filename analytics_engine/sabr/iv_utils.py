# vol_utils.py

from bachelier import bachelier_iv
from black76 import black76_iv

IV_ENGINES = {
    'bachelier': bachelier_iv,
    'black76':   black76_iv,
}

def implied_vol(F, T, K, price, opt_type, engine='black76'):
    """
    Universal interface: converts puts→calls, enforces intrinsic floor,
    then calls the selected pricing‐model inversion engine.
    """
#    intrinsic = max(0.0, F - K)
#    # put→call parity
#    p_call = price + intrinsic if opt_type.upper() == 'P' else price
#    # floor extrinsic
#    p_call = max(p_call, intrinsic + 1e-8)
    # delegate to chosen engine
    solver = IV_ENGINES[engine]
    return solver(F, T, K, price, opt_type=opt_type)
