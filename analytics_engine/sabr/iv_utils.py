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
    solver = IV_ENGINES[engine]
    return solver(F, T, K, price, opt_type=opt_type)
