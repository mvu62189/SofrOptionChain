import numpy as np
import pytest
from bachelier import bachelier_iv, bachelier_price, bachelier_vega

def test_bachelier_price_and_vega_consistency():
    F, K, T, sigma = 100.0, 105.0, 30/365, 0.02
    price = bachelier_price(F, K, T, sigma)
    # Vega should be positive and finite
    vega = bachelier_vega(F, K, T, sigma)
    assert vega > 0 and np.isfinite(vega)
    # If we perturb sigma by a small delta, price changes roughly by vega*delta
    delta = 1e-4
    price_up = bachelier_price(F, K, T, sigma + delta)
    assert pytest.approx(price + vega*delta, rel=1e-2) == price_up

def test_bachelier_iv_round_trip():
    F, K, T, sigma = 100.0, 95.0, 0.5, 0.025
    price = bachelier_price(F, K, T, sigma)
    iv = bachelier_iv(F, T, K, price)
    assert pytest.approx(sigma, rel=1e-3) == iv
