import numpy as np
from analytics_engine.sabr.sabr_rnd import compute_rnd

def test_rnd_shape_and_positive():
    strikes = np.linspace(90, 110, 21)
    F, T = 100.0, 0.5
    alpha, beta, rho, nu = 0.03, 0.5, -0.3, 0.4
    pdf = compute_rnd(strikes, F, T, alpha, beta, rho, nu)
    assert pdf.shape == strikes.shape
    assert np.all(pdf >= 0)  # risk-neutral density must be non-negative
