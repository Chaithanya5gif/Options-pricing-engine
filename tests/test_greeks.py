import unittest
import math
from black_scholes import (
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_vega,
    black_scholes_theta,
    black_scholes_rho,
    implied_vol
)

class TestOptionGreeks(unittest.TestCase):
    def setUp(self):
        # Standard parameters
        self.K = 100
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2

    def test_delta_bounds_and_atm(self):
        # 1. Delta: ATM Delta should be around 0.5 for Call, -0.5 for Put (allowing continuous r drift)
        c_delta_atm, p_delta_atm = black_scholes_delta(100, self.K, self.T, self.r, self.sigma)
        self.assertTrue(0.5 < c_delta_atm < 0.7)
        self.assertTrue(-0.5 < p_delta_atm < -0.3)
        self.assertTrue(math.isclose(c_delta_atm - p_delta_atm, 1.0))

        # 2. Deep ITM/OTM Delta
        c_delta_itm, p_delta_itm = black_scholes_delta(200, self.K, self.T, self.r, self.sigma)
        self.assertTrue(math.isclose(c_delta_itm, 1.0, abs_tol=1e-4))
        self.assertTrue(math.isclose(p_delta_itm, 0.0, abs_tol=1e-4))

        c_delta_otm, p_delta_otm = black_scholes_delta(20, self.K, self.T, self.r, self.sigma)
        self.assertTrue(math.isclose(c_delta_otm, 0.0, abs_tol=1e-4))
        self.assertTrue(math.isclose(p_delta_otm, -1.0, abs_tol=1e-4))

    def test_gamma_peaks_atm_and_positive(self):
        # Gamma must be positive
        gamma_atm = black_scholes_gamma(100, self.K, self.T, self.r, self.sigma)
        gamma_otm = black_scholes_gamma(70, self.K, self.T, self.r, self.sigma)
        gamma_itm = black_scholes_gamma(130, self.K, self.T, self.r, self.sigma)

        self.assertGreater(gamma_atm, 0)
        self.assertGreater(gamma_otm, 0)
        self.assertGreater(gamma_itm, 0)

        # Gamma peaks at ATM (S = K)
        self.assertGreater(gamma_atm, gamma_otm)
        self.assertGreater(gamma_atm, gamma_itm)

    def test_vega_properties(self):
        # Vega is positive for both calls and puts and they are identical
        vega_atm = black_scholes_vega(100, self.K, self.T, self.r, self.sigma)
        vega_otm = black_scholes_vega(70, self.K, self.T, self.r, self.sigma)
        vega_itm = black_scholes_vega(130, self.K, self.T, self.r, self.sigma)

        self.assertGreater(vega_atm, 0)
        self.assertGreater(vega_otm, 0)
        self.assertGreater(vega_itm, 0)

        # Vega peaks at ATM (S = K)
        self.assertGreater(vega_atm, vega_otm)
        self.assertGreater(vega_atm, vega_itm)

    def test_theta_negative_time_decay(self):
        # Theta is negative for long options (options lose value as time decays)
        c_theta, p_theta = black_scholes_theta(100, self.K, self.T, self.r, self.sigma)
        self.assertLess(c_theta, 0)
        self.assertLess(p_theta, 0)

    def test_rho_signs(self):
        # Rho is positive for Call, negative for Put
        c_rho, p_rho = black_scholes_rho(100, self.K, self.T, self.r, self.sigma)
        self.assertGreater(c_rho, 0)
        self.assertLess(p_rho, 0)

    def test_implied_vol_skeleton(self):
        # Verify implied vol skeleton signature exists and returns a mock value
        sig = implied_vol(10.0, 100, 100, 1.0, 0.05, "call")
        self.assertIsNotNone(sig)


if __name__ == "__main__":
    unittest.main()
