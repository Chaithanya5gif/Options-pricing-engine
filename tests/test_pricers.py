import math
import pytest
from src.pricers.bs import black_scholes
from src.pricers.monte_carlo import mc_price
from src.pricers.binomial import price_option
from src.pricers.heston import heston_mc_price

def test_mc_convergence():
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    bs_call, bs_put = black_scholes(S0, K, T, r, sigma)
    
    mc_res = mc_price(S0, K, T, r, sigma, n_paths=100_000, option_type="call", seed=42)
    assert abs(mc_res["price"] - bs_call) < 0.05, "MC call should converge to BS"
    
def test_crr_convergence():
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    bs_call, _ = black_scholes(S0, K, T, r, sigma)
    
    crr_res = price_option(S0, K, T, r, sigma, N=500, option_type="call", exercise="european")
    assert abs(crr_res["price"] - bs_call) < 0.01, "CRR call should converge to BS"

def test_heston_sanity_check():
    # σᵥ→0, ρ→0 → Heston ≈ BS
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    theta = 0.04
    sigma_v = 0.001
    rho = 0.0
    bs_vol = math.sqrt(theta)
    
    bs_call, _ = black_scholes(S0, K, T, r, bs_vol)
    heston_res = heston_mc_price(
        S0, K, T, r, v0=theta, kappa=2.0, theta=theta, 
        sigma_v=sigma_v, rho=rho, n_paths=50_000, n_steps=100, option_type="call", seed=42
    )
    
    assert abs(heston_res["price"] - bs_call) < 0.05, "Heston should approximate BS when vol is constant and ρ=0"

def test_put_call_parity_bs():
    S0, K, T, r, sigma, q = 100.0, 110.0, 1.5, 0.05, 0.2, 0.02
    call, put = black_scholes(S0, K, T, r, sigma, q=q)
    
    lhs = call - put
    rhs = S0 * math.exp(-q * T) - K * math.exp(-r * T)
    assert math.isclose(lhs, rhs, rel_tol=1e-5), "Put-call parity failed for Black-Scholes"

def test_put_call_parity_mc():
    S0, K, T, r, sigma, q = 100.0, 90.0, 0.5, 0.02, 0.25, 0.01
    call_res = mc_price(S0, K, T, r, sigma, q=q, n_paths=100_000, option_type="call", seed=10)
    put_res = mc_price(S0, K, T, r, sigma, q=q, n_paths=100_000, option_type="put", seed=10)
    
    lhs = call_res["price"] - put_res["price"]
    rhs = S0 * math.exp(-q * T) - K * math.exp(-r * T)
    # MC has noise, so tolerance is wider
    assert abs(lhs - rhs) < 0.05, "Put-call parity failed for Monte Carlo"

def test_put_call_parity_crr():
    S0, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.03
    call_res = price_option(S0, K, T, r, sigma, N=200, option_type="call", exercise="european", q=q)
    put_res = price_option(S0, K, T, r, sigma, N=200, option_type="put", exercise="european", q=q)
    
    lhs = call_res["price"] - put_res["price"]
    rhs = S0 * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 0.01, "Put-call parity failed for CRR"

def test_put_call_parity_heston():
    S0, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02
    call_res = heston_mc_price(S0, K, T, r, q=q, n_paths=50_000, option_type="call", seed=42)
    put_res = heston_mc_price(S0, K, T, r, q=q, n_paths=50_000, option_type="put", seed=42)
    
    lhs = call_res["price"] - put_res["price"]
    rhs = S0 * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 0.05, "Put-call parity failed for Heston"
