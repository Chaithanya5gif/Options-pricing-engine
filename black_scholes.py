import math


def norm_cdf(x):
    """
    Cumulative distribution function of the standard normal distribution.
    Implemented from scratch using the Horner / rational approximation
    (Abramowitz & Stegun 26.2.17), accurate to ~1e-7.
    """
    x_abs = abs(x)

    t = 1.0 / (1.0 + 0.2316419 * x_abs)

    # Polynomial coefficients
    a1 =  0.319381530
    a2 = -0.356563782
    a3 =  1.781477937
    a4 = -1.821255978
    a5 =  1.330274429

    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    pdf  = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x_abs ** 2)

    cdf_positive = 1.0 - pdf * poly
    return cdf_positive if x >= 0 else 1.0 - cdf_positive


def black_scholes(S, K, T, r, sigma):
    """
    Black-Scholes European option pricing (pure Python).

    Parameters
    ----------
    S     : float  Current stock price
    K     : float  Strike price
    T     : float  Time to expiry in years
    r     : float  Risk-free interest rate (annualised, continuous)
    sigma : float  Volatility of the underlying (annualised)

    Returns
    -------
    call_price : float
    put_price  : float
    """
    # d1 and d2 — the heart of the model
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Present value of the strike
    pv_K = K * math.exp(-r * T)

    call_price = S * norm_cdf(d1) - pv_K * norm_cdf(d2)
    put_price  = pv_K * norm_cdf(-d2) - S * norm_cdf(-d1)

    return call_price, put_price


# ── Verification ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    call, put = black_scholes(S, K, T, r, sigma)

    print(f"S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"  Call price : {call:.4f}  (expected ≈ 10.45)")
    print(f"  Put  price : {put:.4f}")

    # Put-Call Parity check: C - P = S - K*e^(-rT)
    parity_lhs = call - put
    parity_rhs = S - K * math.exp(-r * T)
    print(f"\nPut-Call Parity check: C - P = {parity_lhs:.4f}, S - PV(K) = {parity_rhs:.4f}")
    print(f"  Parity holds: {math.isclose(parity_lhs, parity_rhs)}")
