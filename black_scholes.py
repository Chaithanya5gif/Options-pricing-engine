import math
import matplotlib.pyplot as plt


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


def black_scholes_delta(S, K, T, r, sigma):
    """
    Calculate Delta for European call and put options.

    Parameters
    ----------
    S     : float  Current stock price
    K     : float  Strike price
    T     : float  Time to expiry in years
    r     : float  Risk-free interest rate (annualised, continuous)
    sigma : float  Volatility of the underlying (annualised)

    Returns
    -------
    call_delta : float
    put_delta  : float
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    call_delta = norm_cdf(d1)
    put_delta = call_delta - 1.0
    return call_delta, put_delta


# ── Verification & Plotting ──────────────────────────────────────────────────
if __name__ == "__main__":
    K, T, r, sigma = 100, 1, 0.05, 0.2

    # 1. Verification of pricing & parity
    S_atm = 100
    call, put = black_scholes(S_atm, K, T, r, sigma)
    print(f"Pricing verification for ATM option (S={S_atm}, K={K}):")
    print(f"  Call price : {call:.4f}  (expected ≈ 10.45)")
    print(f"  Put  price : {put:.4f}")

    # Put-Call Parity check
    parity_lhs = call - put
    parity_rhs = S_atm - K * math.exp(-r * T)
    print(f"  Put-Call Parity holds: {math.isclose(parity_lhs, parity_rhs)}\n")

    # 2. Verification of Delta behavior (ATM, deep ITM, deep OTM)
    print("Delta verification:")
    for S_test, label in [(100, "ATM"), (150, "Deep ITM Call / OTM Put"), (50, "Deep OTM Call / ITM Put")]:
        c_delta, p_delta = black_scholes_delta(S_test, K, T, r, sigma)
        print(f"  S = {S_test:3d} ({label:23s}) -> Call Delta = {c_delta:6.4f}, Put Delta = {p_delta:6.4f}")

    # 3. Generate Delta vs Spot Price plot
    spot_prices = [s for s in range(50, 151)]
    call_deltas = []
    put_deltas = []
    for s in spot_prices:
        c_d, p_d = black_scholes_delta(s, K, T, r, sigma)
        call_deltas.append(c_d)
        put_deltas.append(p_d)

    plt.figure(figsize=(10, 6))
    plt.plot(spot_prices, call_deltas, label="Call Delta", color="#2563eb", lw=2)
    plt.plot(spot_prices, put_deltas, label="Put Delta", color="#dc2626", lw=2)
    plt.axvline(x=K, color="#6b7280", linestyle="--", label=f"Strike Price (K={K})")
    plt.xlabel("Spot Price (S)")
    plt.ylabel("Delta")
    plt.title("Black-Scholes Delta vs. Spot Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("delta_vs_spot.png", dpi=300)
    print("\nDelta vs Spot Price plot saved as 'delta_vs_spot.png'.")
