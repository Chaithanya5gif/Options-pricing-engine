import math
import matplotlib.pyplot as plt
import scipy.optimize


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


def norm_pdf(x):
    """
    Probability density function of the standard normal distribution.
    """
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x ** 2)


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
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    call_delta = norm_cdf(d1)
    put_delta = call_delta - 1.0
    return call_delta, put_delta


def black_scholes_gamma(S, K, T, r, sigma):
    """
    Calculate Gamma for European call and put options.
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))
    return gamma


def black_scholes_vega(S, K, T, r, sigma):
    """
    Calculate Vega for European call and put options.
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    vega = S * norm_pdf(d1) * math.sqrt(T)
    return vega


def black_scholes_theta(S, K, T, r, sigma):
    """
    Calculate Theta for European call and put options.
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    first_term = -(S * norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    second_term_call = r * K * math.exp(-r * T) * norm_cdf(d2)
    second_term_put = r * K * math.exp(-r * T) * norm_cdf(-d2)
    
    call_theta = first_term - second_term_call
    put_theta = first_term + second_term_put
    return call_theta, put_theta


def black_scholes_rho(S, K, T, r, sigma):
    """
    Calculate Rho for European call and put options.
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    call_rho = K * T * math.exp(-r * T) * norm_cdf(d2)
    put_rho = -K * T * math.exp(-r * T) * norm_cdf(-d2)
    return call_rho, put_rho


def implied_vol(market_price, S, K, T, r, option_type="call"):
    """
    Calculate the implied volatility of a European option using the Newton-Raphson method.
    
    Parameters
    ----------
    market_price : float  Observed option price
    S            : float  Current stock price
    K            : float  Strike price
    T            : float  Time to expiry in years
    r            : float  Risk-free interest rate (annualised, continuous)
    option_type  : str    "call" or "put"

    Returns
    -------
    sigma : float  Implied volatility
    """
    MAX_ITERATIONS = 100
    TOLERANCE = 0.0001
    
    sigma = 0.2  # Initial guess
    
    for _ in range(MAX_ITERATIONS):
        call_price, put_price = black_scholes(S, K, T, r, sigma)
        bs_price = call_price if option_type == "call" else put_price
        
        price_diff = bs_price - market_price
        
        if abs(price_diff) < TOLERANCE:
            return sigma
            
        vega = black_scholes_vega(S, K, T, r, sigma)
        
        if vega == 0.0:
            break
            
        sigma_new = sigma - price_diff / vega
        
        # Prevent negative volatility
        if sigma_new <= 0.0:
            sigma_new = 1e-5
            
        sigma = sigma_new

    # Fallback to Brent's method if Newton-Raphson fails
    def target_function(v):
        c, p = black_scholes(S, K, T, r, v)
        bs_p = c if option_type == "call" else p
        return bs_p - market_price

    try:
        sigma_brent = scipy.optimize.brentq(target_function, 0.001, 10.0)
        return sigma_brent
    except (ValueError, RuntimeError):
        return None


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

    # 2. Verification of Greek behaviors
    print("Greeks verification (ATM, S=100):")
    c_delta, p_delta = black_scholes_delta(S_atm, K, T, r, sigma)
    gamma = black_scholes_gamma(S_atm, K, T, r, sigma)
    vega = black_scholes_vega(S_atm, K, T, r, sigma)
    c_theta, p_theta = black_scholes_theta(S_atm, K, T, r, sigma)
    c_rho, p_rho = black_scholes_rho(S_atm, K, T, r, sigma)
    
    print(f"  Delta: Call = {c_delta:.4f}, Put = {p_delta:.4f}")
    print(f"  Gamma: {gamma:.4f}")
    print(f"  Vega : {vega:.4f}")
    print(f"  Theta: Call = {c_theta:.4f}, Put = {p_theta:.4f}")
    print(f"  Rho  : Call = {c_rho:.4f}, Put = {p_rho:.4f}\n")

    # 3. Generate the 5-Greeks Dashboard
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Black-Scholes Option Greeks Dashboard", fontsize=16, fontweight='bold')

    # Subplot 1: Delta vs Spot Price
    spot_prices = list(range(50, 151))
    call_deltas = [black_scholes_delta(s, K, T, r, sigma)[0] for s in spot_prices]
    put_deltas = [black_scholes_delta(s, K, T, r, sigma)[1] for s in spot_prices]
    axs[0, 0].plot(spot_prices, call_deltas, label="Call Delta", color="#2563eb", lw=2)
    axs[0, 0].plot(spot_prices, put_deltas, label="Put Delta", color="#dc2626", lw=2)
    axs[0, 0].axvline(x=K, color="#6b7280", linestyle="--", label=f"Strike (K={K})")
    axs[0, 0].set_xlabel("Spot Price (S)")
    axs[0, 0].set_ylabel("Delta")
    axs[0, 0].set_title("Delta vs. Spot Price")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Subplot 2: Gamma vs Spot Price
    gammas = [black_scholes_gamma(s, K, T, r, sigma) for s in spot_prices]
    axs[0, 1].plot(spot_prices, gammas, label="Gamma", color="#7c3aed", lw=2)
    axs[0, 1].axvline(x=K, color="#6b7280", linestyle="--", label=f"Strike (K={K})")
    axs[0, 1].set_xlabel("Spot Price (S)")
    axs[0, 1].set_ylabel("Gamma")
    axs[0, 1].set_title("Gamma vs. Spot Price (Peaks at ATM)")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Vega vs Volatility
    volatilities = [v / 100.0 for v in range(5, 81)]
    vegas = [black_scholes_vega(S_atm, K, T, r, v) for v in volatilities]
    axs[0, 2].plot([v * 100 for v in volatilities], vegas, label="Vega", color="#059669", lw=2)
    axs[0, 2].set_xlabel("Volatility (%)")
    axs[0, 2].set_ylabel("Vega")
    axs[0, 2].set_title("Vega vs. Volatility")
    axs[0, 2].legend()
    axs[0, 2].grid(True, alpha=0.3)

    # Subplot 4: Theta vs Time to Expiry
    times = [t / 100.0 for t in range(1, 201)]
    call_thetas = [black_scholes_theta(S_atm, K, t, r, sigma)[0] for t in times]
    put_thetas = [black_scholes_theta(S_atm, K, t, r, sigma)[1] for t in times]
    axs[1, 0].plot(times, call_thetas, label="Call Theta", color="#d97706", lw=2)
    axs[1, 0].plot(times, put_thetas, label="Put Theta", color="#db2777", lw=2)
    axs[1, 0].set_xlabel("Time to Expiry (Years)")
    axs[1, 0].set_ylabel("Theta")
    axs[1, 0].set_title("Theta vs. Time to Expiry")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Subplot 5: Rho vs Interest Rate
    rates = [ri / 100.0 for ri in range(0, 21)]
    call_rhos = [black_scholes_rho(S_atm, K, T, ri, sigma)[0] for ri in rates]
    put_rhos = [black_scholes_rho(S_atm, K, T, ri, sigma)[1] for ri in rates]
    axs[1, 1].plot([ri * 100 for ri in rates], call_rhos, label="Call Rho", color="#0284c7", lw=2)
    axs[1, 1].plot([ri * 100 for ri in rates], put_rhos, label="Put Rho", color="#b91c1c", lw=2)
    axs[1, 1].set_xlabel("Interest Rate (%)")
    axs[1, 1].set_ylabel("Rho")
    axs[1, 1].set_title("Rho vs. Interest Rate")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    # Subplot 6: Remove empty slot
    fig.delaxes(axs[1, 2])

    plt.tight_layout()
    plt.savefig("greeks_dashboard.png", dpi=300)
    print("Greeks dashboard saved as 'greeks_dashboard.png'.")

