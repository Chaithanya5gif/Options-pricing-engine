import math
from typing import Tuple
from src.pricers.bs import norm_cdf, norm_pdf


def black_scholes_delta(
    S: float, K: float, T: float, r: float, sigma: float
) -> Tuple[float, float]:
    """Calculate Delta for European call and put options.

    Delta measures the rate of change of the theoretical option value with
    respect to changes in the underlying asset's price.

    Args:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to expiry in years.
        r (float): Risk-free interest rate (annualised, continuous).
        sigma (float): Volatility of the underlying (annualised).

    Returns:
        Tuple[float, float]: A tuple containing the call delta and put delta.
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    call_delta = norm_cdf(d1)
    put_delta = call_delta - 1.0
    return call_delta, put_delta


def black_scholes_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Gamma for European call and put options.

    Gamma measures the rate of change in the delta with respect to changes
    in the underlying price.

    Args:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to expiry in years.
        r (float): Risk-free interest rate (annualised, continuous).
        sigma (float): Volatility of the underlying (annualised).

    Returns:
        float: The gamma value (identical for call and put).
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))
    return gamma


def black_scholes_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Vega for European call and put options.

    Vega measures sensitivity to volatility. Vega is the derivative of the
    option value with respect to the volatility of the underlying asset.

    Args:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to expiry in years.
        r (float): Risk-free interest rate (annualised, continuous).
        sigma (float): Volatility of the underlying (annualised).

    Returns:
        float: The vega value (identical for call and put).
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    vega = S * norm_pdf(d1) * math.sqrt(T)
    return vega


def black_scholes_theta(
    S: float, K: float, T: float, r: float, sigma: float
) -> Tuple[float, float]:
    """Calculate Theta for European call and put options.

    Theta measures the sensitivity of the value of the derivative to the
    passage of time (time decay).

    Args:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to expiry in years.
        r (float): Risk-free interest rate (annualised, continuous).
        sigma (float): Volatility of the underlying (annualised).

    Returns:
        Tuple[float, float]: A tuple containing the call theta and put theta.
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    first_term = -(S * norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    second_term_call = r * K * math.exp(-r * T) * norm_cdf(d2)
    second_term_put = r * K * math.exp(-r * T) * norm_cdf(-d2)

    call_theta = first_term - second_term_call
    put_theta = first_term + second_term_put
    return call_theta, put_theta


def black_scholes_rho(
    S: float, K: float, T: float, r: float, sigma: float
) -> Tuple[float, float]:
    """Calculate Rho for European call and put options.

    Rho measures sensitivity to the interest rate: it is the derivative of
    the option value with respect to the risk-free interest rate.

    Args:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to expiry in years.
        r (float): Risk-free interest rate (annualised, continuous).
        sigma (float): Volatility of the underlying (annualised).

    Returns:
        Tuple[float, float]: A tuple containing the call rho and put rho.
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_rho = K * T * math.exp(-r * T) * norm_cdf(d2)
    put_rho = -K * T * math.exp(-r * T) * norm_cdf(-d2)
    return call_rho, put_rho
