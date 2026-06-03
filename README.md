# Options Pricing Engine

This repository implements a fully featured Black-Scholes options pricing engine and implied volatility calculator from scratch in Python.

## Black-Scholes Mathematical Derivation

The Black-Scholes model is based on the assumption that the underlying asset's price follows a geometric Brownian motion:

$$
dS = \mu S dt + \sigma S dW
$$

where $S$ is the stock price, $\mu$ is the drift, $\sigma$ is the volatility, and $dW$ is a Wiener process.

### The Black-Scholes PDE
By constructing a risk-neutral portfolio (delta hedging), we eliminate the market risk. Applying Itô's Lemma to the option price $V(S,t)$, the no-arbitrage condition leads to the Black-Scholes Partial Differential Equation (PDE):

$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0
$$

where $r$ is the risk-free interest rate.

### Boundary Conditions
For a European Call option $C(S,t)$ with strike $K$ and expiration $T$, the terminal boundary condition is:

$$
C(S,T) = \max(S - K, 0)
$$

### Closed-Form Solution
Solving the PDE yields the Black-Scholes pricing formula for a European Call option:

$$
C(S,t) = S \cdot N(d_1) - K e^{-r(T-t)} \cdot N(d_2)
$$

Where $N(x)$ is the cumulative distribution function of the standard normal distribution, and:

$$
d_1 = \frac{\ln(S/K) + (r + \frac{\sigma^2}{2})(T-t)}{\sigma \sqrt{T-t}}
$$

$$
d_2 = d_1 - \sigma \sqrt{T-t}
$$

For a European Put option $P(S,t)$, the pricing formula by put-call parity is:

$$
P(S,t) = K e^{-r(T-t)} \cdot N(-d_2) - S \cdot N(-d_1)
$$

## Features
- **Pricing & Greeks**: Computes Call/Put prices and Greeks (Delta, Gamma, Vega, Theta, Rho)
- **Implied Volatility**: Uses Newton-Raphson iteration with Brent's method as a fallback for guaranteed convergence.
- **Volatility Smile & Surface**: Fetches real market data using `yfinance` to plot the 2D volatility smile and the 3D volatility surface across strikes and expiries.

## Visualizations
The code generates robust visualizations using `matplotlib`:
- `greeks_dashboard.png`: Dashboard of 5 Greeks
- `volatility_smile.png`: Implied Volatility vs Strike Price
- `volatility_surface_3d.png`: 3D surface plot of Implied Volatility over Strike and Time to Expiry
