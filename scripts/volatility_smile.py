import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from src.pricers.bs import implied_vol, black_scholes


def verify_implied_vol():
    print("=== Task 3: Verifying implied_vol implementation ===")
    S, K, T, r, sigma_true = 100, 105, 1.0, 0.05, 0.2

    # Generate BS price
    call_price, put_price = black_scholes(S, K, T, r, sigma_true)
    print(f"Generated BS Call Price: {call_price:.4f} for sigma={sigma_true}")

    # Feed back into IV calculator
    sigma_implied = implied_vol(call_price, S, K, T, r, option_type="call")
    print(f"Recovered Implied Volatility: {sigma_implied:.4f}")

    assert abs(sigma_true - sigma_implied) < 0.001, "Implied vol verification failed!"
    print("Verification passed: implied_vol successfully recovered sigma.\n")


def fetch_and_plot_volatility_smile():
    print("=== Task 4 & 5: Fetching data and plotting Volatility Smile ===")
    print("Fetching real options data for SPY...")
    ticker = yf.Ticker("SPY")

    # Get available expiration dates
    expirations = ticker.options
    if not expirations:
        print("No options data available for SPY.")
        return

    # Pick nearest expiry with good liquidity
    target_expiry = expirations[min(2, len(expirations) - 1)]
    print(f"Using expiry date: {target_expiry}")

    chain = ticker.option_chain(target_expiry)
    calls = chain.calls

    S = ticker.history(period="1d")["Close"].iloc[-1]

    # Calculate time to expiration T
    expiry_date = datetime.datetime.strptime(target_expiry, "%Y-%m-%d").date()
    today = datetime.date.today()
    days_to_expiry = (expiry_date - today).days
    if days_to_expiry <= 0:
        days_to_expiry = 1  # Prevent zero division
    T = days_to_expiry / 365.0
    r = 0.05  # Assume 5% risk-free rate

    # Filter strikes near money for liquidity
    calls = calls[(calls["strike"] > S * 0.8) & (calls["strike"] < S * 1.2)].copy()

    # Calculate IV for each strike
    implied_vols = []
    strikes = []

    print(f"Current SPY price: {S:.2f}")
    print("Calculating IV for strikes...")

    for _, row in calls.iterrows():
        K = row["strike"]
        market_price = row["lastPrice"]

        # Skip stale or zero prices
        if market_price <= 0:
            continue

        iv = implied_vol(market_price, S, K, T, r, option_type="call")

        if iv is not None:
            implied_vols.append(iv)
            strikes.append(K)

    if not implied_vols:
        print("Failed to calculate IV for any strikes.")
        return

    print(f"Calculated IV for {len(strikes)} strikes. Plotting...")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(
        strikes,
        implied_vols,
        marker="o",
        linestyle="-",
        color="b",
        label="Implied Volatility",
    )
    plt.axvline(S, color="r", linestyle="--", label=f"Current Price (S={S:.2f})")

    plt.title(f"Volatility Smile for SPY (Expiry: {target_expiry})")
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Implied Volatility (\u03c3)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig("volatility_smile.png")
    print("Volatility smile plot saved as 'volatility_smile.png'.")


if __name__ == "__main__":
    verify_implied_vol()
    fetch_and_plot_volatility_smile()
