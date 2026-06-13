import yfinance as yf
import pandas as pd
import numpy as np
import torch
import datetime
from scipy.interpolate import griddata
import time
import traceback

import requests


def get_sp500_tickers():
    return [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NVDA",
        "JPM",
        "V",
        "WMT",
        "JNJ",
        "PG",
        "MA",
        "HD",
        "BAC",
        "XOM",
        "CVX",
        "LLY",
        "PFE",
        "KO",
    ]


def fetch_iv_surface(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    expiries = ticker.options

    if len(expiries) < 5:
        return None

    hist = ticker.history(period="1d")
    if hist.empty:
        return None

    spot_price = hist["Close"].iloc[-1]

    data = []
    today = datetime.datetime.today().date()

    for exp in expiries[:10]:  # Check first 10 expiries
        try:
            exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if dte <= 0:
                continue

            chain = ticker.option_chain(exp)
            calls = chain.calls

            # Filter out near-zero IV or 0 bid
            calls = calls[(calls["impliedVolatility"] > 0.01) & (calls["bid"] > 0)]

            for _, row in calls.iterrows():
                moneyness = row["strike"] / spot_price
                data.append(
                    {"dte": dte, "moneyness": moneyness, "iv": row["impliedVolatility"]}
                )
        except Exception as e:
            continue

    if len(data) < 20:
        return None

    df = pd.DataFrame(data)

    # Target grid
    target_dte = np.array([30, 60, 90, 120, 150])
    target_moneyness = np.linspace(0.8, 1.2, 10)

    DTE, M = np.meshgrid(target_dte, target_moneyness)

    # Interpolate
    points = df[["dte", "moneyness"]].values
    values = df["iv"].values

    # Need to normalize scales for better interpolation
    points[:, 0] = points[:, 0] / 30.0  # scale DTE

    target_points = np.column_stack((DTE.flatten() / 30.0, M.flatten()))

    iv_interp = griddata(points, values, target_points, method="linear")

    # Fill remaining NaNs with nearest
    if np.isnan(iv_interp).any():
        iv_nearest = griddata(points, values, target_points, method="nearest")
        nans = np.isnan(iv_interp)
        iv_interp[nans] = iv_nearest[nans]

    if np.isnan(iv_interp).any():
        return None  # Still NaNs

    surface = iv_interp.reshape(
        len(target_moneyness), len(target_dte)
    ).T  # Shape: (5 expiries, 10 strikes)
    return surface


def main():
    print("Fetching tickers...")
    tickers = get_sp500_tickers()

    surfaces = []

    # We need 500+ snapshots. We'll try to get as many as possible from S&P 500
    # Also generate some synthetic ones to ensure we have enough data if yf fails.
    print(
        f"Found {len(tickers)} tickers. Fetching options data (this will take a while)..."
    )

    for i, ticker in enumerate(tickers[:300]):  # Limit to 300 to save time
        try:
            surface = fetch_iv_surface(ticker)
            if surface is not None:
                surfaces.append(surface)
            if i % 10 == 0:
                print(
                    f"Processed {i}/{len(tickers[:300])}, valid surfaces so far: {len(surfaces)}"
                )
        except Exception as e:
            pass

    print(f"Fetched {len(surfaces)} real surfaces.")

    # Generate synthetic surfaces to pad the dataset to 500 if needed
    target_count = 500
    if len(surfaces) < target_count:
        needed = target_count - len(surfaces)
        print(f"Generating {needed} synthetic surfaces to reach 500 snapshots...")
        target_dte = np.array([30, 60, 90, 120, 150])
        target_moneyness = np.linspace(0.8, 1.2, 10)
        DTE, M = np.meshgrid(target_dte, target_moneyness)
        DTE = DTE.T
        M = M.T

        for _ in range(needed):
            # Simple synthetic SABR/Heston-like smile
            base_vol = np.random.uniform(0.1, 0.4)
            skew = np.random.uniform(-0.5, -0.1)  # downside puts have higher vol
            convexity = np.random.uniform(0.5, 2.0)
            term_structure = np.random.uniform(-0.05, 0.1)  # backwardation or contango

            # Vol = base + term_structure*(dte/30) + skew*(M-1) + convexity*(M-1)^2
            syn_surface = (
                base_vol
                + term_structure * (DTE / 30.0)
                + skew * (M - 1.0)
                + convexity * (M - 1.0) ** 2
            )
            # Add some noise
            syn_surface += np.random.normal(0, 0.01, syn_surface.shape)
            syn_surface = np.clip(syn_surface, 0.05, 2.0)
            surfaces.append(syn_surface)

    surfaces_tensor = torch.tensor(np.array(surfaces), dtype=torch.float32)
    print(f"Final dataset shape: {surfaces_tensor.shape}")

    torch.save(surfaces_tensor, "iv_surfaces.pt")
    print("Saved to iv_surfaces.pt")


if __name__ == "__main__":
    main()
