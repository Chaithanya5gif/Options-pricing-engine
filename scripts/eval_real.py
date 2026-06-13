import torch
import yfinance as yf
from src.pricers.neural_net import MLPPricer
import pandas as pd
import numpy as np
from src.pricers.bs import black_scholes


def test_real_market_fixed(model, device, ticker="SPY"):
    print(f"\n--- Real Market Test on {ticker} ---")

    try:
        stock = yf.Ticker(ticker)
        todays_data = stock.history(period="1d")
        S = todays_data["Close"].iloc[-1]

        expiries = stock.options
        if not expiries:
            print("No options data available.")
            return

        # Pick an expiry with T >= 0.1 years (approx 36 days) to be in-distribution
        current_date = pd.Timestamp.now("UTC")
        selected_expiry = None
        for exp in expiries:
            exp_date = pd.to_datetime(exp).tz_localize("UTC")
            T_days = (exp_date - current_date).days
            if T_days >= 36:
                selected_expiry = exp
                break

        if not selected_expiry:
            selected_expiry = expiries[-1]

        print(f"Spot Price (S): {S:.2f}, Selected Expiry: {selected_expiry}")

        chain = stock.option_chain(selected_expiry)
        calls = chain.calls

        # Filter for somewhat reasonable volume/open interest
        calls = calls[
            (calls["volume"] > 0) & (calls["impliedVolatility"] > 0.01)
        ].copy()

        exp_date = pd.to_datetime(selected_expiry).tz_localize("UTC")
        T_days = (exp_date - current_date).days
        if T_days <= 0:
            T_days = 1
        T = T_days / 365.0
        r = 0.05  # Approximate risk-free rate

        results = []

        for _, row in calls.iterrows():
            K = row["strike"]
            market_price = (row["bid"] + row["ask"]) / 2.0
            sigma = row["impliedVolatility"]

            # BS Price
            bs_c, _ = black_scholes(S, K, T, r, sigma)

            # MLP Price
            scale_factor = S / 100.0
            S_scaled = 100.0
            K_scaled = K / scale_factor

            inp_tensor = torch.tensor(
                [[S_scaled, K_scaled, T, r, sigma]], dtype=torch.float32
            ).to(device)
            with torch.no_grad():
                mlp_c_scaled = model(inp_tensor).item()
            mlp_c = mlp_c_scaled * scale_factor

            results.append(
                {
                    "Strike": K,
                    "Market": market_price,
                    "BS": bs_c,
                    "MLP": mlp_c,
                    "IV": sigma,
                    "Vol": row["volume"],
                }
            )

        res_df = pd.DataFrame(results)
        if len(res_df) == 0:
            print("No valid options left after filtering.")
            return

        res_df["BS_Err"] = np.abs(res_df["BS"] - res_df["Market"])
        res_df["MLP_Err"] = np.abs(res_df["MLP"] - res_df["Market"])

        # Sort by strike
        res_df = res_df.sort_values("Strike").reset_index(drop=True)

        print("\nSample Results (around ATM):")
        atm_idx = (res_df["Strike"] - S).abs().idxmin()
        sample_df = res_df.iloc[max(0, atm_idx - 5) : min(len(res_df), atm_idx + 6)]
        print(sample_df.to_string(index=False, float_format="%.2f"))

        bs_mean_err = res_df["BS_Err"].mean()
        mlp_mean_err = res_df["MLP_Err"].mean()

        print(f"\nOverall Mean Absolute Error vs Market:")
        print(f"BS  Error: {bs_mean_err:.4f}")
        print(f"MLP Error: {mlp_mean_err:.4f}")

        # Check where MLP beats BS
        mlp_wins = res_df[res_df["MLP_Err"] < res_df["BS_Err"]]
        print(
            f"MLP is closer to market than BS on {len(mlp_wins)} / {len(res_df)} options."
        )
        if len(mlp_wins) > 0:
            print("\nExamples where MLP won:")
            print(mlp_wins.head().to_string(index=False, float_format="%.2f"))

    except Exception as e:
        print(f"Failed to fetch or process market data: {e}")


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
model = MLPPricer().to(device)
model.load_state_dict(torch.load("mlp_pricer.pth", map_location=device))
model.eval()

test_real_market_fixed(model, device, ticker="SPY")
