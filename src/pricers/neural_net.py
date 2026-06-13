import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import yfinance as yf
from src.pricers.bs import black_scholes
from datetime import datetime, timezone


# 1. Data Generation
def generate_synthetic_data(num_samples=100000):
    """Generate synthetic options data for training.

    Args:
        num_samples (int, optional): Number of samples to generate. Defaults to 100000.

    Returns:
        pd.DataFrame: DataFrame containing generated data.
    """
    # S in [50, 150], K in [50, 150], T in [0.1, 2.0], r in [0.01, 0.10], sigma in [0.05, 0.80]
    S = np.random.uniform(50, 150, num_samples)
    K = np.random.uniform(50, 150, num_samples)
    T = np.random.uniform(0.1, 2.0, num_samples)
    r = np.random.uniform(0.01, 0.10, num_samples)
    sigma = np.random.uniform(0.05, 0.80, num_samples)

    call_prices = np.zeros(num_samples)

    for i in range(num_samples):
        c, _ = black_scholes(S[i], K[i], T[i], r[i], sigma[i])
        call_prices[i] = c

    df = pd.DataFrame(
        {
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "call_price": call_prices,
            "M": np.log(K / S),
            "norm_price": call_prices / S,
        }
    )
    return df


# 2. MLP Architecture
class MLPPricer(nn.Module):
    """Multi-Layer Perceptron model for options pricing."""

    def __init__(self):
        super(MLPPricer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# 3. Training Function
def train_model(df):
    """Train the MLP model on the provided dataframe.

    Args:
        df (pd.DataFrame): Training data.

    Returns:
        tuple: A tuple containing the trained model, test dataloader, and device used.
    """
    X = df[["M", "T", "r", "sigma"]].values
    y = df["norm_price"].values.reshape(-1, 1)
    S_vals = df["S"].values.reshape(-1, 1)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    S_tensor = torch.tensor(S_vals, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor, S_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = MLPPricer().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    train_losses = []
    val_losses = []

    print(f"Training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, _ in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
            )

    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("MLP Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("mlp_loss_curve.png", dpi=300)
    plt.close()
    print("Training curve saved to 'mlp_loss_curve.png'")

    torch.save(model.state_dict(), "mlp_pricer.pth")
    print("Model saved to 'mlp_pricer.pth'")

    return model, test_loader, device


# 4. Evaluation Function
def evaluate_model(model, test_loader, device):
    """Evaluate the model and perform out-of-distribution tests.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for testing.
        device (torch.device): Device to run inference on.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch, S_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.append((outputs.cpu() * S_batch).numpy())
            all_targets.append((y_batch * S_batch).numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

    print("\n--- Model Evaluation ---")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    # OOD Tests
    print("\n--- Out-of-Distribution (OOD) Tests ---")
    # [S, K, T, r, sigma]
    ood_inputs = [
        [50.0, 150.0, 1.0, 0.05, 0.2],  # Deep OTM
        [150.0, 50.0, 1.0, 0.05, 0.2],  # Deep ITM
        [100.0, 100.0, 0.001, 0.05, 0.2],  # Very short expiry
        [100.0, 100.0, 5.0, 0.05, 0.2],  # Very long expiry
        [100.0, 100.0, 1.0, 0.05, 1.5],  # Extreme vol
        [5.0, 100.0, 1.0, 0.05, 0.2],  # Extremely low price
        [500.0, 100.0, 1.0, 0.05, 0.2],  # Extremely high price
    ]
    ood_names = [
        "Deep OTM",
        "Deep ITM",
        "Short Expiry",
        "Long Expiry",
        "Extreme Vol",
        "Extr. Low Spot",
        "Extr. High Spot",
    ]

    for name, inp in zip(ood_names, ood_inputs):
        S, K, T, r, sigma = inp
        bs_c, _ = black_scholes(S, K, T, r, sigma)

        M_val = np.log(K / S)
        inp_tensor = torch.tensor([[M_val, T, r, sigma]], dtype=torch.float32).to(device)
        with torch.no_grad():
            mlp_norm_c = model(inp_tensor).item()
            mlp_c = mlp_norm_c * S

        print(
            f"{name:15s} -> BS: {bs_c:8.4f} | MLP: {mlp_c:8.4f} | Diff: {mlp_c - bs_c:8.4f}"
        )


# 5. Real Market Test
def test_real_market(model, device, ticker="SPY"):
    """Test the model against real market data.

    Args:
        model (nn.Module): Trained model.
        device (torch.device): Device to run inference on.
        ticker (str, optional): Ticker symbol. Defaults to "SPY".
    """
    print(f"\n--- Real Market Test on {ticker} ---")

    try:
        stock = yf.Ticker(ticker)
        todays_data = stock.history(period="1d")
        S = todays_data["Close"].iloc[-1]

        expiries = stock.options
        if not expiries:
            print("No options data available.")
            return

        # Pick an expiry ~1 month out if possible
        expiry = expiries[min(len(expiries) - 1, 4)]
        print(f"Spot Price (S): {S:.2f}, Selected Expiry: {expiry}")

        chain = stock.option_chain(expiry)
        calls = chain.calls

        # Filter for somewhat reasonable volume/open interest
        calls = calls[
            (calls["volume"] > 0) & (calls["impliedVolatility"] > 0.01)
        ].copy()

        # Make timestamps timezone-aware for math
        expiry_date = pd.to_datetime(expiry).tz_localize("UTC")
        current_date = pd.Timestamp.now("UTC")
        T_days = (expiry_date - current_date).days

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

            # Predict C/S using M = ln(K/S), T, r, sigma
            M_val = np.log(K / S)
            inp_tensor = torch.tensor(
                [[M_val, T, r, sigma]], dtype=torch.float32
            ).to(device)
            with torch.no_grad():
                mlp_norm_c = model(inp_tensor).item()
            mlp_c = mlp_norm_c * S

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

    except Exception as e:
        print(f"Failed to fetch or process market data: {e}")


if __name__ == "__main__":
    print("1. Generating Synthetic Data...")
    np.random.seed(42)
    torch.manual_seed(42)

    df = generate_synthetic_data(100000)
    print("Sample generated data:")
    print(df.head())

    print("\n2 & 3. Building and Training Model...")
    model, test_loader, device = train_model(df)

    print("\n4. Evaluating Model...")
    evaluate_model(model, test_loader, device)

    print("\n5. Testing on Real Market Data...")
    test_real_market(model, device, ticker="SPY")
