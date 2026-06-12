import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from black_scholes import black_scholes

# --- 1. Data Fetching ---
def download_data(ticker="SPY", start_date="2019-01-01", end_date="2024-01-01"):
    print(f"Downloading data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df

# --- 2. Data Preprocessing ---
def preprocess_data(df, window_size=60, vol_window=30):
    # Calculate daily returns
    df['Returns'] = df['Close'].pct_change()
    
    # Calculate rolling 30-day realized volatility (annualized)
    df['Realized_Vol'] = df['Returns'].rolling(window=vol_window).std() * np.sqrt(252)
    
    df.dropna(inplace=True)
    
    # Plot historical volatility
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Realized_Vol'], label=f"{vol_window}-day Realized Volatility")
    plt.title(f"Historical Realized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (Annualized)")
    plt.legend()
    plt.grid(True)
    plt.savefig("historical_volatility.png")
    plt.close()
    
    # Features: Returns, Realized Vol
    # The image mentioned VIX if available, but for simplicity we will use just Returns and Realized Vol.
    # We could add VIX, let's fetch it if possible.
    try:
        vix = yf.Ticker("^VIX").history(start=df.index[0], end=df.index[-1])
        # Ensure indices match by removing timezones
        df.index = df.index.tz_localize(None)
        vix.index = vix.index.tz_localize(None)
        
        df['VIX'] = vix['Close'] / 100.0  # as a percentage
        df.dropna(inplace=True)
        
        if len(df) == 0:
            raise ValueError("All rows dropped after VIX join")
            
        features = df[['Returns', 'Realized_Vol', 'VIX']].values
        print("Using Returns, Realized Vol, and VIX as features.")
    except Exception as e:
        print(f"VIX join failed: {e}. Falling back to 2 features.")
        df.dropna(inplace=True)
        features = df[['Returns', 'Realized_Vol']].values
        print("Using Returns and Realized Vol as features.")

    # Target: Next day's volatility
    targets = df['Realized_Vol'].shift(-1).values[:-1]
    features = features[:-1]
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences
    X, y, dates = [], [], []
    for i in range(len(scaled_features) - window_size):
        X.append(scaled_features[i:(i + window_size)])
        y.append(targets[i + window_size])
        dates.append(df.index[i + window_size + 1]) # Date of the target prediction
        
    X = np.array(X)
    y = np.array(y)
    
    # Train/Test Split (80/20 chronological)
    split_idx = int(0.8 * len(X))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Return previous day's realized vol for baseline comparison
    hist_vol_test = features[split_idx + window_size - 1:, 1] # Realized Vol column is idx 1
    
    return X_train_t, y_train_t, X_test_t, y_test_t, dates_test, hist_vol_test

# --- 3. LSTM Model ---
class LSTMVolatilityPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMVolatilityPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Get the output from the last time step
        out = out[:, -1, :]
        out = self.linear(out)
        return out

# --- 4. Training ---
def train_model(model, X_train, y_train, epochs=50, lr=0.001, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Further split train into train/val to watch for overfitting
    val_split = int(0.8 * len(X_train))
    X_t, X_v = X_train[:val_split], X_train[val_split:]
    y_t, y_v = y_train[:val_split], y_train[val_split:]
    
    train_dataset = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_losses = []
    val_losses = []
    
    print("Training started...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(X_t)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_v)
            val_loss = criterion(val_outputs, y_v).item()
            val_losses.append(val_loss)
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
            
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()
    
    return model

# --- 5. Evaluation and Comparison ---
def evaluate_and_compare(model, X_test, y_test, dates_test, hist_vol_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy().flatten()
        
    actuals = y_test.numpy().flatten()
    
    # Plot Actual vs Predicted Volatility
    plt.figure(figsize=(14, 7))
    plt.plot(dates_test, actuals, label='Actual Realized Volatility (Next Day)', color='blue', alpha=0.7)
    plt.plot(dates_test, preds, label='LSTM Predicted Volatility', color='orange', alpha=0.9)
    plt.title('LSTM Volatility Forecast vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.savefig("volatility_forecast_comparison.png")
    plt.close()
    
    return preds, actuals

# --- 6. Black-Scholes Pricing Experiment ---
def run_bs_experiment(preds, actuals, hist_vol_test):
    """
    Feed LSTM sigma into BS pricer - compare LSTM-BS vs historical-vol BS.
    We use actuals as the 'true' volatility for the pricing ground truth.
    """
    # Sample Option parameters
    S = 100.0
    K = 100.0
    T = 30 / 365.0  # 30 days to expiry
    r = 0.05
    
    mae_lstm = 0
    mae_hist = 0
    n = len(preds)
    
    for i in range(n):
        # We calculate the European Call price
        
        # 1. "True" Price using actual future realized volatility
        true_call, _ = black_scholes(S, K, T, r, actuals[i])
        
        # 2. LSTM Price using predicted volatility
        lstm_call, _ = black_scholes(S, K, T, r, preds[i])
        
        # 3. Historical Price using previous day's 30-day realized volatility
        hist_call, _ = black_scholes(S, K, T, r, hist_vol_test[i])
        
        mae_lstm += abs(lstm_call - true_call)
        mae_hist += abs(hist_call - true_call)
        
    mae_lstm /= n
    mae_hist /= n
    
    print("\n--- Black-Scholes Pricing Experiment Results ---")
    print(f"MAE (LSTM-BS vs True Price)       : {mae_lstm:.4f}")
    print(f"MAE (Historical-BS vs True Price) : {mae_hist:.4f}")
    print(f"Improvement using LSTM            : {mae_hist - mae_lstm:.4f}")
    

if __name__ == "__main__":
    # 1. Download data
    # Download 5 years up to a recent date
    df = download_data(ticker="SPY", start_date="2019-01-01", end_date="2024-01-01")
    
    # 2. Preprocess data
    X_train, y_train, X_test, y_test, dates_test, hist_vol_test = preprocess_data(df)
    
    # 3. Build model
    input_size = X_train.shape[2] # Number of features
    hidden_size = 64
    num_layers = 2
    model = LSTMVolatilityPredictor(input_size, hidden_size, num_layers, dropout=0.2)
    
    # 4. Train model
    model = train_model(model, X_train, y_train, epochs=50)
    
    # 5. Evaluate and Compare (Volatilities)
    preds, actuals = evaluate_and_compare(model, X_test, y_test, dates_test, hist_vol_test)
    
    # 6. Run BS Experiment
    run_bs_experiment(preds, actuals, hist_vol_test)
