#!/usr/bin/env python3
"""
run_definitive_experiment.py
============================
Definitive comparison of 6 options pricing methods on live SPY market data.

Methods tested:
  1. Black-Scholes (BS)                  — analytical, market IV as input
  2. CRR Binomial Tree (N=200)           — discrete lattice, European
  3. Monte Carlo Control Variate (100k)  — GBM simulation
  4. LSTM-BS Hybrid                      — LSTM vol forecast → BS price
  5. MLP Pricer                          — direct neural network (mlp_pricer.pth)
  6. VAE-Interpolated IV → BS            — VAE smoothed surface → BS price

Outputs:
  - test_options.csv       Raw held-out test dataset with market mid-prices
  - results_detailed.csv   All options × all 6 method predictions
  - comparison_table.csv   Per-method aggregate metrics
  - comparison_table.png   Publication-ready styled figure (Table 2)
"""

import os
import sys
import time
import math
import warnings
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Local imports ────────────────────────────────────────────────────────────
from black_scholes import black_scholes
from crr_binomial_tree import price_option
from monte_carlo import mc_price
from mlp_pricer import MLPPricer
from vae_model import VAE
from lstm_volatility_pricer import (
    LSTMVolatilityPredictor,
    download_data,
    preprocess_data,
    train_model as lstm_train,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — FETCH REAL OPTIONS DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_spy_options(target_count: int = 200) -> pd.DataFrame:
    """
    Pull SPY call options across multiple expiries, collect up to target_count rows.
    Ground truth = mid-price (bid + ask) / 2.
    Filters: bid > 0, IV > 1%, volume > 0, T > 7 days.
    Returns a DataFrame with columns:
      ticker, expiry, S, K, T, r, sigma, market_price, moneyness, bucket
    """
    print("\n" + "=" * 65)
    print("  STEP 1 — Fetching real SPY options data")
    print("=" * 65)

    ticker_sym = "SPY"
    r = 0.0525  # approximate Fed funds rate

    stock = yf.Ticker(ticker_sym)
    hist = stock.history(period="1d")
    S = float(hist["Close"].iloc[-1])
    print(f"  SPY spot price: ${S:.2f}")

    expiries = stock.options
    print(f"  Available expiries: {len(expiries)}")

    today = pd.Timestamp.now(tz="UTC")
    rows = []

    for exp in expiries:
        if len(rows) >= target_count * 2:  # collect extra, then trim
            break
        try:
            exp_ts  = pd.to_datetime(exp).tz_localize("UTC")
            T_days  = (exp_ts - today).days
            if T_days < 7:
                continue
            T = T_days / 365.0

            chain = stock.option_chain(exp)
            calls = chain.calls.copy()

            # Quality filters
            calls = calls[
                (calls["bid"] > 0) &
                (calls["ask"] > calls["bid"]) &
                (calls["impliedVolatility"] > 0.01) &
                (calls["volume"] > 0)
            ]
            # Keep strikes in 0.70–1.30 moneyness range (avoid extreme illiquid strikes)
            calls = calls[(calls["strike"] / S >= 0.70) & (calls["strike"] / S <= 1.30)]

            for _, row in calls.iterrows():
                K           = float(row["strike"])
                mid_price   = (float(row["bid"]) + float(row["ask"])) / 2.0
                sigma       = float(row["impliedVolatility"])
                moneyness   = K / S

                if moneyness < 0.90:
                    bucket = "Deep ITM"
                elif moneyness < 0.975:
                    bucket = "ITM"
                elif moneyness <= 1.025:
                    bucket = "ATM"
                elif moneyness <= 1.10:
                    bucket = "OTM"
                else:
                    bucket = "Deep OTM"

                rows.append({
                    "expiry"      : exp,
                    "S"           : S,
                    "K"           : K,
                    "T"           : T,
                    "r"           : r,
                    "sigma"       : sigma,
                    "market_price": mid_price,
                    "moneyness"   : moneyness,
                    "bucket"      : bucket,
                })

        except Exception as e:
            print(f"  Warning — skipped expiry {exp}: {e}")
            continue

    df = pd.DataFrame(rows)
    print(f"  Collected {len(df)} qualifying options across expiries")

    # Stratified sample: aim for balanced buckets, up to target_count
    bucket_order = ["Deep ITM", "ITM", "ATM", "OTM", "Deep OTM"]
    sampled = []
    per_bucket = max(1, target_count // len(bucket_order))
    for b in bucket_order:
        sub = df[df["bucket"] == b]
        n   = min(len(sub), per_bucket)
        if n > 0:
            sampled.append(sub.sample(n, random_state=42))

    # Fill remaining quota from any bucket
    df_sampled  = pd.concat(sampled).drop_duplicates()
    remaining   = target_count - len(df_sampled)
    if remaining > 0:
        leftover = df[~df.index.isin(df_sampled.index)]
        df_sampled = pd.concat([df_sampled, leftover.sample(
            min(remaining, len(leftover)), random_state=42)])

    df_sampled = df_sampled.reset_index(drop=True)
    df_sampled.to_csv("test_options.csv", index=False)
    print(f"  Final test set: {len(df_sampled)} options saved → test_options.csv")

    # Print bucket distribution
    print("\n  Bucket distribution:")
    for b in bucket_order:
        n = (df_sampled["bucket"] == b).sum()
        print(f"    {b:<12}: {n:3d}")
    print()
    return df_sampled


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — LOAD / PREPARE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def load_mlp(device) -> MLPPricer:
    """Load pre-trained MLP pricer."""
    print("  Loading MLP pricer (mlp_pricer.pth)...")
    model = MLPPricer().to(device)
    model.load_state_dict(torch.load("mlp_pricer.pth", map_location=device, weights_only=True))
    model.eval()
    print("  MLP loaded ✓")
    return model


def load_vae(device) -> VAE:
    """Load pre-trained VAE."""
    print("  Loading VAE (vae_weights.pth)...")
    model = VAE(input_dim=50, latent_dim=8).to(device)
    model.load_state_dict(torch.load("vae_weights.pth", map_location=device, weights_only=True))
    model.eval()
    print("  VAE loaded ✓")
    return model


def train_lstm_quick(device) -> tuple:
    """
    Download SPY historical data, preprocess, and train LSTM for 30 epochs.
    Returns (model, predicted_vol) where predicted_vol is the LSTM's
    forecast of today's volatility — used as sigma for all BS calculations.

    Note: lstm_train() in lstm_volatility_pricer.py keeps all data on CPU
    (no .to(device) calls on batches), so we train on CPU and keep the model
    on CPU throughout. Only inference is needed after, which is also CPU.
    """
    print("  Training LSTM on SPY 2019-2024 (30 epochs)...")
    df_hist = download_data("SPY", "2019-01-01", "2024-06-01")
    X_train, y_train, X_test, y_test, dates_test, hist_vol_test = preprocess_data(df_hist)

    input_size = X_train.shape[2]
    # Keep LSTM on CPU — lstm_train does not move tensors to device
    lstm_cpu = LSTMVolatilityPredictor(input_size=input_size, hidden_size=64,
                                       num_layers=2, dropout=0.2)
    # Quick train — 30 epochs
    lstm_cpu = lstm_train(lstm_cpu, X_train, y_train, epochs=30, lr=0.001)
    lstm_cpu.eval()

    # Predict on the last sequence in test set (represents "today's" vol)
    with torch.no_grad():
        preds = lstm_cpu(X_test).numpy().flatten()

    lstm_sigma = float(np.median(preds[-20:]))  # median of last 20 predictions
    print(f"  LSTM predicted sigma: {lstm_sigma:.4f} ✓")
    return lstm_cpu, lstm_sigma


def build_vae_iv_surface(df: pd.DataFrame, vae: VAE, S: float) -> np.ndarray:
    """
    Build an IV surface from live market data, pass through VAE to smooth it,
    return the smoothed surface on the standard grid.

    Grid: 5 expiries [30,60,90,120,150] days × 10 moneyness [0.80..1.20]
    """
    target_dte        = np.array([30, 60, 90, 120, 150], dtype=float)
    target_moneyness  = np.linspace(0.80, 1.20, 10)

    # Collect raw market points
    raw_pts  = []
    raw_vals = []
    for _, row in df.iterrows():
        dte = row["T"] * 365.0
        m   = row["moneyness"]
        iv  = row["sigma"]
        raw_pts.append([dte / 30.0, m])   # normalise DTE
        raw_vals.append(iv)

    raw_pts  = np.array(raw_pts)
    raw_vals = np.array(raw_vals)

    # Interpolate onto standard grid
    DTE_g, M_g = np.meshgrid(target_dte, target_moneyness)   # (10,5)
    DTE_g, M_g = DTE_g.T, M_g.T                              # → (5,10)

    target_pts = np.column_stack((DTE_g.flatten() / 30.0, M_g.flatten()))
    iv_interp  = griddata(raw_pts, raw_vals, target_pts, method="linear")
    iv_nearest = griddata(raw_pts, raw_vals, target_pts, method="nearest")
    nans       = np.isnan(iv_interp)
    iv_interp[nans] = iv_nearest[nans]
    iv_interp  = np.clip(iv_interp, 0.01, 3.0)

    surface_raw = iv_interp.reshape(5, 10)   # (5 expiries, 10 strikes)

    # Pass through VAE for smoothing / denoising
    vae_device = next(vae.parameters()).device
    surf_tensor = torch.tensor(surface_raw.flatten(), dtype=torch.float32).unsqueeze(0).to(vae_device)
    with torch.no_grad():
        recon, mu, _ = vae(surf_tensor)
    # Use the mean (deterministic decode) for a smooth surface
    with torch.no_grad():
        smooth_surf = vae.decode(mu).squeeze(0).cpu().numpy().reshape(5, 10)

    smooth_surf = np.clip(smooth_surf, 0.01, 3.0)
    return smooth_surf, target_dte, target_moneyness


def vae_iv_for_option(smooth_surf, target_dte, target_moneyness, T_years, moneyness) -> float:
    """
    Bilinear interpolation on the smooth VAE surface to get IV for a given (T, moneyness).
    """
    dte = T_years * 365.0
    # Clamp to grid bounds
    dte = float(np.clip(dte, target_dte[0], target_dte[-1]))
    m   = float(np.clip(moneyness, target_moneyness[0], target_moneyness[-1]))

    # Find surrounding indices
    i0 = np.searchsorted(target_dte, dte, side="right") - 1
    i0 = int(np.clip(i0, 0, len(target_dte) - 2))
    j0 = np.searchsorted(target_moneyness, m, side="right") - 1
    j0 = int(np.clip(j0, 0, len(target_moneyness) - 2))

    # Bilinear interpolation weights
    d1, d2 = target_dte[i0], target_dte[i0 + 1]
    m1, m2 = target_moneyness[j0], target_moneyness[j0 + 1]
    wd = (dte - d1) / (d2 - d1 + 1e-10)
    wm = (m - m1) / (m2 - m1 + 1e-10)

    iv = (smooth_surf[i0,     j0    ] * (1 - wd) * (1 - wm) +
          smooth_surf[i0 + 1, j0    ] * wd       * (1 - wm) +
          smooth_surf[i0,     j0 + 1] * (1 - wd) * wm       +
          smooth_surf[i0 + 1, j0 + 1] * wd       * wm       )
    return float(np.clip(iv, 0.01, 3.0))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — RUN ALL 6 PRICERS
# ─────────────────────────────────────────────────────────────────────────────

def price_all(df: pd.DataFrame, mlp: MLPPricer, vae: VAE, lstm_sigma: float,
              smooth_surf, target_dte, target_moneyness, device) -> pd.DataFrame:
    """
    For each option in df, compute prices using all 6 methods.
    Returns the df augmented with price and error columns.
    """
    print("\n" + "=" * 65)
    print("  STEP 3 — Pricing all options with 6 methods")
    print("=" * 65)

    n = len(df)
    cols_price = ["bs_price", "crr_price", "mc_price", "lstm_bs_price",
                  "mlp_price", "vae_bs_price"]
    cols_time  = ["bs_ms", "crr_ms", "mc_ms", "lstm_bs_ms", "mlp_ms", "vae_bs_ms"]

    for c in cols_price + cols_time:
        df[c] = np.nan

    # Pre-batch MLP inference for speed
    S_arr     = df["S"].values
    K_arr     = df["K"].values
    T_arr     = df["T"].values
    r_arr     = df["r"].values
    sigma_arr = df["sigma"].values

    scale_factors = S_arr / 100.0
    S_scaled      = np.full(n, 100.0)
    K_scaled      = K_arr / scale_factors

    mlp_inputs = torch.tensor(
        np.stack([S_scaled, K_scaled, T_arr, r_arr, sigma_arr], axis=1),
        dtype=torch.float32
    ).to(device)
    t0_mlp = time.perf_counter()
    with torch.no_grad():
        mlp_raw = mlp(mlp_inputs).cpu().numpy().flatten()
    mlp_ms_each = (time.perf_counter() - t0_mlp) * 1000 / n
    mlp_prices = mlp_raw * scale_factors

    for idx, row in df.iterrows():
        S        = row["S"]
        K        = row["K"]
        T        = row["T"]
        r        = row["r"]
        sigma    = row["sigma"]
        moneyness = row["moneyness"]

        if (idx + 1) % 50 == 0:
            print(f"  Pricing option {idx+1}/{n}...")

        # ── 1. Black-Scholes ───────────────────────────────────────────
        t0 = time.perf_counter()
        bs_c, _ = black_scholes(S, K, T, r, sigma)
        df.at[idx, "bs_price"] = bs_c
        df.at[idx, "bs_ms"]    = (time.perf_counter() - t0) * 1000

        # ── 2. CRR Binomial (N=200) ────────────────────────────────────
        t0 = time.perf_counter()
        try:
            crr_res = price_option(S, K, T, r, sigma, N=200,
                                   option_type="call", exercise="european")
            df.at[idx, "crr_price"] = crr_res["price"]
        except Exception:
            df.at[idx, "crr_price"] = bs_c  # fallback
        df.at[idx, "crr_ms"] = (time.perf_counter() - t0) * 1000

        # ── 3. Monte Carlo (100k paths) ────────────────────────────────
        t0 = time.perf_counter()
        mc_res = mc_price(S, K, T, r, sigma, n_paths=100_000, seed=idx)
        df.at[idx, "mc_price"] = mc_res["price"]
        df.at[idx, "mc_ms"]    = (time.perf_counter() - t0) * 1000

        # ── 4. LSTM-BS (LSTM predicted sigma → BS) ─────────────────────
        t0 = time.perf_counter()
        lstm_c, _ = black_scholes(S, K, T, r, lstm_sigma)
        df.at[idx, "lstm_bs_price"] = lstm_c
        df.at[idx, "lstm_bs_ms"]    = (time.perf_counter() - t0) * 1000

        # ── 5. MLP (already batched above) ────────────────────────────
        df.at[idx, "mlp_price"] = mlp_prices[idx]
        df.at[idx, "mlp_ms"]    = mlp_ms_each

        # ── 6. VAE-interpolated IV → BS ───────────────────────────────
        t0 = time.perf_counter()
        vae_sigma = vae_iv_for_option(smooth_surf, target_dte, target_moneyness,
                                      T, moneyness)
        vae_c, _ = black_scholes(S, K, T, r, vae_sigma)
        df.at[idx, "vae_bs_price"] = vae_c
        df.at[idx, "vae_bs_ms"]    = (time.perf_counter() - t0) * 1000

    # Compute absolute errors
    for col in cols_price:
        err_col = col.replace("_price", "_err")
        df[err_col] = np.abs(df[col] - df["market_price"])

    df.to_csv("results_detailed.csv", index=False)
    print(f"\n  Detailed results saved → results_detailed.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────────────────

METHODS = [
    ("Black-Scholes",      "bs"),
    ("CRR Binomial N=200", "crr"),
    ("Monte Carlo 100k",   "mc"),
    ("LSTM-BS Hybrid",     "lstm_bs"),
    ("MLP Pricer",         "mlp"),
    ("VAE-IV → BS",        "vae_bs"),
]

BEST_USE = {
    "Black-Scholes"     : "Fast European baseline",
    "CRR Binomial N=200": "American / early-exercise",
    "Monte Carlo 100k"  : "Exotic / path-dependent",
    "LSTM-BS Hybrid"    : "Time-series vol forecasting",
    "MLP Pricer"        : "Ultra-fast batch pricing",
    "VAE-IV → BS"       : "IV surface interpolation",
}


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame with per-method metrics."""
    print("\n" + "=" * 65)
    print("  STEP 4 — Computing metrics")
    print("=" * 65)

    bucket_order = ["Deep ITM", "ITM", "ATM", "OTM", "Deep OTM"]
    records = []

    for label, key in METHODS:
        err_col   = f"{key}_err"
        price_col = f"{key}_price"
        ms_col    = f"{key}_ms"

        errors = df[err_col].dropna()
        prices = df[price_col].dropna()
        market = df.loc[prices.index, "market_price"]

        mae   = float(errors.mean())
        rmse  = float(np.sqrt((errors ** 2).mean()))
        pct5  = float((errors / (market + 1e-8) <= 0.05).mean() * 100)
        speed = float(df[ms_col].mean())

        # ATM / OTM buckets
        atm_mask = df["bucket"] == "ATM"
        otm_mask = df["bucket"].isin(["OTM", "Deep OTM"])

        mae_atm = float(df.loc[atm_mask, err_col].mean()) if atm_mask.sum() > 0 else np.nan
        mae_otm = float(df.loc[otm_mask, err_col].mean()) if otm_mask.sum() > 0 else np.nan

        record = {
            "Method"      : label,
            "MAE (all)"   : mae,
            "RMSE"        : rmse,
            "% within 5%" : pct5,
            "MAE ATM"     : mae_atm,
            "MAE OTM"     : mae_otm,
            "Speed ms/opt": speed,
            "Best Use"    : BEST_USE[label],
        }
        records.append(record)

        print(f"  {label:<22}  MAE={mae:.4f}  RMSE={rmse:.4f}"
              f"  %5%={pct5:.1f}%  ATM={mae_atm:.4f}  OTM={mae_otm:.4f}"
              f"  {speed:.2f} ms/opt")

    summary = pd.DataFrame(records)
    summary.to_csv("comparison_table.csv", index=False)
    print("\n  Summary saved → comparison_table.csv")

    # Print bucket-level winner table
    print("\n  Bucket winners:")
    for b in bucket_order:
        mask = df["bucket"] == b
        if mask.sum() == 0:
            continue
        best_label = min(METHODS,
                         key=lambda m: df.loc[mask, f"{m[1]}_err"].mean())[0]
        print(f"    {b:<12}: {best_label}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PUBLICATION-READY COMPARISON TABLE FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_table(summary: pd.DataFrame, df: pd.DataFrame):
    """
    Generate a premium dark-mode publication figure with:
      - Top panel: Styled metrics table (green highlights for winners)
      - Bottom panel: MAE bar chart per method × moneyness bucket
    """
    print("\n" + "=" * 65)
    print("  STEP 5 — Generating publication figure")
    print("=" * 65)

    plt.style.use("dark_background")
    BG      = "#0d1117"
    PANEL   = "#161b22"
    BORDER  = "#30363d"
    TEXT    = "#e6edf3"
    SUBTEXT = "#8b949e"
    GREEN   = "#3fb950"
    GOLD    = "#d29922"

    METHOD_COLORS = [
        "#38bdf8",   # BS — sky blue
        "#f472b6",   # CRR — pink
        "#fb923c",   # MC  — orange
        "#a78bfa",   # LSTM-BS — violet
        "#34d399",   # MLP — emerald
        "#facc15",   # VAE — gold
    ]

    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[1.2, 1],
                            hspace=0.38, wspace=0.30,
                            left=0.03, right=0.97,
                            top=0.91, bottom=0.06)

    # ── Panel 1: Metrics table (full width) ──────────────────────────────────
    ax_tbl = fig.add_subplot(gs[0, :])
    ax_tbl.set_facecolor(PANEL)
    ax_tbl.axis("off")

    numeric_cols = ["MAE (all)", "RMSE", "% within 5%", "MAE ATM", "MAE OTM", "Speed ms/opt"]
    display_cols = ["Method", "MAE (all)", "RMSE", "% within 5%", "MAE ATM",
                    "MAE OTM", "Speed ms/opt", "Best Use"]

    cell_data = []
    for _, row in summary.iterrows():
        cell_data.append([
            row["Method"],
            f"{row['MAE (all)']:.4f}",
            f"{row['RMSE']:.4f}",
            f"{row['% within 5%']:.1f}%",
            f"{row['MAE ATM']:.4f}" if not np.isnan(row["MAE ATM"]) else "—",
            f"{row['MAE OTM']:.4f}" if not np.isnan(row["MAE OTM"]) else "—",
            f"{row['Speed ms/opt']:.3f}",
            row["Best Use"],
        ])

    col_widths = [0.16, 0.09, 0.09, 0.10, 0.09, 0.09, 0.12, 0.22]

    tbl = ax_tbl.table(
        cellText=cell_data,
        colLabels=display_cols,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)

    # Style header
    for j, _ in enumerate(display_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#21262d")
        cell.set_text_props(color=TEXT, fontweight="bold", fontsize=10)
        cell.set_edgecolor(BORDER)
        cell.set_height(0.14)

    # Find winning row index per numeric column
    winners = {}
    for col in numeric_cols:
        col_vals = summary[col].values
        if col == "% within 5%":         # higher is better
            winners[col] = int(np.argmax(col_vals))
        elif col == "Speed ms/opt":       # lower is better
            winners[col] = int(np.argmin(col_vals))
        else:                             # lower is better (MAE, RMSE)
            winners[col] = int(np.argmin(col_vals))

    col_to_idx = {c: i for i, c in enumerate(display_cols)}

    for i, (_, row_data) in enumerate(summary.iterrows()):
        row_color = "#1c2128" if i % 2 == 0 else PANEL
        for j, col_name in enumerate(display_cols):
            cell = tbl[i + 1, j]
            cell.set_facecolor(row_color)
            cell.set_edgecolor(BORDER)
            cell.set_height(0.13)
            is_winner = (col_name in winners and winners[col_name] == i)
            cell.set_text_props(
                color=GREEN if is_winner else TEXT,
                fontweight="bold" if is_winner else "normal",
            )
            # Method column: colour dot
            if j == 0:
                cell.set_text_props(color=METHOD_COLORS[i], fontweight="bold")

    ax_tbl.set_title(
        "Table 2 — Head-to-Head Comparison of All 6 Pricing Methods  "
        f"(n = {len(df)} SPY options, live market data)",
        fontsize=13, color=TEXT, fontweight="bold", pad=14, loc="left"
    )

    # ── Panel 2: MAE bar chart by method ─────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_bar.set_facecolor(PANEL)

    methods_short = ["BS", "CRR", "MC", "LSTM-BS", "MLP", "VAE-BS"]
    maes = summary["MAE (all)"].values
    bars = ax_bar.bar(methods_short, maes, color=METHOD_COLORS,
                      edgecolor=BORDER, linewidth=0.8, width=0.6)

    # Annotate bars
    for bar, v in zip(bars, maes):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:.4f}", ha="center", va="bottom",
                    color=TEXT, fontsize=9, fontweight="bold")

    ax_bar.set_title("Overall MAE vs Market Price", color=TEXT,
                     fontweight="bold", fontsize=11, pad=10)
    ax_bar.set_ylabel("Mean Absolute Error ($)", color=SUBTEXT, fontsize=9)
    ax_bar.tick_params(colors=SUBTEXT, labelsize=9)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor(BORDER)
    ax_bar.grid(True, axis="y", alpha=0.12, color="white", linewidth=0.5)
    ax_bar.set_facecolor(PANEL)

    # ── Panel 3: % within 5% of market ───────────────────────────────────────
    ax_pct = fig.add_subplot(gs[1, 1])
    ax_pct.set_facecolor(PANEL)

    pcts = summary["% within 5%"].values
    bars2 = ax_pct.bar(methods_short, pcts, color=METHOD_COLORS,
                       edgecolor=BORDER, linewidth=0.8, width=0.6)

    for bar, v in zip(bars2, pcts):
        ax_pct.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom",
                    color=TEXT, fontsize=9, fontweight="bold")

    ax_pct.set_title("% of Options Priced Within 5% of Market", color=TEXT,
                     fontweight="bold", fontsize=11, pad=10)
    ax_pct.set_ylabel("% of Options", color=SUBTEXT, fontsize=9)
    ax_pct.set_ylim(0, 110)
    ax_pct.tick_params(colors=SUBTEXT, labelsize=9)
    for sp in ax_pct.spines.values():
        sp.set_edgecolor(BORDER)
    ax_pct.grid(True, axis="y", alpha=0.12, color="white", linewidth=0.5)
    ax_pct.set_facecolor(PANEL)

    # ── Super-title ───────────────────────────────────────────────────────────
    fig.suptitle(
        "Options Pricing Engine — Definitive Method Comparison",
        fontsize=16, color=TEXT, fontweight="bold", y=0.97
    )

    plt.savefig("comparison_table.png", dpi=180, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    print("  Figure saved → comparison_table.png")

    # Also save HTML table
    styled_html = summary[display_cols[:-1]].to_html(index=False, float_format="%.4f")
    with open("comparison_table.html", "w") as f:
        f.write(f"""<!DOCTYPE html><html><head>
<style>
body {{ background:#0d1117; color:#e6edf3; font-family:Inter,sans-serif; padding:40px; }}
table {{ border-collapse:collapse; width:100%; }}
th {{ background:#21262d; padding:10px 14px; text-align:center; border:1px solid #30363d; }}
td {{ padding:8px 14px; text-align:center; border:1px solid #30363d; }}
tr:nth-child(even) {{ background:#1c2128; }}
</style></head><body>
<h2>Options Pricing Engine — Method Comparison (SPY Live Data)</h2>
{styled_html}
</body></html>""")
    print("  HTML table saved → comparison_table.html")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  OPTIONS PRICING ENGINE — DEFINITIVE COMPARISON EXPERIMENT")
    print("=" * 65)

    device = torch.device(
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"  Device: {device}\n")

    # ── 1. Fetch data ────────────────────────────────────────────────────────
    df = fetch_spy_options(target_count=200)
    S  = df["S"].iloc[0]

    # ── 2. Load / train models ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 2 — Loading / training models")
    print("=" * 65)

    mlp = load_mlp(device)
    vae = load_vae(device)
    lstm_model, lstm_sigma = train_lstm_quick(device)

    # Build VAE-smoothed IV surface from live data
    print("  Building VAE-smoothed IV surface from live options data...")
    smooth_surf, target_dte, target_moneyness = build_vae_iv_surface(df, vae, S)
    print(f"  Smooth surface range: [{smooth_surf.min():.3f}, {smooth_surf.max():.3f}] ✓")

    # ── 3. Price all options ─────────────────────────────────────────────────
    df = price_all(df, mlp, vae, lstm_sigma,
                   smooth_surf, target_dte, target_moneyness, device)

    # ── 4. Compute metrics ───────────────────────────────────────────────────
    summary = compute_metrics(df)

    # ── 5. Plot ──────────────────────────────────────────────────────────────
    plot_comparison_table(summary, df)

    # ── 6. Print final summary ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EXPERIMENT COMPLETE")
    print("=" * 65)
    print("  Output files:")
    print("    test_options.csv       — held-out market data")
    print("    results_detailed.csv   — per-option predictions")
    print("    comparison_table.csv   — aggregate metrics")
    print("    comparison_table.png   — publication figure")
    print("    comparison_table.html  — interactive HTML table")
    print("\n  Top-line results:")
    best_mae = summary.loc[summary["MAE (all)"].idxmin()]
    best_pct = summary.loc[summary["% within 5%"].idxmax()]
    print(f"    Lowest  MAE    : {best_mae['Method']}  ({best_mae['MAE (all)']:.4f})")
    print(f"    Best %5% match : {best_pct['Method']}  ({best_pct['% within 5%']:.1f}%)")
    print(f"    Fastest method : {summary.loc[summary['Speed ms/opt'].idxmin(), 'Method']}")
    print("=" * 65 + "\n")

    return summary, df


if __name__ == "__main__":
    summary, df = main()
