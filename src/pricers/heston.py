"""
Heston Stochastic Volatility Model — Monte Carlo Pricer
========================================================

Model dynamics (Heston, 1993):
  dS = r·S·dt + √v·S·dW₁
  dv = κ(θ−v)dt + σᵥ·√v·dW₂,   corr(dW₁, dW₂) = ρ

Discretisation : Euler-Maruyama with full truncation  [v ← max(v, 0)]
Variance reduction : Antithetic variates (applied jointly to W₁ & W₂)

Default parameters (June 11 schedule):
  κ=2.0,  θ=0.04,  σᵥ=0.3,  ρ=−0.7,  v₀=0.04
  N=252 steps (daily),  50,000 paths

Results produced:
  [1] Heston MC price vs Black-Scholes    — sanity check at ρ→0, σᵥ→0
  [2] Heston IV smile vs flat BS smile    — KEY RESULT  →  heston_smile.png
  [3] 200-option SPY test-set evaluation  — MAE / RMSE metrics for paper

Reference:
  Heston, S.L. (1993). A Closed-Form Solution for Options with Stochastic
  Volatility with Applications to Bond and Currency Options. The Review of
  Financial Studies, 6(2), 327–343.
"""

import time
import math
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from black_scholes import black_scholes, implied_vol


# ── 1. Core Pricer ────────────────────────────────────────────────────────────

def heston_mc_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    v0: float    = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma_v: float = 0.3,
    rho: float   = -0.7,
    n_paths: int = 50_000,
    n_steps: int = 252,
    option_type: str = "call",
    seed: int = 42,
) -> dict:
    """
    Heston MC pricer via Euler-Maruyama with antithetic variates.

    Discretisation (dt = T/N):
      v_{t+dt}  = max(v_t + κ(θ−v_t)dt + σᵥ·√(v_t·dt)·Z₂, 0)    [full truncation]
      S_{t+dt}  = S_t · exp((r − ½·v_t)·dt + √(max(v_t,0)·dt)·Z₁) [log-Euler]

    Correlated Brownians:
      Z₁ ~ N(0,1),   Z₂ = ρ·Z₁ + √(1−ρ²)·Z₃,   Z₃ ~ N(0,1)

    Antithetic variates: run (Z₁, Z₂) and (−Z₁, −Z₂) simultaneously,
    average paired payoffs before taking the grand mean.

    Parameters
    ----------
    S0       : float  Current spot price
    K        : float  Strike
    T        : float  Time to expiry (years)
    r        : float  Risk-free rate (annualised, continuous)
    v0       : float  Initial variance  (≈ σ₀²)
    kappa    : float  Mean-reversion speed
    theta    : float  Long-run variance (≈ σ_∞²)
    sigma_v  : float  Volatility of variance (vol-of-vol)
    rho      : float  Spot-vol correlation  (−1, 1)
    n_paths  : int    Number of MC paths  (antithetic → 2×n_paths sims)
    n_steps  : int    Time steps per path
    option_type : str "call" or "put"
    seed     : int    RNG seed for reproducibility

    Returns
    -------
    dict  price, std_error, elapsed_ms, n_paths, n_steps
    """
    rng = np.random.default_rng(seed)
    t0  = time.perf_counter()

    dt   = T / n_steps
    sqrt_rho2 = math.sqrt(1.0 - rho * rho)

    # Allocate price & variance paths  (shape: n_paths)
    S  = np.full(n_paths, S0, dtype=np.float64)
    v  = np.full(n_paths, v0, dtype=np.float64)
    Sa = np.full(n_paths, S0, dtype=np.float64)   # antithetic
    va = np.full(n_paths, v0, dtype=np.float64)

    for _ in range(n_steps):
        # Draw two independent standard normals  (shape: n_paths)
        Z1 = rng.standard_normal(n_paths)
        Z3 = rng.standard_normal(n_paths)
        Z2 = rho * Z1 + sqrt_rho2 * Z3           # correlated with Z1

        # Current vols  (truncated to ≥0 for diffusion term)
        v_pos  = np.maximum(v,  0.0)
        va_pos = np.maximum(va, 0.0)
        sv     = np.sqrt(v_pos  * dt)
        sva    = np.sqrt(va_pos * dt)

        # ── Update variance (Euler-Maruyama, full truncation) ──
        v_new  = v  + kappa * (theta - v)  * dt + sigma_v * sv  * Z2
        va_new = va + kappa * (theta - va) * dt + sigma_v * sva * (-Z2)
        v  = np.maximum(v_new,  0.0)
        va = np.maximum(va_new, 0.0)

        # ── Update log-spot (log-Euler to guarantee S > 0) ──
        S  = S  * np.exp((r - 0.5 * v_pos)  * dt + sv  * Z1)
        Sa = Sa * np.exp((r - 0.5 * va_pos) * dt + sva * (-Z1))

    # ── Payoffs ──
    if option_type == "call":
        pay  = np.maximum(S  - K, 0.0)
        paya = np.maximum(Sa - K, 0.0)
    else:
        pay  = np.maximum(K - S,  0.0)
        paya = np.maximum(K - Sa, 0.0)

    disc   = math.exp(-r * T)
    paired = 0.5 * (pay + paya)               # antithetic estimator
    price  = disc * paired.mean()
    se     = disc * paired.std(ddof=1) / math.sqrt(n_paths)

    return {
        "price"      : price,
        "std_error"  : se,
        "elapsed_ms" : (time.perf_counter() - t0) * 1000,
        "n_paths"    : n_paths,
        "n_steps"    : n_steps,
        "method"     : "Heston MC",
    }


# ── 2. Implied Vol Smile ───────────────────────────────────────────────────────

def compute_heston_smile(
    S0: float = 100.0,
    T: float  = 1.0,
    r: float  = 0.05,
    v0: float    = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma_v: float = 0.3,
    rho: float   = -0.7,
    n_strikes: int = 10,
    n_paths: int   = 50_000,
    n_steps: int   = 252,
) -> tuple[list, list, list]:
    """
    Price calls at n_strikes evenly spaced between 80% and 120% moneyness.
    Back out the Black-Scholes implied vol for each Heston price.
    Also compute the flat BS IV smile (constant = √θ = 20%).

    Returns
    -------
    strikes      : list[float]
    heston_ivs   : list[float | None]   (None if IV extraction fails)
    bs_flat_ivs  : list[float]
    """
    moneyness  = np.linspace(0.80, 1.20, n_strikes)
    strikes    = (S0 * moneyness).tolist()
    bs_sigma   = math.sqrt(theta)              # flat BS: σ = √θ = 20%

    heston_ivs  = []
    bs_flat_ivs = []

    for i, K in enumerate(strikes):
        res = heston_mc_price(
            S0, K, T, r, v0, kappa, theta, sigma_v, rho,
            n_paths=n_paths, n_steps=n_steps,
            option_type="call", seed=i + 100,
        )
        h_price = res["price"]

        # Intrinsic value check — IV extraction requires price > intrinsic
        intrinsic = max(S0 * math.exp(-0.0 * T) - K * math.exp(-r * T), 0.0)
        if h_price <= intrinsic + 1e-8:
            heston_ivs.append(None)
        else:
            iv = implied_vol(h_price, S0, K, T, r, option_type="call")
            heston_ivs.append(iv)

        bs_call, _ = black_scholes(S0, K, T, r, bs_sigma)
        bs_flat_ivs.append(bs_sigma)

        print(
            f"  K={K:6.1f}  m={moneyness[i]:.2f}  "
            f"Heston={h_price:.4f}  IV={f'{heston_ivs[-1]:.4f}' if heston_ivs[-1] is not None else 'N/A'}"
        )

    return strikes, heston_ivs, bs_flat_ivs


# ── 3. Smile Plot ─────────────────────────────────────────────────────────────

def plot_heston_smile(
    S0: float = 100.0,
    T: float  = 1.0,
    r: float  = 0.05,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma_v: float = 0.3,
    rho: float   = -0.7,
    v0: float    = 0.04,
    n_paths: int = 50_000,
    n_steps: int = 252,
    save_path: str = "heston_smile.png",
):
    """
    Two-panel plot:
      Left  — IV smile: Heston (curved) vs BS (flat) — KEY RESULT
      Right — Heston price surface across strikes (price vs moneyness)
    """
    print("\n[→] Computing Heston smile (10 strikes, 50k paths each) …")
    strikes, heston_ivs, bs_ivs = compute_heston_smile(
        S0, T, r, v0, kappa, theta, sigma_v, rho,
        n_strikes=10, n_paths=n_paths, n_steps=n_steps,
    )

    # Filter out None IVs for plotting
    valid_idx    = [i for i, iv in enumerate(heston_ivs) if iv is not None]
    valid_K      = [strikes[i] for i in valid_idx]
    valid_hiv    = [heston_ivs[i] for i in valid_idx]
    valid_bsiv   = [bs_ivs[i] for i in valid_idx]
    moneyness_v  = [k / S0 for k in valid_K]

    # Also compute Heston prices for the right panel
    print("\n[→] Computing full price curve …")
    heston_prices = []
    bs_prices     = []
    bs_sigma = math.sqrt(theta)
    for K, iv in zip(valid_K, valid_hiv):
        h_res = heston_mc_price(
            S0, K, T, r, v0, kappa, theta, sigma_v, rho,
            n_paths=n_paths, n_steps=n_steps, seed=int(K),
        )
        heston_prices.append(h_res["price"])
        bs_c, _ = black_scholes(S0, K, T, r, bs_sigma)
        bs_prices.append(bs_c)

    # ── Styling ──────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(1, 2, wspace=0.36, left=0.07, right=0.97,
                            top=0.87, bottom=0.12)

    GRID_KW = dict(alpha=0.12, color="white", linewidth=0.5)
    C_HESTON = "#f472b6"    # pink  — Heston curved smile
    C_BS     = "#38bdf8"    # blue  — flat BS
    C_ATM    = "#facc15"    # gold  — ATM line
    C_PRICE  = "#34d399"    # green — Heston price

    def _style(ax, title, xlabel, ylabel, legend_loc="best"):
        ax.set_facecolor("#161b22")
        ax.set_title(title, fontsize=11.5, color="white", fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, fontsize=9.5, color="#94a3b8")
        ax.set_ylabel(ylabel, fontsize=9.5, color="#94a3b8")
        ax.tick_params(colors="#94a3b8", labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.grid(True, **GRID_KW)
        ax.legend(fontsize=9, framealpha=0.3, loc=legend_loc,
                  labelcolor="white", facecolor="#21262d", edgecolor="#30363d")

    # ── Panel 1: IV Smile ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    # Heston smile curve (interpolated for visual smoothness)
    ax1.plot(moneyness_v, [iv * 100 for iv in valid_hiv],
             "o-", color=C_HESTON, lw=2.5, ms=7, zorder=5,
             label=f"Heston MC  (κ={kappa}, θ={theta}, σᵥ={sigma_v}, ρ={rho})")

    # Flat BS line
    ax1.axhline(math.sqrt(theta) * 100, color=C_BS, lw=2.0, ls="--",
                label=f"Black-Scholes flat  (σ = √θ = {math.sqrt(theta)*100:.0f}%)")

    # ATM marker
    ax1.axvline(1.0, color=C_ATM, lw=1.2, ls=":", alpha=0.7, label="ATM (K/S = 1)")

    # Shade the smile premium region
    bs_line = [math.sqrt(theta) * 100] * len(moneyness_v)
    ax1.fill_between(moneyness_v, [iv * 100 for iv in valid_hiv], bs_line,
                     alpha=0.12, color=C_HESTON)

    ax1.set_xlim(0.77, 1.23)
    ax1.set_ylim(
        min(iv * 100 for iv in valid_hiv) * 0.92,
        max(iv * 100 for iv in valid_hiv) * 1.08,
    )
    _style(ax1,
           "Heston Implied Volatility Smile  vs  Flat BS Smile\n"
           "KEY RESULT: Heston reproduces real-market skew",
           "Moneyness  K/S", "Implied Volatility  (%)",
           legend_loc="upper center")

    # Annotate left/right wings
    if valid_hiv[0] is not None:
        ax1.annotate(
            f"OTM put wing\n(left skew from ρ<0)",
            xy=(moneyness_v[0], valid_hiv[0] * 100),
            xytext=(moneyness_v[0] + 0.04, valid_hiv[0] * 100 + 1.2),
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.2),
            color="#94a3b8", fontsize=8,
        )

    # ── Panel 2: Price Curve ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.plot(moneyness_v, heston_prices, "o-", color=C_HESTON, lw=2.5, ms=7,
             label="Heston MC price")
    ax2.plot(moneyness_v, bs_prices,     "s--", color=C_BS,     lw=2.0, ms=6,
             label=f"BS price  (σ={math.sqrt(theta)*100:.0f}%)")

    # Highlight price difference
    ax2.fill_between(moneyness_v, heston_prices, bs_prices,
                     alpha=0.15, color=C_PRICE, label="Price diff (Heston − BS)")
    ax2.axvline(1.0, color=C_ATM, lw=1.2, ls=":", alpha=0.7)

    _style(ax2,
           "Heston vs BS Call Prices Across Strikes",
           "Moneyness  K/S", "Call Price ($)",
           legend_loc="upper right")

    # ── Super-title ───────────────────────────────────────────────────────────
    fig.suptitle(
        f"Heston Stochastic Volatility Model  —  MC with Euler-Maruyama  "
        f"[S={S0}, T={T}yr, r={r:.0%}, v₀={v0}, N={n_steps} steps, {n_paths:,} paths]",
        fontsize=12, color="white", fontweight="bold", y=0.97,
    )

    plt.savefig(save_path, dpi=180, facecolor=fig.get_facecolor())
    print(f"\n[✓] Heston smile plot saved → {save_path}")
    return strikes, heston_ivs, bs_ivs


# ── 4. SPY Test-Set Evaluation ────────────────────────────────────────────────

def evaluate_heston_on_test_set(
    csv_path: str = "results_detailed.csv",
    kappa: float  = 2.0,
    sigma_v: float = 0.3,
    rho: float    = -0.7,
    n_paths: int  = 10_000,   # fewer paths for speed — 200 options × ~2s each
    n_steps: int  = 63,       # ~quarterly steps (fast, good enough for eval)
    seed: int     = 42,
) -> dict:
    """
    Evaluate Heston on the 200-option SPY test set used in paper Table 2.

    Per-option Heston parameters:
      θ  = σ_market²        (long-run var = market IV² for that option)
      v₀ = θ                (start at long-run var)
      κ, σᵥ, ρ              (global, fixed — same for all options)

    This mirrors how BS/CRR/MC use per-option market IV directly.

    Reads columns: Strike, Market, IV, T (from results_detailed.csv).
    Falls back gracefully if CSV is unavailable.
    """
    import os

    if not os.path.exists(csv_path):
        print(f"[!] {csv_path} not found — skipping SPY evaluation.")
        return {}

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[!] results_detailed.csv is empty.")
        return {}

    # Detect columns
    all_cols = list(rows[0].keys())
    print(f"[i] CSV columns: {all_cols[:10]} …")

    # Resolve column names flexibly
    def _col(candidates):
        for c in candidates:
            if c in all_cols:
                return c
        return None

    col_k      = _col(["Strike", "strike", "K"])
    col_market = _col(["Market", "market_price", "mid_price", "price"])
    col_iv     = _col(["IV", "iv", "impliedVolatility", "sigma"])
    col_t      = _col(["T", "t", "T_years", "tte"])
    col_S      = _col(["S", "spot", "Spot", "S0"])
    col_mono   = _col(["Moneyness", "moneyness", "bucket", "Bucket"])

    if not all([col_k, col_market, col_iv]):
        print(f"[!] Cannot find required columns. Found: {all_cols}")
        return {}

    S0_default = 736.70   # SPY spot on test date
    r          = 0.0525   # 5.25%

    print(f"\n[→] Evaluating Heston on {len(rows)} options "
          f"(n_paths={n_paths:,}, n_steps={n_steps}) …")

    errors, sq_errors, within_5pct = [], [], []
    bucket_errors = {"Deep ITM": [], "ITM": [], "ATM": [], "OTM": [], "Deep OTM": []}
    timings = []

    for i, row in enumerate(rows):
        try:
            K      = float(row[col_k])
            mkt    = float(row[col_market])
            sigma  = float(row[col_iv])
            T_opt  = float(row[col_t]) if col_t else 0.25
            S0     = float(row[col_S]) if col_S else S0_default
            mono   = float(row[col_mono]) if col_mono and row.get(col_mono, "").replace(".","").isdigit() else K / S0

            if mkt <= 0 or sigma <= 0.001 or T_opt <= 0:
                continue

            theta  = sigma ** 2    # long-run var = market IV²
            v0_opt = theta

            res = heston_mc_price(
                S0, K, T_opt, r,
                v0=v0_opt, kappa=kappa, theta=theta,
                sigma_v=sigma_v, rho=rho,
                n_paths=n_paths, n_steps=n_steps,
                option_type="call", seed=seed + i,
            )
            timings.append(res["elapsed_ms"])
            h_price = res["price"]

            err = abs(h_price - mkt)
            errors.append(err)
            sq_errors.append(err ** 2)
            within_5pct.append(1 if err / max(mkt, 0.01) <= 0.05 else 0)

            # Bucket assignment
            if mono < 0.90:
                bkt = "Deep ITM"
            elif mono < 0.975:
                bkt = "ITM"
            elif mono <= 1.025:
                bkt = "ATM"
            elif mono <= 1.10:
                bkt = "OTM"
            else:
                bkt = "Deep OTM"
            bucket_errors[bkt].append(err)

            if (i + 1) % 20 == 0:
                print(f"  [{i+1:3d}/{len(rows)}]  running MAE = {np.mean(errors):.4f}")

        except (ValueError, TypeError, ZeroDivisionError):
            continue

    if not errors:
        print("[!] No valid rows processed.")
        return {}

    mae  = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    pct5 = float(np.mean(within_5pct) * 100)
    spd  = float(np.mean(timings))

    mae_atm = float(np.mean(bucket_errors["ATM"])) if bucket_errors["ATM"] else float("nan")
    mae_otm = float(np.mean(bucket_errors["OTM"])) if bucket_errors["OTM"] else float("nan")

    print("\n" + "=" * 62)
    print("  HESTON MC — SPY TEST SET RESULTS")
    print("=" * 62)
    print(f"  N options evaluated : {len(errors)}")
    print(f"  MAE (all)           : {mae:.4f}")
    print(f"  RMSE                : {rmse:.4f}")
    print(f"  % within 5%         : {pct5:.1f}%")
    print(f"  MAE ATM             : {mae_atm:.4f}")
    print(f"  MAE OTM             : {mae_otm:.4f}")
    print(f"  Speed               : {spd:.2f} ms/option")
    print("=" * 62)

    for bkt, errs in bucket_errors.items():
        if errs:
            print(f"  {bkt:10s}  MAE = {np.mean(errs):.4f}  (n={len(errs)})")

    return {
        "MAE"       : mae,
        "RMSE"      : rmse,
        "pct5"      : pct5,
        "MAE_ATM"   : mae_atm,
        "MAE_OTM"   : mae_otm,
        "speed_ms"  : spd,
        "n"         : len(errors),
        "bucket_errors": bucket_errors,
    }


# ── 5. Main — Run All Checks ──────────────────────────────────────────────────

if __name__ == "__main__":

    # ── [0] Model parameters (from June 11 schedule) ──────────────────────────
    S0      = 100.0
    K_atm   = 100.0
    T       = 1.0
    r       = 0.05
    KAPPA   = 2.0
    THETA   = 0.04     # long-run variance  → σ_∞ = 20%
    SIGMA_V = 0.3
    RHO     = -0.7
    V0      = 0.04     # initial variance  → σ₀ = 20%
    N_PATHS = 50_000
    N_STEPS = 252

    BS_SIGMA = math.sqrt(THETA)    # 20% — equivalent flat vol for BS comparison
    bs_ref, _ = black_scholes(S0, K_atm, T, r, BS_SIGMA)

    print("=" * 70)
    print("  HESTON MC PRICER — VERIFICATION  (Euler-Maruyama, Antithetic)")
    print("=" * 70)
    print(f"  Parameters: S={S0}, K={K_atm}, T={T}yr, r={r:.0%}")
    print(f"  Heston:  κ={KAPPA}, θ={THETA}, σᵥ={SIGMA_V}, ρ={RHO}, v₀={V0}")
    print(f"  Paths:   {N_PATHS:,}  |  Steps: {N_STEPS}")
    print(f"  BS ref (σ=√θ=20%): {bs_ref:.5f}")
    print()

    # ── [1] Sanity check — ρ→0, σᵥ small → Heston ≈ BS ──────────────────────
    print("[1] Sanity check: ρ=0, σᵥ=0.001  (Heston → GBM → BS)")
    res_sanity = heston_mc_price(
        S0, K_atm, T, r,
        v0=THETA, kappa=KAPPA, theta=THETA, sigma_v=0.001, rho=0.0,
        n_paths=N_PATHS, n_steps=N_STEPS, seed=0,
    )
    err_sanity = abs(res_sanity["price"] - bs_ref)
    print(f"    Heston price = {res_sanity['price']:.5f}")
    print(f"    BS ref       = {bs_ref:.5f}")
    print(f"    |Error|      = {err_sanity:.5f}  {'✓ PASS' if err_sanity < 0.10 else '✗ FAIL'}")
    print(f"    Std error    = {res_sanity['std_error']:.5f}")
    print(f"    Time         = {res_sanity['elapsed_ms']:.1f} ms")
    print()

    # ── [2] Full Heston price (with skew) ─────────────────────────────────────
    print(f"[2] Full Heston MC (κ={KAPPA}, θ={THETA}, σᵥ={SIGMA_V}, ρ={RHO})")
    res_full = heston_mc_price(
        S0, K_atm, T, r,
        v0=V0, kappa=KAPPA, theta=THETA, sigma_v=SIGMA_V, rho=RHO,
        n_paths=N_PATHS, n_steps=N_STEPS, seed=42,
    )
    print(f"    Heston price = {res_full['price']:.5f}")
    print(f"    BS ref       = {bs_ref:.5f}")
    print(f"    Diff (skew premium) = {res_full['price'] - bs_ref:+.5f}")
    print(f"    95% CI: {res_full['price']:.5f} ± {1.96*res_full['std_error']:.5f}")
    print(f"    Time         = {res_full['elapsed_ms']:.1f} ms")
    print()

    # ── [3] IV Smile plot  ← KEY RESULT ───────────────────────────────────────
    print("[3] Generating Heston IV smile vs flat BS smile …")
    plot_heston_smile(
        S0=S0, T=T, r=r,
        kappa=KAPPA, theta=THETA, sigma_v=SIGMA_V, rho=RHO, v0=V0,
        n_paths=N_PATHS, n_steps=N_STEPS,
        save_path="heston_smile.png",
    )
    print()

    # ── [4] SPY test-set evaluation ───────────────────────────────────────────
    print("[4] SPY test-set evaluation …")
    metrics = evaluate_heston_on_test_set(
        csv_path="results_detailed.csv",
        kappa=KAPPA, sigma_v=SIGMA_V, rho=RHO,
        n_paths=10_000, n_steps=63,
    )
    print()

    # ── [5] Summary for paper ─────────────────────────────────────────────────
    print("=" * 70)
    print("  PAPER TABLE 2 — HESTON ROW (copy into comparison_table.csv)")
    print("=" * 70)
    if metrics:
        print(f"  Method         : Heston MC")
        print(f"  MAE (all)      : {metrics['MAE']:.4f}")
        print(f"  RMSE           : {metrics['RMSE']:.4f}")
        print(f"  % within 5%    : {metrics['pct5']:.1f}%")
        print(f"  MAE ATM        : {metrics['MAE_ATM']:.4f}")
        print(f"  MAE OTM        : {metrics['MAE_OTM']:.4f}")
        print(f"  Speed ms/opt   : {metrics['speed_ms']:.2f}")
    else:
        print("  [!] SPY evaluation not available — results_detailed.csv missing.")
    print("=" * 70)
    print("\n[✓] All Heston checks complete.")
