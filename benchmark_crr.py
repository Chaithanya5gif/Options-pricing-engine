"""
Benchmark — CRR Binomial Tree speed vs number of steps N
=========================================================

Measures wall-clock time for pricing a European call + American put
at N = 10, 25, 50, 100, 200, 300, 500, 750, 1000 steps.

Output
------
  • Console table  : N | time (ms) | EU Call price | AM Put price
  • crr_benchmark.png : dual-axis plot — time vs N and price vs N

Methodology relevance
---------------------
This plot exposes the accuracy-speed tradeoff:
  - N=100  converges to 4 decimal places of BS  →  < 5 ms
  - N=1000 adds ~3 more decimal places           →  ~500 ms
The "good enough" operating point is typically N=200–300.
"""

import time
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from crr_binomial_tree import price_option
from black_scholes import black_scholes


# ── Parameters ────────────────────────────────────────────────────────────────
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
REPEATS = 2          # average over this many runs per N to reduce jitter

N_VALUES = [10, 25, 50, 100, 150, 200, 300, 500]


# ── Run benchmark ─────────────────────────────────────────────────────────────
print("=" * 68)
print("  CRR BINOMIAL TREE — SPEED BENCHMARK")
print("=" * 68)
print(f"  Params : S={S0}, K={K}, T={T}yr, r={r:.0%}, σ={sigma:.0%}")
print(f"  Repeats: {REPEATS} runs per N (averaged)")
print("-" * 68)
print(f"{'N':>6}  {'Time (ms)':>10}  {'EU Call':>10}  {'AM Put':>10}  {'BS err (call)':>14}")
print("-" * 68)

bs_call, bs_put = black_scholes(S0, K, T, r, sigma)

times_ms   = []
eu_calls   = []
am_puts    = []
bs_errors  = []

for N in N_VALUES:
    # ── Time the pricing call (average over REPEATS)
    elapsed_total = 0.0
    eu_price = am_price = 0.0
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        eu_res = price_option(S0, K, T, r, sigma, N, "call", "european")
        am_res = price_option(S0, K, T, r, sigma, N, "put",  "american")
        elapsed_total += time.perf_counter() - t0
        eu_price = eu_res["price"]
        am_price = am_res["price"]

    avg_ms  = (elapsed_total / REPEATS) * 1000   # convert to ms
    err     = abs(eu_price - bs_call)

    times_ms.append(avg_ms)
    eu_calls.append(eu_price)
    am_puts.append(am_price)
    bs_errors.append(err)

    print(f"{N:>6}  {avg_ms:>9.2f}ms  {eu_price:>10.6f}  {am_price:>10.6f}  {err:>14.8f}")

print("-" * 68)
print(f"  Black-Scholes reference  Call={bs_call:.6f}  Put={bs_put:.6f}")
print("=" * 68)


# ── Plot ──────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#0d1117")

gs = gridspec.GridSpec(2, 2, hspace=0.44, wspace=0.36,
                       left=0.07, right=0.97, top=0.90, bottom=0.08)

GRID_KW = dict(alpha=0.12, color="white", linewidth=0.5)

def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor("#161b22")
    ax.set_title(title, fontsize=11, color="white", fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color="#94a3b8")
    ax.set_ylabel(ylabel, fontsize=9, color="#94a3b8")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.grid(True, **GRID_KW)
    ax.legend(fontsize=8, framealpha=0.25, loc="upper left",
              labelcolor="white", facecolor="#21262d")


# ── Panel 1: Time vs N (linear) ───────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(N_VALUES, times_ms, "o-", color="#38bdf8", lw=2, ms=6,
         label="Wall-clock time (ms)")
# Annotate key points
for n, t in zip(N_VALUES, times_ms):
    if n in (100, 500, 1000):
        ax1.annotate(f"{t:.1f} ms", (n, t),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=7, color="#94a3b8")
_style(ax1, "Computation Time vs N  (linear scale)",
       "Number of steps N", "Time per price call (ms)")


# ── Panel 2: Time vs N (log-log) to reveal O(N²) ─────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.loglog(N_VALUES, times_ms, "o-", color="#f472b6", lw=2, ms=6,
           label="Wall-clock time (ms)")

# Overlay ideal O(N²) reference line
n_arr   = np.array(N_VALUES, dtype=float)
scale   = times_ms[0] / (N_VALUES[0] ** 2)
on2     = scale * n_arr ** 2
ax2.loglog(N_VALUES, on2, "--", color="#facc15", lw=1.4, alpha=0.7,
           label="O(N²) reference")

_style(ax2, "Computation Time vs N  (log-log scale — O(N²) complexity)",
       "N (log scale)", "Time (ms, log scale)")


# ── Panel 3: EU Call price convergence + BS reference ─────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(N_VALUES, eu_calls, "o-", color="#34d399", lw=2, ms=6,
         label="CRR European Call")
ax3.axhline(bs_call, color="#facc15", lw=1.8, ls="--",
            label=f"BS Call = {bs_call:.4f}")
_style(ax3, "Price Accuracy vs N  (European Call)",
       "Number of steps N", "Call Price ($)")


# ── Panel 4: BS pricing error vs N ───────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.semilogy(N_VALUES, bs_errors, "o-", color="#fb923c", lw=2, ms=6,
             label="|CRR − BS|  (call)")
ax4.axhline(1e-4, color="#facc15", lw=1.4, ls=":",
            label="4 decimal place threshold (1e-4)")
ax4.axhline(1e-2, color="#f87171", lw=1.4, ls=":",
            label="2 decimal place threshold (1e-2)")
_style(ax4, "Pricing Error vs N  (|CRR − BS|, log scale)",
       "Number of steps N", "Absolute error (log scale)")


# ── Main title ────────────────────────────────────────────────────────────────
fig.suptitle(
    f"CRR Binomial Tree — Speed vs Accuracy Benchmark  "
    f"[S={S0}, K={K}, T={T}yr, r={r:.0%}, σ={sigma:.0%}]",
    fontsize=13, color="white", fontweight="bold", y=0.965,
)

save_path = "crr_benchmark.png"
plt.savefig(save_path, dpi=180, facecolor=fig.get_facecolor())
print(f"\n[✓] Benchmark plot saved → {save_path}")
