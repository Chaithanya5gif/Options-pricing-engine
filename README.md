# Options Pricing Engine

A research-grade options pricing engine implemented from scratch in Python, covering two complementary models: the analytical **Black-Scholes** closed-form solution and the **Cox-Ross-Rubinstein (CRR) Binomial Tree** lattice model.

---

## Table of Contents

1. [Black-Scholes Model](#1-black-scholes-model)
2. [CRR Binomial Tree Model](#2-crr-binomial-tree-model)
3. [Speed Benchmark](#3-speed-benchmark--accuracy-vs-speed-tradeoff)
4. [Visualizations](#4-visualizations)
5. [Features](#5-features)
6. [Usage](#6-usage)

---

## 1. Black-Scholes Model

### Mathematical Derivation

The Black-Scholes model assumes the underlying asset's price $S$ follows **Geometric Brownian Motion**:

$$
dS = \mu S \, dt + \sigma S \, dW
$$

where $\mu$ is the drift, $\sigma$ is the annualised volatility, and $dW$ is a Wiener process increment.

### The Black-Scholes PDE

By constructing a risk-neutral, self-financing delta-hedged portfolio and applying **Itô's Lemma** to the option value $V(S,t)$, all market-risk terms cancel. The no-arbitrage condition yields the Black-Scholes PDE:

$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0
$$

### Boundary Conditions

For a European Call $C(S,t)$ with strike $K$ and expiry $T$:

$$
C(S,T) = \max(S - K,\; 0)
$$

### Closed-Form Solution

$$
C(S,t) = S \cdot N(d_1) - K e^{-r(T-t)} \cdot N(d_2)
$$

$$
d_1 = \frac{\ln(S/K) + \bigl(r + \tfrac{\sigma^2}{2}\bigr)(T-t)}{\sigma \sqrt{T-t}}, \qquad d_2 = d_1 - \sigma\sqrt{T-t}
$$

For a European Put by put-call parity:

$$
P(S,t) = K e^{-r(T-t)} \cdot N(-d_2) - S \cdot N(-d_1)
$$

---

## 2. CRR Binomial Tree Model

### Motivation

Black-Scholes cannot price **American options** because early exercise creates a free-boundary problem with no closed-form solution. The CRR lattice model solves this by discretising time and pricing via **backward induction**, checking at every node whether early exercise is optimal.

### CRR Parameter Derivation

The continuous GBM is matched to a discrete binomial lattice over $N$ time steps of length $\Delta t = T/N$.

**Up and down factors** are chosen so the lattice volatility matches $\sigma$:

$$
u = e^{\sigma\sqrt{\Delta t}}, \qquad d = \frac{1}{u} = e^{-\sigma\sqrt{\Delta t}}
$$

**Risk-neutral probability** $p$ is derived by matching the expected return to the risk-free rate:

$$
e^{r\Delta t} = p \cdot u + (1-p) \cdot d
\quad\Longrightarrow\quad
p = \frac{e^{r\Delta t} - d}{u - d}
$$

For the lattice to be arbitrage-free, we require $0 < p < 1$.

### Stock Price Lattice (Forward Pass)

Each node $(i,j)$ at time step $i$ with $j$ up-moves has stock price:

$$
S_{i,j} = S_0 \cdot u^j \cdot d^{i-j}, \qquad 0 \le j \le i \le N
$$

This is computed **in one vectorised NumPy operation** — no nested loops.

### Backward Induction (European)

Starting from terminal payoffs at $t = T$:

$$
V_{N,j} = \max(S_{N,j} - K,\; 0) \quad \text{(call)}
$$

Each preceding node is the discounted risk-neutral expectation:

$$
V_{i,j} = e^{-r\Delta t}\bigl[p\cdot V_{i+1,\,j+1} + (1-p)\cdot V_{i+1,\,j}\bigr]
$$

### Early Exercise Check (American)

At every node, the American price adds one comparison:

$$
V_{i,j}^{\text{Am}} = \max\!\bigl(\underbrace{S_{i,j} - K}_{\text{intrinsic}},\; \underbrace{V_{i,j}^{\text{cont}}}_{\text{continuation}}\bigr)
$$

This single line is the **entire difference** between European and American pricing. Black-Scholes cannot do this; the lattice can.

### Convergence Property

As $N \to \infty$, the CRR binomial price converges to the Black-Scholes price for European options:

$$
\lim_{N\to\infty} C_{\text{CRR}}(N) = C_{\text{BS}}
$$

The characteristic **odd/even oscillation** (visible in the convergence plot below) arises because the strike $K$ alternately falls on or between lattice nodes. By $N = 500$, the error is $< 10^{-4}$.

![CRR Convergence Plot](crr_convergence.png)

---

## 3. Speed Benchmark — Accuracy vs Speed Tradeoff

The benchmark measures pricing time for $N = 10, 25, 50, 100, 200, 300, 500, 750, 1000$ steps.

| N | Time (ms) | BS Error (call) |
|------:|----------:|----------------:|
| 10 | ~0.1 | ~0.05 |
| 100 | ~0.5 | ~0.001 |
| 500 | ~10 | ~0.0001 |
| 1000 | ~50 | ~0.00003 |

**Complexity**: The CRR tree is $O(N^2)$ in both time and memory (visible on the log-log plot). The practical sweet spot is **N = 200–300**, giving 4+ decimal-place accuracy in under 5 ms.

![CRR Speed Benchmark](crr_benchmark.png)

---

## 4. Visualizations

| File | Description |
|------|-------------|
| `greeks_dashboard.png` | Dashboard of all 5 Greeks (Δ, Γ, ν, Θ, ρ) vs spot/vol/rate |
| `crr_convergence.png` | CRR binomial price vs N (10 → 500), converging to BS |
| `crr_benchmark.png` | Speed vs N on linear and log-log axes + pricing error decay |
| `volatility_smile.png` | Implied Volatility vs Strike Price from market data |
| `volatility_surface_3d.png` | 3D IV surface across strikes and expiries |

---

## 5. Features

- **Black-Scholes Pricing**: Closed-form Call/Put prices for European options
- **Greeks**: Δ (Delta), Γ (Gamma), ν (Vega), Θ (Theta), ρ (Rho) — all derived analytically
- **Implied Volatility**: Newton-Raphson with Brent's method fallback for guaranteed convergence
- **CRR Binomial Tree**: Vectorised NumPy lattice for both European and American options
- **Convergence Analysis**: Plots binomial price vs N against BS analytical benchmark
- **Speed Benchmark**: Wall-clock timing with O(N²) complexity analysis
- **Volatility Smile & Surface**: Real market data via `yfinance` for 2D smile and 3D surface

---

## 6. Usage

```bash
# Set up environment
python3 -m venv .venv && source .venv/bin/activate
pip install matplotlib numpy scipy yfinance

# Black-Scholes pricing + Greeks dashboard
python3 black_scholes.py

# CRR Binomial Tree (European + American) + convergence plot
python3 crr_binomial_tree.py

# Speed benchmark (time vs N, accuracy vs N)
python3 benchmark_crr.py

# Unit tests for Greeks
python3 -m unittest test_greeks.py -v
```

### Quick API example

```python
from crr_binomial_tree import price_option

# Price an American put (N=200 steps)
result = price_option(S0=100, K=100, T=1.0, r=0.05, sigma=0.20,
                      N=200, option_type="put", exercise="american")
print(f"American Put  : {result['price']:.4f}")
print(f"Delta         : {result['delta']:.4f}")
print(f"u={result['u']:.4f}  d={result['d']:.4f}  p={result['p']:.4f}")
```
