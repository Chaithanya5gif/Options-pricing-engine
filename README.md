# Options Pricing Engine 📈







---

## Overview

This repository contains the complete implementation, training pipelines, and evaluation notebooks for the paper **"A Comparative Study of Numerical Methods for European Option Pricing"** (SSRN, 2026). The project implements and benchmarks seven distinct pricing methodologies:

| Method | Type | Key Feature | Best For |
|--------|------|-------------|----------|
| **Black-Scholes** | Analytical | Closed-form, constant volatility | Ultra-fast European baseline |
| **CRR Binomial Tree** | Numerical | Discrete-time lattice, early exercise | American options |
| **Monte Carlo (GBM)** | Simulation | Path-dependent, variance reduction | Exotic derivatives |
| **Heston Model** | Stochastic Vol | Per-option Nelder-Mead calibration | Smile-consistent pricing |
| **MLP Pricer** | Deep Learning | Log-moneyness normalization | Batch risk calculations |
| **LSTM-BS Hybrid** | Deep Learning | 30-day volatility forecasting | Dynamic regime adaptation |
| **VAE-IV → BS** | Deep Learning | β-VAE implied volatility surfaces | IV surface interpolation |

### Key Results

The dynamically calibrated **Heston model** achieves the highest precision with a **MAE of 0.141** (95% CI: [0.11, 0.17]), accurately pricing **65% of options within 5%** of market mid-prices. The **LSTM-BS hybrid** achieves a competitive MAE of 0.448, outperforming constant-volatility baselines by incorporating dynamic volatility forecasts.

---

## Paper

**A Comparative Study of Numerical Methods for European Option Pricing**  
*Chaithanya L* — Independent Researcher, June 2026

&gt; **Abstract:** Option pricing accuracy is paramount for effective risk management and derivative trading. This paper investigates the efficacy of seven distinct option pricing methods benchmarked against a live dataset of 200 SPY options collected in June 2026. The evaluated methods span classical frameworks (Black-Scholes, CRR binomial trees, Monte Carlo with variance reduction), stochastic volatility (the Heston model with per-option Nelder-Mead calibration), and modern deep learning architectures (MLP trained on log-moneyness, LSTM-BS hybrid, and a Variational Autoencoder for implied volatility surfaces). Results indicate that the dynamically calibrated Heston model achieves the highest precision with a Mean Absolute Error (MAE) of 0.141 (95% CI: [0.11, 0.17]), accurately pricing 65% of options within 5% of market midprices. The LSTM-BS hybrid achieves a competitive MAE of 0.448, outperforming constant-volatility baselines by incorporating dynamic volatility forecasts. These findings demonstrate that while deep learning provides computational and forecasting advantages, stochastic volatility models remain the optimal framework for pricing smile-consistent options, and Monte Carlo frameworks retain definitive superiority for path-dependent derivatives.

📄 **[Read the Paper on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6955158)**  
📄 **[Download PDF](options_paper_2026.pdf)**  
📝 **[LaTeX Source](options_paper_2026.tex)**  
📊 **[Interactive Dashboard](https://options-pricing-engine.streamlit.app)** *(if deployed)*

**Keywords:** Option pricing, Black-Scholes, Heston model, Monte Carlo simulation, deep learning, LSTM, variational autoencoder, volatility smile, Diebold-Mariano test

---

## Mathematical Foundations

### Black-Scholes Model

The classic Black-Scholes PDE:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0$$

Closed-form solution for a European Call:

$$C(S, t) = S N(d_1) - K e^{-r(T-t)} N(d_2)$$

where

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}, \quad d_2 = d_1 - \sigma\sqrt{T-t}$$

### Monte Carlo Simulation (GBM)

Under the risk-neutral measure, the terminal asset price is simulated as:

$$S_T = S_0 \exp\left[ \left( r - \frac{\sigma^2}{2} \right)T + \sigma \sqrt{T} Z \right], \quad Z \sim \mathcal{N}(0, 1)$$

We employ **antithetic variates** and **control variates** to reduce variance.

### Heston Stochastic Volatility Model

The Heston model generalizes GBM by allowing volatility to evolve stochastically:

$$dS_t = r S_t dt + \sqrt{v_t} S_t dW_{1,t}$$
$$dv_t = \kappa(\theta - v_t) dt + \sigma_v \sqrt{v_t} dW_{2,t}$$

where $\text{corr}(dW_{1,t}, dW_{2,t}) = \rho$ and the Feller condition $2\kappa\theta &gt; \sigma_v^2$ ensures positivity of variance.

The model admits a closed-form solution via characteristic functions and Fourier inversion:

$$P_j = \frac{1}{2} + \frac{1}{\pi} \int_0^\infty \text{Re}\left[ \frac{e^{-i\phi \ln K} f_j(\phi)}{i\phi} \right] d\phi$$

---

## Empirical Results

Benchmarked on 200 live SPY options (June 2026), 80/20 stratified train-test split.

| Method | MAE (SE) | RMSE (SE) | % within 5% | MAE ATM | MAE OTM | Speed (ms/opt) |
|--------|----------|-----------|-------------|---------|---------|----------------|
| **Heston MC (Calib.)** | **0.141 (0.012)** | **0.351 (0.028)** | **65.0%** | 0.090 | 0.018 | 18.84 |
| LSTM-BS Hybrid | 0.448 (0.034) | 0.734 (0.061) | 41.5% | 1.152 | 0.028 | 0.01 |
| Monte Carlo (100k) | 0.855 (0.042) | 1.266 (0.073) | 46.5% | 0.607 | 0.016 | 1.31 |
| Black-Scholes | 0.861 (0.043) | 1.265 (0.072) | 45.0% | 0.603 | 0.016 | 0.01 |
| CRR Binomial (N=200) | 0.861 (0.043) | 1.265 (0.072) | 44.5% | 0.605 | 0.017 | 14.44 |
| VAE-IV → BS | 2.093 (0.156) | 2.907 (0.218) | 30.5% | 1.381 | 3.896 | 0.10 |
| MLP Pricer | 3.374 (0.248) | 4.057 (0.312) | 27.0% | 5.184 | 1.209 | 0.04 |

*Standard errors computed via bootstrap resampling (1,000 iterations). 95% CIs are bias-corrected and accelerated (BCa).*

### Key Findings

1. **Stochastic volatility dominates.** The dynamically calibrated Heston model achieves state-of-the-art accuracy (MAE 0.141) due to its capacity to capture the empirical volatility smile through per-option parameter optimization.

2. **Classical methods are statistically equivalent.** Black-Scholes, CRR, and Monte Carlo exhibit statistical parity (confirmed via Diebold-Mariano test, p &gt; 0.45), as their pricing errors stem inherently from the constant-volatility assumption.

3. **Deep learning offers targeted advantages.** LSTM networks excel at time-series volatility forecasting (48% error reduction vs. baselines), VAEs generate arbitrage-free volatility surfaces for interpolation, and MLPs enable rapid batch evaluation despite limited accuracy in low-data regimes.

4. **Monte Carlo remains essential for exotics.** The simulation framework is the definitive numerical method for path-dependent derivatives such as barrier options, handling knock-out conditions natively without structural modifications.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Chaithanya5gif/Options-pricing-engine.git
cd Options-pricing-engine

# Install dependencies
pip install -r requirements.txt
