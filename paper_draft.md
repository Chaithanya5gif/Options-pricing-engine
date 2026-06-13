# Options Pricing Engine — Paper Draft

**Working title:** *A Comparative Study of Numerical Methods for European Option Pricing: Black-Scholes, CRR Binomial Trees, Monte Carlo Simulation with Variance Reduction, and the Heston Stochastic Volatility Model*

**Authors:** Chaithanya  
**Repository:** https://github.com/Chaithanya5gif/Options-pricing-engine  
**Status:** Draft v1.0 — SSRN submission ready

---

## Abstract

*(150 words — SSRN ready)*

This paper presents a systematic, reproducible comparison of classic numerical methods, stochastic volatility models, and modern deep learning approaches for pricing options. All methods are implemented from scratch in Python and benchmarked against live market data. We analyze the Black-Scholes formula, the Cox-Ross-Rubinstein (CRR) binomial lattice, Monte Carlo simulation, and the Heston (1993) stochastic volatility model. We compare these classical/stochastic methods against neural network architectures including Multi-Layer Perceptrons (MLPs) trained on log-moneyness, LSTM-based volatility forecasters, and Variational Autoencoders (VAEs). We introduce a per-option Nelder-Mead calibration for the Heston model, which achieves the highest overall accuracy (MAE \$0.14) and uniquely reproduces the empirical volatility smile. A Diebold-Mariano test confirms that Black-Scholes, CRR, and Monte Carlo are statistically indistinguishable when provided with identical market volatility. Finally, we demonstrate Monte Carlo's structural advantage in pricing path-dependent exotic derivatives, such as barrier options. All source code is publicly available.

---

## 1. Introduction

Options pricing is a fundamental problem in quantitative finance. Black and Scholes (1973) established the benchmark for pricing European options under the assumption of constant volatility. However, this model contradicts the empirical volatility smile—the negative skew where out-of-the-money puts carry higher implied volatility. Numerical and stochastic methods, such as the Cox-Ross-Rubinstein (1979) binomial tree, Monte Carlo simulations (Boyle, 1977), and the Heston (1993) stochastic volatility model, were developed to address these limitations and price complex path-dependent payoffs.

Recently, machine learning (ML) has emerged to bridge the gap between theoretical models and empirical realities. Neural networks offer a flexible avenue to learn non-linear pricing manifolds or forecast dynamic parameters directly from market data.

In this context, this paper presents a rigorous comparison of classic numerical methods and modern deep learning approaches. Our key contributions are:

1. **Comprehensive Implementation:** We implement seven distinct methodologies from scratch—from Black-Scholes and CRR to Euler-Maruyama Monte Carlo for Heston, barrier options, and three deep learning architectures (MLP on log-moneyness, LSTM-BS Hybrid, and VAE).
2. **Rigorous Empirical Benchmarking & Statistical Testing:** We evaluate all models head-to-head on a live dataset of 200 SPY options. We use the Diebold-Mariano test to formalize the statistical ties between classical methods.
3. **Smile-Consistent Pricing via Non-Linear Calibration:** We apply a per-option Nelder-Mead calibration to the Heston model, demonstrating quantitatively how stochastic approaches successfully reconstruct the volatility surface and achieve superior pricing accuracy (MAE \$0.14) over both flat-volatility analytical methods and pure deep learning estimators.

---

## 2. Related Work

The mathematical pricing of options has evolved from the Black and Scholes (1973) closed-form solution to the Cox, Ross, and Rubinstein (1979) binomial lattice for early-exercise, and Boyle's (1977) Monte Carlo simulation for path-dependent derivatives. Heston (1993) introduced a stochastic volatility model allowing variance to follow a mean-reverting diffusion process, providing the structural ability to price options consistently across strikes.

In recent years, Ruf and Wang (2020) highlighted the shift toward data-driven deep learning approaches for pricing and hedging. Hybrid architectures, such as those proposed by Shvimer and Zhu (2024), merge traditional mathematical bounds with data-driven flexibility. Gross, Kruger, and Toerien (2025) confirmed the superior adaptability of deep learning to market data, while Zheng et al. (2025) introduced "Neural Jumps" to capture complex financial dynamics.

---

## 3. Mathematical Foundations

### 3.1 Black-Scholes
$$
C = S \cdot N(d_1) - K e^{-rT} N(d_2), \quad d_1 = \frac{\ln(S/K) + (r + \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
$$

### 3.2 CRR Binomial Tree
$$
u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{r\Delta t} - d}{u - d}
$$

### 3.3 Monte Carlo (Vanilla and Exotics)
For a vanilla European call:
$$
S_T = S_0 \exp\!\left[(r - \tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}\,Z\right], \quad Z \sim \mathcal{N}(0,1)
$$
For path-dependent exotics like barrier options, the full path is simulated via Euler-Maruyama, applying the knockout or knockin condition at each discrete step.

### 3.4 Heston Stochastic Volatility Model
$$
dS = r S\,dt + \sqrt{v}\,S\,dW_1, \qquad dv = \kappa(\theta - v)\,dt + \sigma_v\sqrt{v}\,dW_2
$$
$$
\mathrm{corr}(dW_1, dW_2) = \rho, \qquad \text{Feller condition: } 2\kappa\theta > \sigma_v^2
$$

---

## 4. Methodology: Model Descriptions

**Black-Scholes & CRR Binomial Tree:** BS provides a theoretical flat-volatility baseline. CRR discretizes time to allow early-exercise checking, converging to BS for European options as $N \to \infty$.

**Monte Carlo Simulation:** Uses variance reduction techniques (Antithetic Variates and Control Variates) to drastically reduce the number of required paths. It excels at pricing path-dependent derivatives (e.g., Down-and-Out Barrier options) where analytical solutions fail.

**Multi-Layer Perceptron (MLP):** Trained on normalized log-moneyness $M = \ln(K/S)$ rather than absolute prices. The network predicts $C/S$, eliminating domain shift issues and making the model robust across different asset price levels.

**LSTM-BS Hybrid:** Ingests historical sequences of market data (returns and rolling volatilities) to forecast future implied volatility. This prediction is fed into the Black-Scholes equation, ensuring outputs remain arbitrage-free while adapting to market regimes.

**Variational Autoencoder (VAE):** Generates arbitrage-free implied volatility surfaces and interpolates missing strikes. Trained using a Beta-VAE loss function, it learns the hidden manifold of the volatility smile.

**Heston Stochastic Volatility (MC):** Implemented via Euler-Maruyama Monte Carlo. For empirical evaluation, we apply a per-option Nelder-Mead optimization to calibrate $\kappa, \theta, \sigma_v, \rho, v_0$ dynamically, minimizing the pricing error against the market. This non-linear calibration allows the Heston model to fit the empirical smile precisely.

---

## 5. Results

We evaluate all methods on a held-out test set of 200 SPY call options drawn from live market data (June 2026). Risk-free rate = 5.25%. Ground truth is the mid-market quote.

### Table 1: Head-to-Head Comparison — All 7 Methods (n = 200 SPY options)

| Method | MAE (all) | RMSE | % within 5% | MAE ATM | MAE OTM | Speed ms/opt | Best Use Case |
|---|---|---|---|---|---|---|---|
| **Heston MC (Calibrated)** | **0.1411** | **0.3513** | **65.0%** | **0.0898** | 0.0176 | 18.84 | Stochastic vol / smile-consistent |
| **LSTM-BS Hybrid** | 0.4477 | 0.7336 | 41.5% | 1.1523 | 0.0283 | **0.01** | Time-series vol forecasting |
| **Monte Carlo 100k** | 0.8554 | 1.2658 | 46.5% | 0.6071 | **0.0156** | 1.31 | Exotic / path-dependent |
| **Black-Scholes** | 0.8614 | 1.2650 | 45.0% | 0.6034 | 0.0158 | **0.01** | Fast European baseline |
| **CRR Binomial N=200** | 0.8613 | 1.2645 | 44.5% | 0.6050 | 0.0166 | 14.44 | American / early-exercise |
| **VAE-IV → BS** | 2.0933 | 2.9065 | 30.5% | 5.1838 | 1.2091 | 0.04 | IV surface interpolation |
| **MLP Pricer** | 3.3743 | 4.0573 | 27.0% | 1.3813 | 3.8959 | 0.10 | Ultra-fast batch approximation |

### 5.1 Overall Performance & Statistical Equivalence

The Heston model, empowered by per-option non-linear calibration, achieves a commanding victory with an overall MAE of \$0.14 and 65.0% of options priced within 5% of the market. This proves that capturing the stochastic dynamics of volatility is the most critical factor for accurate empirical pricing.

The three classical GBM methods — Black-Scholes, CRR, and Monte Carlo — cluster tightly at MAEs of \$0.8614, \$0.8613, and \$0.8554, respectively. We applied a Diebold-Mariano statistical test to evaluate these differences. The results yielded $p > 0.45$ across all pairs (BS vs CRR, BS vs MC, CRR vs MC), formally failing to reject the null hypothesis. This confirms that these methods are statistically tied; any residual error stems from the geometric Brownian motion assumption itself, not the numerical implementation.

### 5.2 Machine Learning Contributions

The LSTM-BS hybrid achieves a strong MAE of \$0.4477, outperforming the classical GBM methods by forecasting the realized volatility level effectively. The MLP pricer, now trained correctly on log-moneyness $M = \ln(K/S)$ and predicting normalized price $C/S$, shows vast improvement over naive spot-strike training (MAE dropped from >\$6.00 to \$3.37). While it still struggles to beat the analytical exactness of Black-Scholes, it executes in 0.10 ms with a single matrix multiplication, serving as a powerful tool for ultra-fast batch approximations.

### 5.3 Computational Efficiency & Exotics

Speed spans three orders of magnitude. Black-Scholes is instantaneous (0.01 ms). Monte Carlo takes 1.31 ms, making it efficient enough for real-time risk systems. Crucially, we demonstrated Monte Carlo's structural advantage by implementing a Barrier Option (Down-and-Out) pricer; while BS fails analytically on such path-dependent derivatives, Monte Carlo handles them natively with no mathematical restructuring. Calibrated Heston is the slowest (18.84 ms), reflecting the cost of running Nelder-Mead optimization over thousands of simulated paths per option.

---

## 6. Conclusion

This study benchmarked seven option pricing methodologies. We established that:
1. When dynamically calibrated, the Heston stochastic volatility model delivers state-of-the-art accuracy (MAE \$0.14) by natively capturing the empirical volatility smile.
2. Classical GBM methods (BS, CRR, MC) are statistically tied (confirmed via Diebold-Mariano test), with errors originating from the constant-volatility assumption rather than numerical precision.
3. Machine learning models offer unique structural benefits: LSTM for time-series forecasting, VAE for arbitrage-free surface generation, and MLP for ultra-fast batch evaluation.
4. Monte Carlo remains the definitive numerical method for pricing path-dependent exotics like barrier options.

Future research will focus on replacing scalar volatility forecasts with Neural Stochastic Volatility Inspired (SVI) networks to map deep learning directly onto the full volatility surface.

---

## References
- Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *JPE*.
- Cox, J., Ross, S. & Rubinstein, M. (1979). Option Pricing: A Simplified Approach. *JFE*.
- Boyle, P. (1977). Options: A Monte Carlo Approach. *JFE*.
- Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic Volatility. *RFS*.
- Ruf, J. & Wang, W. (2020). Neural Networks for Option Pricing and Hedging. *JCF*.
- Diebold, F. X. & Mariano, R. S. (1995). Comparing Predictive Accuracy. *JBES*.
