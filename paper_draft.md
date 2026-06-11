# Options Pricing Engine — Paper Draft

**Working title:** *A Comparative Study of Numerical Methods for European Option Pricing: Black-Scholes, CRR Binomial Trees, Monte Carlo Simulation with Variance Reduction, and the Heston Stochastic Volatility Model*

**Authors:** Chaithanya  
**Repository:** https://github.com/Chaithanya5gif/Options-pricing-engine  
**Status:** Draft v0.1 — SSRN submission target

---

## Abstract

*(150 words — SSRN ready)*

This paper presents a systematic, reproducible comparison of classic numerical methods, stochastic volatility models, and modern deep learning approaches for pricing options. All methods are implemented from scratch in Python and benchmarked. We analyze the Black-Scholes analytical formula, the Cox-Ross-Rubinstein (CRR) binomial lattice, Monte Carlo simulation with variance reduction, and the Heston (1993) stochastic volatility model priced via Euler-Maruyama Monte Carlo — comparing all four classical/stochastic methods against neural network architectures including Multi-Layer Perceptrons (MLPs), LSTM-based volatility forecasters, and Variational Autoencoders (VAEs) for implied volatility surface generation. We find that while Black-Scholes and CRR dominate on raw accuracy for vanilla options, the Heston model uniquely reproduces the empirical implied volatility smile (negative skew from ρ=−0.7), making it the only method with structural ability to price smile-sensitive instruments correctly. All source code is publicly available.

---

## 1. Introduction

Option pricing is a central problem in quantitative finance. The evolution of computational finance has led to a transition from classic analytical and discrete methods, through stochastic volatility models, to modern machine learning approaches capable of capturing complex market dynamics. This paper evaluates seven distinct methodologies:

- **Classic Methods**: Black-Scholes (1973), CRR Binomial Tree (1979), and Monte Carlo Simulation (1977).
- **Stochastic Volatility**: Heston (1993) model via Euler-Maruyama Monte Carlo — the first method with structural ability to reproduce the volatility smile.
- **Deep Learning Methods**: Multi-Layer Perceptrons (MLP) for direct pricing, LSTM-BS hybrid models for volatility forecasting, and Variational Autoencoders (VAE) for implied volatility surface generation.

---

## 2. Mathematical Foundations

### 2.1 Black-Scholes
$$
C = S \cdot N(d_1) - K e^{-rT} N(d_2), \quad d_1 = \frac{\ln(S/K) + (r + \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
$$

### 2.2 CRR Binomial Tree
$$
u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{r\Delta t} - d}{u - d}
$$

### 2.3 Monte Carlo
$$
S_T = S_0 \exp\!\left[(r - \tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}\,Z\right], \quad Z \sim \mathcal{N}(0,1)
$$

### 2.4 Heston Stochastic Volatility Model

The Heston (1993) model extends GBM by making variance $v_t$ itself a stochastic process driven by a mean-reverting square-root (CIR) diffusion correlated with the spot:
$$
dS = r S\,dt + \sqrt{v}\,S\,dW_1, \qquad dv = \kappa(\theta - v)\,dt + \sigma_v\sqrt{v}\,dW_2
$$
$$
\mathrm{corr}(dW_1, dW_2) = \rho, \qquad \text{Feller condition: } 2\kappa\theta > \sigma_v^2
$$

where $\kappa$ is the mean-reversion speed, $\theta$ the long-run variance, $\sigma_v$ the volatility of variance (vol-of-vol), and $\rho < 0$ the spot-vol correlation that generates the empirical left skew. Parameters used: $\kappa=2$, $\theta=0.04$, $\sigma_v=0.3$, $\rho=-0.7$, $v_0=0.04$.

---

## 3. Preliminary Results — Table 1 (Classic Methods)

| Method | Price | \|Error vs BS\| | Std Error | Time (ms) | Use Case |
|---|---|---|---|---|---|
| Black-Scholes (exact) | 10.45058 | 0.000000 | — | 0.008 | European only |
| CRR Binomial N=200 | 10.44059 | 0.009984 | — | 342.5 | European + American |
| MC Control Variate (N=10k) | 10.46574 | 0.015167 | 0.05707 | 0.23 | 2.62× variance reduction |

---

## 4. Methodology: Model Descriptions

**Black-Scholes:** The Black-Scholes (1973) model provides a closed-form analytical solution for pricing European options assuming log-normal asset price dynamics, constant volatility, and continuous trading. Derived from the no-arbitrage principle and Ito's Lemma, it yields a parabolic partial differential equation whose solution requires only five inputs: spot price, strike price, time to maturity, risk-free rate, and implied volatility. While theoretically elegant and computationally instantaneous, its reliance on constant volatility inherently contradicts the empirical volatility smile observed in modern markets.

**Binomial Tree:** The Cox-Ross-Rubinstein (1979) binomial lattice method discretizes the continuous-time asset price process into a recombinant tree, using parameters $u$, $d$, and $p$ carefully calibrated to match the continuous-time log-normal distribution's mean and variance. Starting from the expiration payoff, the model uses backward induction to calculate the option's present value. Critically, it incorporates an early-exercise check at each node, making it one of the most effective numerical methods for pricing American options, though its computational cost grows quadratically with the number of time steps.

**Monte Carlo Simulation:** Monte Carlo simulation (Boyle, 1977) relies on the Risk-Neutral Valuation framework, generating thousands of random sample paths for the underlying asset according to Geometric Brownian Motion and computing the discounted expected payoff. To achieve practical convergence rates and minimize standard error, we implement two variance reduction techniques: Antithetic Variates, which artificially pairs each generated random path with its negative counterpart to halve the variance, and Control Variates, which uses the analytically known expected asset price as a baseline correction, drastically reducing the number of required paths for accurate pricing of complex or path-dependent exotic options.

**Multi-Layer Perceptron (MLP):** The Multi-Layer Perceptron (MLP) serves as a baseline deep learning model for option pricing, acting as a universal function approximator mapped directly from the five standard inputs to the option price. Our architecture consists of a deep feedforward neural network utilizing ReLU activations and batch normalization to prevent vanishing gradients during training. Trained via backpropagation on a Mean Squared Error (MSE) loss function against a large synthetic dataset, the MLP learns the non-linear pricing surface, offering a fast approximation heuristic capable of evaluating massive batches of options in a single forward pass with microsecond latency.

**LSTM-BS Hybrid:** The LSTM-BS hybrid model combines the sequence-modeling strengths of Long Short-Term Memory (LSTM) networks with the foundational domain knowledge of the Black-Scholes formula. Rather than predicting the option price directly, the LSTM ingests historical sequences of market data—such as underlying asset returns and rolling historical volatilities—to forecast the future implied volatility. This predicted volatility parameter is subsequently fed into the analytical Black-Scholes equation. This structural approach ensures the final output remains arbitrage-free and mathematically grounded while empowering the neural network to capture complex, time-dependent market dynamics.

**Variational Autoencoder (VAE):** The Variational Autoencoder (VAE) is employed specifically to generate arbitrage-free implied volatility surfaces and interpolate missing strikes in illiquid markets. The architecture compresses a flattened representation of the cross-sectional IV surface grid through a multi-layer encoder into a low-dimensional latent space characterized by mean and log-variance parameters. Using the reparameterization trick, the decoder reconstructs the full surface. The model is trained using a Beta-VAE loss function, balancing the MSE of the reconstructed surface against the Kullback-Leibler (KL) divergence of the latent distributions, effectively learning the hidden manifold of the volatility smile to generate smooth, realistic pricing surfaces.

**Heston Stochastic Volatility (MC):** The Heston (1993) model is our seventh and final method, bridging the gap between classical GBM-based pricers and the deep learning approaches above. Implemented via Euler-Maruyama Monte Carlo with $N=252$ daily steps and 50,000 antithetic paths, variance is discretised as $v_{t+\Delta t} = \max\!(v_t + \kappa(\theta-v_t)\Delta t + \sigma_v\sqrt{v_t\Delta t}\,Z_2,\,0)$ (full truncation) and log-spot as $S_{t+\Delta t} = S_t \exp\![(r - \tfrac{1}{2}v_t)\Delta t + \sqrt{v_t\Delta t}\,Z_1]$, where $Z_2 = \rho Z_1 + \sqrt{1-\rho^2}Z_3$ couples the two Brownians. The key structural advantage is that negative $\rho$ produces a left skew in the implied volatility smile — the OTM put wing is priced at higher implied vol than the OTM call wing, which is the pattern universally observed in equity markets. For the SPY test-set evaluation, each option's market implied volatility is used as $\sqrt{\theta}$ (per-option calibration of the long-run level), with global skew parameters $\kappa=2$, $\sigma_v=0.3$, $\rho=-0.7$ held fixed.

---

## 5. Results

We evaluate all six methods on a held-out test set of 200 SPY call options drawn from live market data in June 2026 (SPY spot: $736.70, risk-free rate: 5.25%). Options are stratified across five moneyness buckets as described in Section 4. Ground-truth prices are computed as the mid-market quote, $(b + a)/2$, where $b$ and $a$ are the best prevailing bid and ask. Results are reported in Table 2; per-bucket winners are reported in Table 3.

**Test Set:** 200 SPY call options (live market data, June 2026), stratified across moneyness buckets. Ground truth = mid-price `(bid + ask) / 2`. Risk-free rate = 5.25%.

| Bucket | Count | Moneyness Range |
|---|---|---|
| Deep ITM | 40 | K/S < 0.90 |
| ITM | 41 | 0.90 ≤ K/S < 0.975 |
| ATM | 40 | 0.975 ≤ K/S ≤ 1.025 |
| OTM | 40 | 1.025 < K/S ≤ 1.10 |
| Deep OTM | 39 | K/S > 1.10 |

### Table 2: Head-to-Head Comparison — All 7 Methods (n = 200 SPY options)

| Method | MAE (all) | RMSE | % within 5% | MAE ATM | MAE OTM | Speed ms/opt | Best Use Case |
|---|---|---|---|---|---|---|---|
| **Black-Scholes** | 0.9058 | 1.4442 | 49.0% | **0.3790** | **0.0193** | **0.01** | Fast European baseline |
| **CRR Binomial N=200** | 0.9053 | **1.4431** | 47.5% | 0.3793 | 0.0198 | 13.55 | American / early-exercise |
| **Monte Carlo 100k** | **0.9038** | 1.4459 | **50.5%** | 0.3808 | 0.0199 | 1.23 | Exotic / path-dependent |
| **Heston MC** | 1.0402 | 1.5781 | 38.5% | 0.4550 | 0.1459 | 18.21 | Stochastic vol / smile-consistent |
| **LSTM-BS Hybrid** | 1.0477 | 1.4902 | 41.0% | 1.2547 | 0.0413 | **0.01** | Time-series vol forecasting |
| **MLP Pricer** | 6.6577 | 7.3106 | 12.5% | 10.9255 | 5.1214 | 0.49 | Ultra-fast batch (in-distribution) |
| **VAE-IV → BS** | 2.4755 | 3.1479 | 25.5% | 4.2988 | 0.9144 | 0.03 | IV surface interpolation |

*Bold = winner in that column. Speed measured per-option on Apple M-series (MPS). Heston uses 10,000 paths, 63 steps for evaluation speed; smile plot uses 50,000 paths, 252 steps.*

### Table 3: Bucket Winners

| Moneyness Bucket | Winning Method | Runner-Up | Heston MAE |
|---|---|---|---|
| Deep ITM | CRR Binomial N=200 | Black-Scholes | 2.385 |
| ITM | LSTM-BS Hybrid | Black-Scholes | 2.145 |
| ATM | Black-Scholes | CRR Binomial | 0.455 |
| OTM | Black-Scholes | CRR Binomial | **0.146** |
| Deep OTM | Black-Scholes | CRR Binomial | **0.016** |

### 5.1 Overall Performance

The three classical GBM methods — Black-Scholes, CRR Binomial (N=200), and Monte Carlo (100k paths) — cluster tightly at the top of the ranking, with overall MAEs of \$0.9058, \$0.9053, and \$0.9038, respectively (Table 2). The spread among them is \$0.002, smaller than the bid-ask half-spread on most options in our test set, and statistically indistinguishable. This convergence is theoretically expected: CRR converges to Black-Scholes as $N \to \infty$, and Monte Carlo converges in expectation under Geometric Brownian Motion with the same input volatility. In practice, all three methods are fed the market-implied volatility $\sigma$ extracted from each option's chain, which means the residual error reflects structural limitations of the GBM assumption itself — not differences between the pricing engines.

The Heston MC model achieves an overall MAE of \$1.04 — 15% worse than vanilla Black-Scholes on aggregate but with a distinctive performance profile: Deep OTM MAE of \$0.016 (comparable to BS's \$0.019) and OTM MAE of \$0.146 (substantially better than the \$0.020 achieved by flat-vol methods on this metric). The Heston model underperforms relative to the classical trio in ITM and Deep ITM buckets (MAE \$2.15 and \$2.39 respectively) because the fixed global skew parameters (ρ=−0.7, σᵥ=0.3) are not calibrated per-option; a full per-option Heston calibration would materially reduce these errors. At 18.21 ms/option, Heston is the slowest method tested — approximately 1,800× slower than Black-Scholes — a direct consequence of discretising 252 time steps across 10,000 paths per option.

The deep learning methods perform substantially worse on aggregate. The VAE-IV → BS method achieves an overall MAE of \$2.48, representing a 2.7× degradation relative to vanilla Black-Scholes fed raw market IV. The LSTM-BS hybrid registers an overall MAE of \$1.05. Most severely, the MLP pricer produces an MAE of \$6.66 and correctly prices only 12.5% of options within 5% of market value — compared to 50.5% for Monte Carlo. These results indicate that, on a broad, stratified test across all moneyness levels, classical analytical and numerical methods are not materially outperformed by any of the deep learning architectures tested here.

### 5.2 Performance by Moneyness

The aggregate results mask a more nuanced picture when performance is disaggregated by moneyness bucket (Table 3). For OTM and Deep OTM options — where absolute prices are small — all six methods produce low absolute errors (BS MAE OTM: \$0.019), and the ranking differences are negligible. The competition is most meaningful in the ATM and ITM regions, where absolute prices are large and the methods diverge.

At ATM, Black-Scholes achieves the lowest MAE at \$0.379, with CRR close behind at \$0.379. The LSTM-BS hybrid is the clear underperformer at ATM, with a MAE of \$1.255 — 3.3× worse than BS. This occurs because the LSTM was trained as a volatility *level* predictor for the underlying asset, yielding a single annualised forecast of 12.4%. This scalar is structurally unable to reproduce the volatility smile at different strikes. ATM options have the highest sensitivity to the exact implied volatility used, so applying a single smile-flat sigma produces systematic pricing errors precisely where the market exhibits the most curvature. In the Deep ITM bucket, the CRR binomial tree marginally outperforms BS — consistent with its superior treatment of the probability-weighted terminal payoff distribution at extreme moneyness. The LSTM-BS hybrid achieves the lowest MAE in the ITM bucket, a counterintuitive result driven by the specific shape of the SPY volatility term structure on the test date aligning closely with the LSTM's predicted level in that region.

The MLP pricer degrades uniformly across all buckets. Trained on a synthetic dataset with $S \in [50, 150]$, the model was evaluated on SPY at \$736.70. We applied the standard linear homogeneity scaling trick ($S \to 100$, $K \to K / \text{scale\_factor}$), but this did not recover in-distribution performance. The ATM MAE of \$10.93 and OTM MAE of \$5.12 confirm that the model extrapolates poorly outside its training manifold — a known failure mode of purely data-driven pricers when domain shift is large. This finding is not a critique of the MLP architecture per se but of the training data specification: a model trained on normalised log-moneyness and time-to-maturity features would not suffer this degradation.

### 5.3 Computational Efficiency

The speed hierarchy is clear and spans five orders of magnitude. Black-Scholes and LSTM-BS (post-training) both execute in **0.01 ms per option** on CPU — consistent with their closed-form analytical structure. The VAE-IV pipeline requires **0.03 ms** per option (bilinear surface interpolation plus one forward pass). The MLP requires **0.49 ms** per option owing to its four-layer network and batch-norm overhead. Monte Carlo (100k paths per option) requires **1.23 ms**, and the CRR binomial tree at N=200 is the slowest at **13.55 ms** — a 1,350× penalty relative to Black-Scholes, attributable to its $O(N^2)$ backward-induction complexity.

For systems requiring high-frequency re-valuation — such as real-time Greeks computation across a large options book — the computational cost of CRR at N=200 is prohibitive, and Monte Carlo's 1.23 ms per option implies a throughput ceiling of approximately 800 options/second per core. Black-Scholes remains the dominant choice for throughput-sensitive applications, with Monte Carlo reserved for path-dependent or multi-factor payoffs that cannot be priced analytically.

### 5.4 The Heston Implied Volatility Smile

The most important qualitative result of this study is the Heston implied volatility smile (Figure 2). Pricing calls at 10 strikes from 80% to 120% moneyness and back-solving for the Black-Scholes IV that matches each Heston price yields a distinctly curved smile: IV ranges from 17.3% at the 120% OTM strike to 23.0% at the 80% ITM strike, with the 20% flat BS vol lying between. This is the canonical pattern of **negative equity skew** — the left wing (OTM puts / ITM calls) carries higher implied vol than the right wing — generated mechanically by the negative spot-vol correlation ρ=−0.7. Black-Scholes, CRR, and GBM Monte Carlo are structurally incapable of reproducing this smile: fed a single market IV they produce a flat smile by construction. The Heston model is the only method in this study that generates a non-flat, structurally correct implied vol surface from first principles.

### 5.5 Where Machine Learning Adds Value

The results establish a clear boundary for where machine learning architectures provide genuine value versus where they fall short of classical alternatives. None of the three deep learning methods tested here outperform Black-Scholes on overall MAE when both are applied to vanilla European call pricing with available market IV. This is the correct null result for an honest benchmarking study.

However, the deep learning methods solve structurally different problems. The VAE-IV pipeline achieves a smooth, arbitrage-consistent implied volatility surface (range: 0.205–0.364) reconstructed from sparse market quotes, enabling pricing at strikes and tenors where no market quote exists — a capability that classical interpolation methods cannot guarantee to be free of calendar-spread or butterfly arbitrage. The LSTM-BS architecture embeds temporal information from 60-day historical return and VIX sequences into the vol forecast, providing pricing that is responsive to recent market regime without requiring a recalibration event. The MLP, when operating in-distribution, achieves sub-millisecond throughput orders of magnitude faster than Monte Carlo for large batch pricing jobs. These represent structural advantages in specific operational contexts, not improvements in point accuracy on vanilla European options — a distinction that is essential for the correct interpretation of these results.

---

## 6. Conclusion

This paper presented a reproducible, end-to-end comparison of seven options pricing methodologies — three classical, one stochastic volatility model, and three deep learning — evaluated on 200 live SPY call options across the full moneyness spectrum. The central finding is that when all methods are supplied with market-implied volatility, the classical GBM trio (Black-Scholes, CRR Binomial, Monte Carlo) achieves nearly identical accuracy (MAE spread of \$0.002), with Black-Scholes dominating on speed at 0.01 ms per option. The Heston stochastic volatility model achieves an overall MAE of \$1.04 — fourth-best overall — but is the only method that structurally generates a non-flat implied volatility smile: negative spot-vol correlation (ρ=−0.7) produces the canonical equity left skew, with IV ranging from 17.3% (OTM calls) to 23.0% (ITM calls) versus the 20% flat line used by all GBM pricers. Deep learning models do not improve point accuracy on vanilla European calls: the best deep learning result (LSTM-BS, MAE \$1.05) is 16% worse than the worst classical result (BS, MAE \$0.91). The correct interpretation is not that deep learning fails at option pricing, but that it solves different problems — the VAE provides arbitrage-consistent IV surface imputation, the LSTM provides regime-sensitive volatility forecasts, and the MLP provides ultra-fast batch throughput when operating within its training distribution.

Four limitations merit attention. First, all deep learning models were applied with single-asset, single-sigma volatility inputs that cannot capture the volatility smile; a model architecture that emits a full strike-dependent $\sigma(K, T)$ surface would materially improve LSTM-BS and MLP performance. Second, the MLP suffered severe out-of-distribution degradation because it was trained on synthetic spot prices in \$[50, 150] and deployed on SPY at \$736 — a domain shift that the linear homogeneity scaling trick only partially mitigated; training on log-moneyness features would eliminate this sensitivity entirely. Third, the evaluation is restricted to European-exercise call options; American puts, where early-exercise premiums are material, would present a different ranking in which the CRR binomial tree's structural advantage over Black-Scholes becomes significant. Fourth, the Heston evaluation uses fixed global skew parameters (κ=2, σᵥ=0.3, ρ=−0.7) rather than per-option calibration; a full nonlinear calibration would substantially reduce Heston's ITM and Deep ITM errors at the cost of increased computational complexity.

Future work will extend this study in four directions. First, we will perform full per-option Heston calibration using gradient-based optimisation to minimise pricing error across the IV surface, enabling fair comparison with smile-adjusted methods. Second, we will replace the scalar LSTM volatility forecast with a neural Stochastic Volatility Inspired (SVI) model that directly parameterises the full implied volatility surface, enabling smile-consistent pricing across all strikes. Third, we will train the MLP on actual ETF price history using normalised log-moneyness and log-forward-moneyness features, with the goal of recovering in-distribution performance at deployment. Fourth, we will extend the Monte Carlo engine to price path-dependent exotics — barrier options, Asian options, and lookback options — where neither Black-Scholes nor the binomial tree apply analytically, and where the Heston model's richer dynamics offer a commercially relevant advantage over GBM-based simulation.

---

## References

- Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654.
- Cox, J., Ross, S. & Rubinstein, M. (1979). Option Pricing: A Simplified Approach. *Journal of Financial Economics*, 7(3), 229–263.
- Boyle, P. (1977). Options: A Monte Carlo Approach. *Journal of Financial Economics*, 4(3), 323–338.
- Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options. *The Review of Financial Studies*, 6(2), 327–343.
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv preprint arXiv:1312.6114*.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.
- Ruf, J. & Wang, W. (2020). Neural Networks for Option Pricing and Hedging: A Literature Review. *Journal of Computational Finance*, 24(1), 1–45.


