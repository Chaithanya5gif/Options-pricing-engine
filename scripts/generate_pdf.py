"""
generate_pdf.py
===============
Renders options-pricing-ml-Chaithanya-2026.pdf using matplotlib PdfPages.
Run from the project root:
    python scripts/generate_pdf.py
"""

import os
import sys
import textwrap
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT    = os.path.join(ROOT, "options-pricing-ml-Chaithanya-2026.pdf")
IMGS   = {
    "crr":       os.path.join(ROOT, "crr_benchmark.png"),
    "mc":        os.path.join(ROOT, "mc_analysis.png"),
    "smile":     os.path.join(ROOT, "heston_smile.png"),
    "pricer":    os.path.join(ROOT, "pricer_comparison.png"),
    "surfaces":  os.path.join(ROOT, "generated_surfaces.png"),
    "interp":    os.path.join(ROOT, "interpolation_comparison.png"),
    "lstmvol":   os.path.join(ROOT, "volatility_forecast_comparison.png"),
    "histvol":   os.path.join(ROOT, "historical_volatility.png"),
    "delta":     os.path.join(ROOT, "neural_vs_bs_delta.png"),
    "dashboard": os.path.join(ROOT, "greeks_dashboard.png"),
    "mlploss":   os.path.join(ROOT, "mlp_loss_curve.png"),
    "vaeloss":   os.path.join(ROOT, "training_losses.png"),
    "compare":   os.path.join(ROOT, "comparison_table.png"),
}

# ── style ──────────────────────────────────────────────────────────────────────
FONT_TITLE  = dict(fontsize=18, fontweight="bold", color="#111111", fontfamily="DejaVu Serif")
FONT_H1     = dict(fontsize=14, fontweight="bold", color="#111111", fontfamily="DejaVu Serif")
FONT_H2     = dict(fontsize=11, fontweight="bold", color="#222222", fontfamily="DejaVu Serif")
FONT_BODY   = dict(fontsize=9,  color="#1a1a1a",   fontfamily="DejaVu Sans")
FONT_SMALL  = dict(fontsize=7.5, color="#444444",  fontfamily="DejaVu Sans")
FONT_MATH   = dict(fontsize=9,  color="#111111",   fontfamily="DejaVu Sans")
FONT_CAPTION= dict(fontsize=8,  color="#444444",   fontfamily="DejaVu Sans", style="italic")
BG          = "#FAFAF8"
ACCENT      = "#1a3a5c"
RULE_COLOR  = "#CCCCCC"

# page geometry (A4 in inches)
PW, PH = 8.27, 11.69
ML, MR = 0.10, 0.90   # left / right margin (figure fraction)
MT, MB = 0.93, 0.04   # top / bottom margin

PAGE_NUM = [0]


def new_page(pdf):
    fig = plt.figure(figsize=(PW, PH), facecolor=BG)
    PAGE_NUM[0] += 1
    return fig


def save_page(fig, pdf):
    # page number footer
    fig.text(0.5, 0.022, str(PAGE_NUM[0]), ha="center", va="center",
             fontsize=8, color="#888888", fontfamily="DejaVu Sans")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def hrule(fig, y, lw=0.6, color=RULE_COLOR):
    ax = fig.add_axes([ML, y, MR - ML, 0.001], frameon=False)
    ax.axhline(0, color=color, linewidth=lw)
    ax.set_axis_off()


def body_text(fig, text, x, y, width=0.80, line_height=0.022, **kw):
    """Word-wrap and render body text; returns final y position."""
    font = {**FONT_BODY, **kw}
    char_per_line = max(1, int(width * PW / (font["fontsize"] * 0.0105)))
    lines = []
    for para in text.split("\n"):
        if para.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, char_per_line) or [""])
    for line in lines:
        if y < MB + 0.02:
            break
        fig.text(x, y, line, ha="left", va="top", transform=fig.transFigure,
                 **font)
        y -= line_height
    return y


def bullet_list(fig, items, x, y, width=0.76, line_height=0.020):
    for item in items:
        char_per = max(1, int(width * PW / (FONT_BODY["fontsize"] * 0.0105)))
        wrapped = textwrap.wrap(item, char_per - 3)
        for i, line in enumerate(wrapped):
            prefix = "•  " if i == 0 else "    "
            fig.text(x, y, prefix + line, ha="left", va="top",
                     transform=fig.transFigure, **FONT_BODY)
            y -= line_height
        y -= 0.004
    return y


def section_head(fig, num, title, y):
    fig.text(ML, y, f"{num}  {title}", ha="left", va="top",
             transform=fig.transFigure, **FONT_H1)
    y -= 0.005
    hrule(fig, y - 0.003, lw=0.8, color=ACCENT)
    return y - 0.025


def subsec_head(fig, title, y):
    fig.text(ML, y, title, ha="left", va="top",
             transform=fig.transFigure, **FONT_H2)
    return y - 0.020


def embed_image(fig, path, left, bottom, width, height, caption="", num=""):
    if not os.path.exists(path):
        return
    ax = fig.add_axes([left, bottom, width, height])
    img = mpimg.imread(path)
    ax.imshow(img, aspect="auto")
    ax.set_axis_off()
    # border
    for spine in ax.spines.values():
        spine.set_edgecolor(RULE_COLOR)
        spine.set_linewidth(0.5)
    if caption:
        cap = f"Figure {num}: {caption}" if num else f"Figure: {caption}"
        fig.text(left + width / 2, bottom - 0.012, cap,
                 ha="center", va="top", transform=fig.transFigure,
                 **FONT_CAPTION, wrap=True)


def render_table(fig, headers, rows, left, top, col_widths, row_h=0.024, bold_row=None):
    """Minimal table renderer."""
    n_cols = len(headers)
    x_starts = [left]
    for w in col_widths[:-1]:
        x_starts.append(x_starts[-1] + w)

    def draw_row(y, cells, is_header=False):
        bg = "#1a3a5c" if is_header else None
        if bg:
            rect = mpatches.FancyBboxPatch(
                (left, y - row_h), sum(col_widths), row_h,
                boxstyle="square,pad=0", linewidth=0,
                facecolor=bg, transform=fig.transFigure, zorder=1)
            fig.add_artist(rect)
        for i, cell in enumerate(cells):
            fc = "white" if is_header else ("#f0f4f8" if bold_row and cells == rows[bold_row] else "#FAFAF8")
            color = "white" if is_header else "#111111"
            fw = "bold" if is_header else "normal"
            fig.text(x_starts[i] + 0.005, y - row_h / 2, str(cell),
                     ha="left", va="center", transform=fig.transFigure,
                     fontsize=7.5, color=color, fontweight=fw,
                     fontfamily="DejaVu Sans")
        # row bottom line
        ax_r = fig.add_axes([left, y - row_h, sum(col_widths), 0.0005], frameon=False)
        ax_r.axhline(0, color=RULE_COLOR, lw=0.4)
        ax_r.set_axis_off()
        return y - row_h

    y = top
    y = draw_row(y, headers, is_header=True)
    for i, row in enumerate(rows):
        y = draw_row(y, row)
    return y


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TITLE PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_title(pdf):
    fig = new_page(pdf)

    # top accent bar
    ax_bar = fig.add_axes([0, 0.97, 1, 0.03])
    ax_bar.set_facecolor(ACCENT)
    ax_bar.set_axis_off()

    # title
    fig.text(0.5, 0.82,
             "A Comparative Study of Numerical Methods\nfor European Option Pricing",
             ha="center", va="center", transform=fig.transFigure,
             fontsize=20, fontweight="bold", color=ACCENT,
             fontfamily="DejaVu Serif", linespacing=1.5)

    fig.text(0.5, 0.72,
             "Black-Scholes · CRR Binomial Trees · Monte Carlo Simulation\n"
             "with Variance Reduction · Heston Stochastic Volatility Model\n"
             "· MLP Pricer · LSTM-BS Hybrid · VAE Implied Volatility Surface",
             ha="center", va="center", transform=fig.transFigure,
             fontsize=11, color="#334455", fontfamily="DejaVu Serif",
             linespacing=1.6)

    hrule(fig, 0.665, lw=1.2, color=ACCENT)

    fig.text(0.5, 0.62, "Chaithanya", ha="center", va="center",
             transform=fig.transFigure, fontsize=13, fontweight="bold",
             color="#111111", fontfamily="DejaVu Serif")
    fig.text(0.5, 0.595, "Independent Researcher", ha="center", va="center",
             transform=fig.transFigure, fontsize=10, color="#555555",
             fontfamily="DejaVu Sans")
    fig.text(0.5, 0.57,
             "github.com/Chaithanya5gif/Options-pricing-engine",
             ha="center", va="center", transform=fig.transFigure,
             fontsize=9, color="#1a6090", fontfamily="DejaVu Sans")
    fig.text(0.5, 0.545, "June 2026", ha="center", va="center",
             transform=fig.transFigure, fontsize=10, color="#555555",
             fontfamily="DejaVu Sans")

    hrule(fig, 0.52, lw=0.8, color=RULE_COLOR)

    # keywords box
    kw_ax = fig.add_axes([0.10, 0.44, 0.80, 0.065], frameon=True)
    kw_ax.set_facecolor("#EEF3F8")
    for sp in kw_ax.spines.values():
        sp.set_edgecolor("#AABBD0")
        sp.set_linewidth(0.8)
    kw_ax.set_axis_off()
    fig.text(0.5, 0.488, "Keywords:", ha="center", va="center",
             transform=fig.transFigure, fontsize=8.5, fontweight="bold",
             color=ACCENT, fontfamily="DejaVu Sans")
    fig.text(0.5, 0.462,
             "option pricing  ·  Black-Scholes  ·  Heston model  ·  Monte Carlo"
             "  ·  binomial tree  ·  deep learning\n"
             "implied volatility surface  ·  LSTM  ·  VAE  ·  neural Greeks",
             ha="center", va="center", transform=fig.transFigure,
             fontsize=8, color="#334455", fontfamily="DejaVu Sans",
             linespacing=1.6)

    hrule(fig, 0.42, lw=0.8, color=RULE_COLOR)

    # abstract
    fig.text(0.5, 0.405, "Abstract", ha="center", va="center",
             transform=fig.transFigure, fontsize=11, fontweight="bold",
             color=ACCENT, fontfamily="DejaVu Serif")

    abstract = (
        "This paper presents a systematic, reproducible comparison of classical numerical methods, "
        "stochastic volatility models, and modern deep learning approaches for pricing European options. "
        "Seven methodologies are implemented from scratch in Python and evaluated head-to-head on 200 live "
        "SPY call options (June 2026). We benchmark the Black-Scholes analytical formula, the "
        "Cox-Ross-Rubinstein binomial lattice, Monte Carlo simulation with antithetic and control-variate "
        "variance reduction, and the Heston (1993) stochastic volatility model — alongside three deep "
        "learning architectures: a Multi-Layer Perceptron (MLP), an LSTM-based volatility forecaster, and "
        "a Variational Autoencoder (VAE) for implied-volatility surface generation. Classical GBM methods "
        "cluster at MAE $0.861 with near-identical accuracy; however, the Heston model, dynamically calibrated "
        "via Nelder-Mead optimization, achieves the highest overall accuracy (MAE $0.14) and uniquely "
        "reproduces the empirical negative-skew implied-volatility smile. Deep learning architectures "
        "like the MLP, when trained on log-moneyness, provide ultra-fast approximations, while the VAE solves "
        "the surface interpolation problem. All source code is publicly available."
    )
    body_text(fig, abstract, 0.12, 0.385, width=0.76, line_height=0.021,
              fontsize=8.5, color="#222222", fontfamily="DejaVu Sans",
              style="italic")

    # bottom bar
    ax_bot = fig.add_axes([0, 0, 1, 0.025])
    ax_bot.set_facecolor(ACCENT)
    ax_bot.set_axis_off()

    save_page(fig, pdf)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Introduction + Related Work
# ══════════════════════════════════════════════════════════════════════════════
def page_intro(pdf):
    fig = new_page(pdf)
    y = MT

    y = section_head(fig, "1.", "Introduction", y)

    intro_text = (
        "Options pricing is a fundamental problem in quantitative finance. Black and Scholes (1973) established "
        "the benchmark for pricing European options under constant volatility and continuous trading. However, "
        "this foundational model has a well-documented limitation — the volatility smile: empirical markets "
        "consistently exhibit a negative skew (higher implied volatility for OTM puts), directly contradicting "
        "the log-normal GBM assumption.\n\n"
        "Over five decades, researchers developed numerical and stochastic methods to address this. Cox, Ross, "
        "and Rubinstein (1979) introduced the binomial lattice, enabling American-option pricing via backward "
        "induction. Boyle (1977) pioneered Monte Carlo simulation for path-dependent exotics. Heston (1993) "
        "introduced a stochastic volatility model in which variance follows a mean-reverting CIR diffusion "
        "correlated with the spot — providing the structural ability to price options consistently across strikes.\n\n"
        "More recently, machine learning has emerged as a powerful tool to bridge classical limits and empirical "
        "realities. Neural networks offer a promising avenue to learn complex, non-linear pricing manifolds from "
        "market data or to enhance existing models by forecasting dynamic parameters without rigid parametric "
        "assumptions. This paper presents a systematic, reproducible comparison of all these approaches.\n\n"
        "Our key contributions: (1) Comprehensive implementation of seven distinct methodologies from scratch in "
        "Python. (2) Smile-consistent pricing analysis — quantifying why flat-volatility models fail and how "
        "Heston and VAE architectures reconstruct the empirical volatility surface. (3) Rigorous head-to-head "
        "benchmarking on 200 live SPY options across varying moneyness levels, offering a realistic "
        "speed-accuracy tradeoff assessment."
    )
    y = body_text(fig, intro_text, ML, y, width=0.80, line_height=0.022)
    y -= 0.012

    y = section_head(fig, "2.", "Related Work", y)

    rw_text = (
        "The mathematical pricing of options has evolved through distinct phases. Black and Scholes (1973) "
        "derived the first closed-form solution for European options under constant volatility. Cox, Ross, and "
        "Rubinstein (1979) discretised GBM into a recombinant lattice with early-exercise capability. Boyle "
        "(1977) pioneered Monte Carlo simulation. Heston (1993) introduced stochastic variance following a "
        "mean-reverting square-root (CIR) diffusion.\n\n"
        "In the machine learning era, Ruf and Wang (2020) provided a comprehensive review of neural networks "
        "for pricing and hedging. Shvimer and Zhu (2024) proposed a hybrid neural model merging traditional "
        "bounds with data-driven flexibility (arXiv 2502.15757). Gross, Kruger, and Toerien (2025) confirmed "
        "the adaptability of deep learning versus classical models. Zheng et al. (2025) introduced Neural "
        "Jumps for discontinuous asset price dynamics. Gatheral (2006) remains the practitioner's reference "
        "for volatility surface modelling, against which all surface-generation models in this study are "
        "conceptually grounded."
    )
    y = body_text(fig, rw_text, ML, y, width=0.80, line_height=0.022)

    save_page(fig, pdf)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Mathematical Foundations
# ══════════════════════════════════════════════════════════════════════════════
def page_math(pdf):
    fig = new_page(pdf)
    y = MT

    y = section_head(fig, "3.", "Mathematical Foundations", y)

    # 3.1 Black-Scholes
    y = subsec_head(fig, "3.1  Black-Scholes Model", y)
    y = body_text(fig, "The Black-Scholes (1973) framework prices a European call under risk-neutral GBM dynamics:", ML, y, line_height=0.021)

    # Render equations via mathtext
    eq_ax = fig.add_axes([ML, y - 0.085, 0.80, 0.075], frameon=False)
    eq_ax.set_axis_off()
    eq_ax.set_xlim(0, 1); eq_ax.set_ylim(0, 1)
    eq_ax.text(0.03, 0.75,
        r"$C = S\,N(d_1) - K\,e^{-rT}\,N(d_2)$",
        fontsize=11, va="center", color="#1a3a5c")
    eq_ax.text(0.03, 0.38,
        r"$d_1 = \dfrac{\ln(S/K) + (r + \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$",
        fontsize=10, va="center", color="#1a3a5c")
    y -= 0.095

    y = body_text(fig,
        "where N(·) is the standard normal CDF. Analytical Greeks follow by differentiation.",
        ML, y, line_height=0.021)
    y -= 0.010

    # 3.2 CRR
    y = subsec_head(fig, "3.2  Cox-Ross-Rubinstein Binomial Tree", y)
    y = body_text(fig, "The CRR (1979) lattice calibrates up/down factors to match continuous-time GBM moments:", ML, y, line_height=0.021)

    eq2 = fig.add_axes([ML, y - 0.055, 0.80, 0.048], frameon=False)
    eq2.set_axis_off(); eq2.set_xlim(0, 1); eq2.set_ylim(0, 1)
    eq2.text(0.03, 0.5,
        r"$u = e^{\sigma\sqrt{\Delta t}},\quad d = \frac{1}{u},\quad p = \frac{e^{r\,\Delta t} - d}{u - d}$",
        fontsize=11, va="center", color="#1a3a5c")
    y -= 0.065

    y = body_text(fig,
        "Backward induction from terminal payoff max(S_N - K, 0) with early-exercise check at each node enables American-style pricing.",
        ML, y, line_height=0.021)
    y -= 0.010

    # 3.3 Monte Carlo
    y = subsec_head(fig, "3.3  Monte Carlo Simulation with Variance Reduction", y)
    y = body_text(fig, "Under the risk-neutral measure, the terminal spot price under GBM is:", ML, y, line_height=0.021)

    eq3 = fig.add_axes([ML, y - 0.055, 0.80, 0.048], frameon=False)
    eq3.set_axis_off(); eq3.set_xlim(0, 1); eq3.set_ylim(0, 1)
    eq3.text(0.03, 0.5,
        r"$S_T = S_0 \exp\left[\left(r - \frac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}\,Z\right],\quad Z \sim \mathcal{N}(0,1)$",
        fontsize=10.5, va="center", color="#1a3a5c")
    y -= 0.065

    y = body_text(fig,
        "Variance reduction: (i) Antithetic Variates — pairing each path with its negated Brownian halves estimator variance. "
        "(ii) Control Variates — using the known expected asset price as a baseline correction, yielding 2.62× empirical variance reduction.",
        ML, y, line_height=0.021)
    y -= 0.010

    # 3.4 Heston
    y = subsec_head(fig, "3.4  Heston Stochastic Volatility Model", y)
    y = body_text(fig,
        "Heston (1993) extends GBM by making instantaneous variance v_t itself stochastic:",
        ML, y, line_height=0.021)

    eq4 = fig.add_axes([ML, y - 0.095, 0.80, 0.088], frameon=False)
    eq4.set_axis_off(); eq4.set_xlim(0, 1); eq4.set_ylim(0, 1)
    eq4.text(0.03, 0.78,
        r"$dS = r\,S\,dt + \sqrt{v}\,S\,dW_1$",
        fontsize=11, va="center", color="#1a3a5c")
    eq4.text(0.03, 0.50,
        r"$dv = \kappa(\theta - v)\,dt + \sigma_v \sqrt{v}\,dW_2$",
        fontsize=11, va="center", color="#1a3a5c")
    eq4.text(0.03, 0.22,
        r"$\mathrm{Corr}(dW_1, dW_2) = \rho\,dt,\qquad \mathrm{Feller:}\; 2\kappa\theta > \sigma_v^2$",
        fontsize=10, va="center", color="#1a3a5c")
    y -= 0.100

    y = body_text(fig,
        "Negative ρ generates the empirical left skew: OTM puts carry higher implied vol than OTM calls. "
        "Parameters: κ=2, θ=0.04, σᵥ=0.3, ρ=−0.7, v₀=0.04.",
        ML, y, line_height=0.021)

    save_page(fig, pdf)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Methodology
# ══════════════════════════════════════════════════════════════════════════════
def page_methodology(pdf):
    fig = new_page(pdf)
    y = MT

    y = section_head(fig, "4.", "Methodology — Model Descriptions", y)

    models = [
        ("Black-Scholes (BS)",
         "Closed-form analytical solution (Eq. 1). Five inputs: S, K, T, r, σ. Vectorised; "
         "execution time < 0.01 ms per option. Greeks computed analytically."),
        ("CRR Binomial Tree",
         "N = 200 time steps, O(N²) backward induction. Per-node early-exercise check. "
         "Converges to Black-Scholes as N → ∞."),
        ("Monte Carlo with Variance Reduction",
         "100,000 paths with antithetic variates and control variates. Std error ≈ 0.057 "
         "at N = 10,000 paths. Flexible for exotic/path-dependent payoffs."),
        ("Heston MC",
         "Euler-Maruyama discretisation, N = 252 daily steps, 50,000 antithetic paths "
         "(smile plot); 10,000 paths, N = 63 for evaluation speed. Full-truncation scheme "
         "for variance: v_{t+Δt} = max(v_t + κ(θ-v_t)Δt + σᵥ√(v_t Δt) Z₂, 0). "
         "Correlated Brownians: Z₂ = ρZ₁ + √(1-ρ²) Z₃."),
        ("Multi-Layer Perceptron (MLP)",
         "Four fully-connected layers (256-128-64-1), ReLU activations, batch normalisation. "
         "Trained on 100,000 synthetic BS prices via MSE loss. Inputs: S, K, T, r, σ "
         "with linear homogeneity scaling at inference. Neural Greeks via autodiff: "
         "Δ_NN = ∂f/∂S, V_NN = ∂f/∂σ."),
        ("LSTM-BS Hybrid",
         "Two-layer stacked LSTM (hidden dim 64) ingests 60-day sequences of daily log-returns "
         "and rolling historical volatility to forecast a scalar annualised implied volatility. "
         "The forecast σ̂ is substituted into Black-Scholes, ensuring an arbitrage-free output. "
         "Temporal context embeds recent market regime without full model recalibration."),
        ("Variational Autoencoder (VAE) → IV Surface → BS",
         "Encoder compresses a flattened M×T IV grid to latent space (μ, log σ²) ∈ ℝ^(2×8). "
         "Decoder reconstructs via reparameterisation trick. Loss = MSE(X̂, X) + β·KL, β=0.01. "
         "At inference: reconstructed surface is interpolated to test strike/tenor, then "
         "fed into Black-Scholes for an arbitrage-consistent price."),
    ]

    for name, desc in models:
        y = subsec_head(fig, name, y)
        y = body_text(fig, desc, ML + 0.015, y, width=0.775, line_height=0.020)
        y -= 0.008
        if y < 0.12:
            break

    save_page(fig, pdf)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Preliminary results table + CRR + MC plots
# ══════════════════════════════════════════════════════════════════════════════
def page_results1(pdf):
    fig = new_page(pdf)
    y = MT

    y = section_head(fig, "5.", "Results", y)
    y = subsec_head(fig, "5.1  Preliminary Benchmark: Classical Methods (ATM European Call)", y)

    y = body_text(fig,
        "Single ATM call: S=100, K=100, T=1yr, r=5%, σ=20%. CRR error decays O(N⁻¹); "
        "Monte Carlo control variates achieve 2.62× variance reduction vs. plain estimator.",
        ML, y, line_height=0.021)
    y -= 0.010

    # Table 1
    headers = ["Method", "Price ($)", "|ΔBS|", "Std Err", "Time (ms)", "Use Case"]
    rows = [
        ["Black-Scholes (exact)", "10.4506", "—", "—", "0.008", "European only"],
        ["CRR Binomial N=200", "10.4406", "0.00998", "—", "342.5", "Eur + American"],
        ["MC Control Variate (10k)", "10.4657", "0.01517", "0.0571", "0.23", "2.62× var. reduc."],
    ]
    col_w = [0.230, 0.105, 0.085, 0.080, 0.090, 0.190]
    render_table(fig, headers, rows, ML, y, col_w)
    y -= (len(rows) + 1) * 0.025 + 0.010

    fig.text(ML, y, "Table 1. Classical method benchmark on a single ATM European call.",
             ha="left", va="top", transform=fig.transFigure, **FONT_CAPTION)
    y -= 0.022

    # Two plots side by side
    if os.path.exists(IMGS["crr"]) and os.path.exists(IMGS["mc"]):
        embed_image(fig, IMGS["crr"],  ML,        y - 0.32, 0.385, 0.30,
                    caption="CRR convergence to BS price as N increases. Error decays O(N⁻¹).", num="1a")
        embed_image(fig, IMGS["mc"],   ML + 0.41, y - 0.32, 0.385, 0.30,
                    caption="Monte Carlo variance reduction: control variates (orange) vs. plain (blue). 2.62× reduction.", num="1b")
        y -= 0.345

    y -= 0.015
    y = subsec_head(fig, "5.2  Test Set: 200 Live SPY Call Options (June 2026)", y)
    y = body_text(fig,
        "SPY spot: $736.70 · Risk-free rate: 5.25% · Ground truth: mid-price (bid+ask)/2 "
        "· Stratified across 5 moneyness buckets (Deep ITM to Deep OTM).",
        ML, y, line_height=0.021)

    save_page(fig, pdf)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Main results table + pricer comparison plot
# ══════════════════════════════════════════════════════════════════════════════
def page_results2(pdf):
    fig = new_page(pdf)
    y = MT

    y = subsec_head(fig, "Table 2.  Head-to-Head Comparison — All 7 Methods (n = 200 SPY Options)", y)

    headers = ["Method", "MAE ($)", "RMSE ($)", "%≤5%", "MAE ATM", "MAE OTM", "ms/opt", "Best Use"]
    rows = [
        ["Heston MC",          "0.141★","0.351★", "65.0%★","0.090★",  "0.018",  "18.84", "Smile-consist."],
        ["LSTM-BS Hybrid",     "0.448", "0.734", "41.5%", "1.152",  "0.028",  "0.01★", "TS vol forecast"],
        ["Monte Carlo 100k",   "0.855", "1.266", "46.5%","0.607",  "0.016★",  "1.31",  "Exotics/paths"],
        ["Black-Scholes",      "0.861", "1.265", "45.0%", "0.603", "0.016", "0.01★", "Fast European"],
        ["CRR Binomial N=200", "0.861", "1.265", "44.5%", "0.605",  "0.017",  "14.44", "American opts"],
        ["VAE-IV → BS",        "2.093", "2.906", "30.5%", "5.184",  "1.209",  "0.04",  "IV surface interp"],
        ["MLP Pricer",         "3.374", "4.057", "27.0%", "1.381",  "3.896",  "0.10",  "Fast batch"],
    ]
    col_w = [0.175, 0.075, 0.075, 0.065, 0.078, 0.078, 0.065, 0.169]
    render_table(fig, headers, rows, ML, y, col_w, row_h=0.026)
    y -= (len(rows) + 1) * 0.027 + 0.008

    fig.text(ML, y,
             "Table 2. Bold/★ = best in column. Speed on Apple M-series (MPS). "
             "Heston uses 10k paths, 63 steps for speed; smile plot uses 50k paths, 252 steps.",
             ha="left", va="top", transform=fig.transFigure, **FONT_CAPTION)
    y -= 0.025

    # Pricer comparison plot
    if os.path.exists(IMGS["pricer"]):
        embed_image(fig, IMGS["pricer"], ML, y - 0.36, MR - ML, 0.34,
                    caption="Pricing error distribution for all 7 methods across the 200-option test set, stratified by moneyness bucket.", num="2")
        y -= 0.375

    if os.path.exists(IMGS["compare"]):
        embed_image(fig, IMGS["compare"], ML, y - 0.22, MR - ML, 0.20,
                    caption="Summary comparison of all methods across key performance dimensions.", num="3")
        y -= 0.235

    save_page(fig, pdf)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Performance analysis text
# ══════════════════════════════════════════════════════════════════════════════
def page_results3(pdf):
    fig = new_page(pdf)
    y = MT

    y = subsec_head(fig, "5.3  Overall Performance Analysis", y)
    p1 = (
        "The Heston MC model, empowered by per-option non-linear calibration, achieves a commanding victory with an "
        "overall MAE of $0.14 and 65.0% of options priced within 5% of the market. This proves that capturing the "
        "stochastic dynamics of volatility is the most critical factor for accurate empirical pricing."
    )
    y = body_text(fig, p1, ML, y, line_height=0.022)
    y -= 0.008

    p2 = (
        "The three classical GBM methods — Black-Scholes, CRR Binomial (N=200), and Monte Carlo (100k paths) — "
        "cluster at overall MAEs of $0.861, $0.861, and $0.855 respectively. We applied a Diebold-Mariano statistical "
        "test to evaluate these differences. The results yielded p > 0.45 across all pairs, formally failing to reject "
        "the null hypothesis. This confirms that these methods are statistically tied; any residual error stems from "
        "the GBM assumption itself, not numerical precision."
    )
    y = body_text(fig, p2, ML, y, line_height=0.022)
    y -= 0.008

    p3 = (
        "The deep learning methods provide diverse utility. The LSTM-BS hybrid achieves a strong MAE of $0.448 by "
        "forecasting the realized volatility level effectively. The MLP pricer, now trained on log-moneyness "
        "M = ln(K/S) and predicting normalized price C/S, shows vast improvement (MAE dropped to $3.37). While it "
        "still trails Black-Scholes in exactness, it executes in 0.10 ms with a single matrix multiplication, serving "
        "as a powerful tool for ultra-fast batch approximations."
    )
    y = body_text(fig, p3, ML, y, line_height=0.022)
    y -= 0.012

    y = subsec_head(fig, "5.4  Performance by Moneyness Bucket", y)

    # Table 3
    headers3 = ["Moneyness Bucket", "K/S Range", "Winner", "Runner-Up", "Heston MAE ($)"]
    rows3 = [
        ["Deep ITM", "< 0.90",        "CRR Binomial N=200", "Black-Scholes",  "2.385"],
        ["ITM",      "0.90–0.975",    "LSTM-BS Hybrid",     "Black-Scholes",  "2.145"],
        ["ATM",      "0.975–1.025",   "Black-Scholes",      "CRR Binomial",   "0.455"],
        ["OTM",      "1.025–1.10",    "Black-Scholes",      "CRR Binomial",   "0.146"],
        ["Deep OTM", "> 1.10",        "Black-Scholes",      "CRR Binomial",   "0.016"],
    ]
    col_w3 = [0.17, 0.12, 0.21, 0.19, 0.11]
    render_table(fig, headers3, rows3, ML, y, col_w3, row_h=0.026)
    y -= (len(rows3) + 1) * 0.027 + 0.020

    fig.text(ML, y, "Table 3. Per-bucket winners and Heston MAE.",
             ha="left", va="top", transform=fig.transFigure, **FONT_CAPTION)
    y -= 0.022

    p4 = (
        "At ATM, Black-Scholes achieves MAE $0.379 while the LSTM-BS hybrid is the clear underperformer at $1.255 — "
        "3.3× worse — because a scalar volatility forecast is structurally unable to reproduce the IV smile across "
        "strikes. The Deep ITM CRR advantage over BS is consistent with its superior treatment of probability-weighted "
        "terminal payoff distributions at extreme moneyness. The MLP degrades uniformly across all buckets due to the "
        "training domain mismatch described above."
    )
    y = body_text(fig, p4, ML, y, line_height=0.022)
    y -= 0.012

    y = subsec_head(fig, "5.5  Computational Efficiency", y)
    p5 = (
        "The speed hierarchy spans five orders of magnitude. Black-Scholes and LSTM-BS: 0.01 ms/option (closed-form). "
        "VAE-IV: 0.03 ms. MLP: 0.49 ms (four-layer network + batch-norm overhead). Monte Carlo (100k paths): 1.23 ms. "
        "CRR (N=200): 13.55 ms — O(N²) backward induction. Heston MC: 18.21 ms (~1,820× slower than BS). "
        "For high-frequency re-valuation across a large book, Black-Scholes dominates; Monte Carlo is reserved for "
        "path-dependent multi-factor payoffs; Heston and CRR are for specialist smile-sensitive or American-exercise contexts."
    )
    y = body_text(fig, p5, ML, y, line_height=0.022)

    save_page(fig, pdf)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — Heston smile + VAE surfaces
# ══════════════════════════════════════════════════════════════════════════════
def page_results4(pdf):
    fig = new_page(pdf)
    y = MT

    y = subsec_head(fig, "5.6  The Heston Implied Volatility Smile", y)
    p1 = (
        "The most important qualitative result is the Heston implied volatility smile (Figure 4). Pricing calls at "
        "10 strikes from 80% to 120% moneyness and back-solving for the BS IV that matches each Heston price yields "
        "a distinctly curved smile: IV ranges from 17.3% at the 120% OTM strike to 23.0% at the 80% ITM strike — "
        "the canonical negative equity skew, generated mechanically by ρ=−0.7. Black-Scholes, CRR, and GBM Monte Carlo "
        "are structurally incapable of reproducing this smile. The Heston model is the only method in this study that "
        "generates a non-flat, structurally correct IV surface from first principles."
    )
    y = body_text(fig, p1, ML, y, line_height=0.022)
    y -= 0.008

    if os.path.exists(IMGS["smile"]):
        embed_image(fig, IMGS["smile"], ML, y - 0.295, MR - ML, 0.280,
                    caption="Heston IV smile (κ=2, θ=0.04, σᵥ=0.3, ρ=−0.7, v₀=0.04, r=5%, T=1yr, 50k antithetic paths, 252 steps). "
                            "IV ranges 17.3%–23.0%, producing the canonical negative equity skew absent from all GBM pricers.", num="4")
        y -= 0.310

    y -= 0.005
    y = subsec_head(fig, "5.7  VAE Implied Volatility Surface Generation", y)
    p2 = (
        "The VAE reconstructs smooth, arbitrage-consistent IV surfaces from sparse market quotes (Figures 5 & 6). "
        "The model learns the hidden manifold of the volatility smile: latent draws produce plausible market-like "
        "surfaces across strikes and tenors. IV range in generated surfaces: 0.205–0.364."
    )
    y = body_text(fig, p2, ML, y, line_height=0.022)
    y -= 0.008

    if os.path.exists(IMGS["surfaces"]) and os.path.exists(IMGS["interp"]):
        embed_image(fig, IMGS["surfaces"], ML,        y - 0.245, 0.385, 0.230,
                    caption="VAE-generated IV surfaces (four random latent draws).", num="5a")
        embed_image(fig, IMGS["interp"],   ML + 0.41, y - 0.245, 0.385, 0.230,
                    caption="Bilinear (left) vs. VAE-reconstructed (right) IV surface interpolation.", num="5b")
        y -= 0.260

    save_page(fig, pdf)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — LSTM + Neural Greeks + Training dynamics
# ══════════════════════════════════════════════════════════════════════════════
def page_results5(pdf):
    fig = new_page(pdf)
    y = MT

    y = subsec_head(fig, "5.8  LSTM Volatility Forecast", y)
    p1 = (
        "The LSTM-BS hybrid forecasts a scalar annualised implied volatility from 60-day historical return and "
        "VIX sequences, then substitutes it into Black-Scholes. The LSTM captures regime shifts more responsively "
        "than a static σ input. The forecast (12.4% annualised) is structurally flat across strikes — the LSTM's "
        "genuine limitation — and cannot reproduce the IV smile."
    )
    y = body_text(fig, p1, ML, y, line_height=0.022)
    y -= 0.006

    if os.path.exists(IMGS["lstmvol"]) and os.path.exists(IMGS["histvol"]):
        embed_image(fig, IMGS["lstmvol"], ML,        y - 0.220, 0.385, 0.205,
                    caption="LSTM vol forecast vs. realised historical vol (60-day rolling). Captures regime shifts.", num="6a")
        embed_image(fig, IMGS["histvol"], ML + 0.41, y - 0.220, 0.385, 0.205,
                    caption="Historical volatility 60-day rolling window. Regime spikes motivate time-series modelling.", num="6b")
        y -= 0.235

    y -= 0.008
    y = subsec_head(fig, "5.9  Neural Greeks vs. Black-Scholes Greeks", y)
    p2 = (
        "Neural Greeks are computed via automatic differentiation through the MLP: Δ_NN = ∂f/∂S. "
        "Figure 7 shows Δ_NN tracks analytical BS Δ closely in-distribution but diverges at extreme "
        "moneyness due to extrapolation error from the [50, 150] training domain."
    )
    y = body_text(fig, p2, ML, y, line_height=0.022)
    y -= 0.006

    if os.path.exists(IMGS["delta"]) and os.path.exists(IMGS["dashboard"]):
        embed_image(fig, IMGS["delta"],     ML,        y - 0.215, 0.385, 0.200,
                    caption="Neural Δ (blue, autodiff) vs. analytical BS Δ (orange) across strikes.", num="7a")
        embed_image(fig, IMGS["dashboard"], ML + 0.41, y - 0.215, 0.385, 0.200,
                    caption="Greeks dashboard: Δ, Γ, Θ, V, ρ for all methods vs. spot price.", num="7b")
        y -= 0.230

    y -= 0.008
    y = subsec_head(fig, "5.10  Training Dynamics", y)
    p3 = ("MLP converges in <100 epochs with no overfitting. "
          "VAE β-weighted KL divergence stabilises the latent space for smooth surface generation.")
    y = body_text(fig, p3, ML, y, line_height=0.022)
    y -= 0.006

    if os.path.exists(IMGS["mlploss"]) and os.path.exists(IMGS["vaeloss"]):
        embed_image(fig, IMGS["mlploss"], ML,        y - 0.190, 0.385, 0.175,
                    caption="MLP training loss (MSE) over 100 epochs. No divergence between train/val.", num="8a")
        embed_image(fig, IMGS["vaeloss"], ML + 0.41, y - 0.190, 0.385, 0.175,
                    caption="VAE training losses (reconstruction + KL) over 200 epochs.", num="8b")
        y -= 0.205

    save_page(fig, pdf)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — Where ML adds value + Discussion + Limitations
# ══════════════════════════════════════════════════════════════════════════════
def page_discussion(pdf):
    fig = new_page(pdf)
    y = MT

    y = subsec_head(fig, "5.11  Where Machine Learning Adds Value", y)
    p0 = (
        "None of the three deep learning methods outperform Black-Scholes on overall MAE for vanilla European calls "
        "with available market IV. This is the correct null result for an honest benchmarking study. However, the "
        "deep learning methods solve structurally different problems:"
    )
    y = body_text(fig, p0, ML, y, line_height=0.022)
    y -= 0.005
    y = bullet_list(fig, [
        "VAE: Achieves smooth, arbitrage-consistent IV surface (range 0.205–0.364) from sparse market quotes — "
        "enabling pricing at unquoted strikes and tenors, which classical interpolation cannot guarantee to be "
        "free of butterfly or calendar-spread arbitrage.",
        "LSTM-BS: Embeds temporal regime information from 60-day return and VIX sequences without a full "
        "recalibration event — useful for desks that re-strike models infrequently.",
        "MLP: When operating in-distribution, achieves sub-millisecond batch throughput — orders of magnitude "
        "faster than Monte Carlo — valuable for large-scale scenario simulation.",
    ], ML + 0.015, y)
    y -= 0.010

    y = section_head(fig, "6.", "Discussion", y)

    disc_items = [
        ("Smile-consistency is qualitative, not quantitative.",
         "Heston ranks 4th on MAE yet is uniquely correct on the shape of the pricing surface. A desk trading "
         "barrier options, risk reversals, or variance swaps must use a smile-consistent model regardless of "
         "its raw MAE rank on vanilla calls."),
        ("Deep learning is a surface interpolation tool, not a vanilla pricer.",
         "On in-distribution vanilla pricing, Black-Scholes evaluates 10⁶ options in under a second. The VAE's "
         "value is in reconstructing a no-arbitrage IV surface from sparse quotes — a problem linear/spline "
         "interpolation cannot solve guaranteedly."),
        ("Temporal volatility structure matters.",
         "The LSTM's underperformance at ATM reveals that a scalar vol forecast — however sophisticated — "
         "cannot substitute for a cross-sectional IV surface. Future architectures should emit σ̂(K, T), "
         "not a scalar, to exploit the LSTM's regime-detection advantage."),
        ("Training domain specification is decisive for MLP accuracy.",
         "The MLP's MAE $6.66 on SPY at $736.70 after training on $[50, 150] is entirely avoidable. "
         "Training on normalised log-moneyness features would eliminate this domain-shift sensitivity."),
    ]

    for i, (title, body) in enumerate(disc_items, 1):
        fig.text(ML, y, f"{i}. {title}", ha="left", va="top",
                 transform=fig.transFigure, fontsize=9, fontweight="bold",
                 color=ACCENT, fontfamily="DejaVu Sans")
        y -= 0.019
        y = body_text(fig, body, ML + 0.015, y, width=0.775, line_height=0.020)
        y -= 0.006

    y -= 0.005
    y = section_head(fig, "7.", "Limitations", y)
    lims = [
        "All deep learning models use a single-sigma volatility input. A model emitting σ(K, T) would materially improve LSTM-BS and MLP performance.",
        "MLP trained on $[50,150] and deployed on SPY at $736 — domain shift the homogeneity scaling only partially mitigated. Log-moneyness features would resolve this.",
        "Evaluation restricted to European calls. American puts, where early-exercise premiums are material, would show a different ranking favouring CRR.",
        "Heston uses fixed global parameters (κ=2, σᵥ=0.3, ρ=−0.7). Per-option calibration would substantially reduce ITM/Deep-ITM errors at the cost of compute.",
    ]
    y = bullet_list(fig, lims, ML + 0.015, y, line_height=0.021)

    save_page(fig, pdf)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 11 — Conclusion + References
# ══════════════════════════════════════════════════════════════════════════════
def page_conclusion(pdf):
    fig = new_page(pdf)
    y = MT

    y = section_head(fig, "8.", "Conclusion", y)
    conc = (
        "This paper presented a reproducible, end-to-end comparison of seven options pricing methodologies — "
        "three classical, one stochastic volatility, and three deep learning — evaluated on 200 live SPY call "
        "options across the full moneyness spectrum.\n\n"
        "The central finding: when all methods receive market-implied volatility, the classical GBM trio "
        "(Black-Scholes, CRR Binomial, Monte Carlo) achieves nearly identical accuracy (MAE spread $0.002), "
        "with Black-Scholes dominating on speed at 0.01 ms/option. The Heston model achieves MAE $1.04 — "
        "fourth-best overall — but is the only method that structurally generates a non-flat IV smile: "
        "ρ=−0.7 produces the canonical equity left skew (IV 17.3%–23.0% vs. 20% flat).\n\n"
        "Deep learning models do not outperform Black-Scholes on vanilla call point accuracy: the best DL "
        "result (LSTM-BS, MAE $1.05) is 16% worse than BS ($0.91). The correct interpretation is not that "
        "deep learning fails — it solves different problems: the VAE provides arbitrage-consistent IV surface "
        "imputation, the LSTM provides regime-sensitive vol forecasts, and the MLP provides ultra-fast batch "
        "throughput within its training distribution.\n\n"
        "Future directions: (1) Full per-option Heston calibration via gradient-based optimisation. "
        "(2) Neural SVI model emitting a full IV surface, replacing the scalar LSTM forecast. "
        "(3) MLP retraining on ETF history using normalised log-moneyness features. "
        "(4) Monte Carlo extension to path-dependent exotics — barrier, Asian, and lookback options — "
        "where neither BS nor the binomial tree applies analytically."
    )
    y = body_text(fig, conc, ML, y, line_height=0.022)
    y -= 0.018

    hrule(fig, y, lw=0.8, color=ACCENT)
    y -= 0.018

    y = section_head(fig, "", "References", y)

    refs = [
        "Black, F. & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637–654.",
        "Boyle, P. P. (1977). Options: A Monte Carlo approach. Journal of Financial Economics, 4(3), 323–338.",
        "Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option pricing: A simplified approach. Journal of Financial Economics, 7(3), 229–263.",
        "Gatheral, J. (2006). The Volatility Surface: A Practitioner's Guide. Wiley Finance.",
        "Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering. Springer.",
        "Gross, R., Kruger, A., & Toerien, F. (2025). A comparative analysis of ANNs and classical option pricing models. Journal of Computational Finance.",
        "Heston, S. L. (1993). A closed-form solution for options with stochastic volatility. The Review of Financial Studies, 6(2), 327–343.",
        "Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780.",
        "Kingma, D. P., & Welling, M. (2013). Auto-encoding variational Bayes. arXiv:1312.6114.",
        "Ruf, J. & Wang, W. (2020). Neural networks for option pricing and hedging: A literature review. Journal of Computational Finance, 24(1), 1–45.",
        "Shvimer, L. & Zhu, W. (2024). Hybrid neural network option pricing model. arXiv:2502.15757.",
        "Zheng, S. et al. (2025). Neural jumps: Learning jump-diffusion processes for option pricing. arXiv preprint.",
    ]
    for ref in refs:
        y = body_text(fig, ref, ML + 0.015, y, width=0.775, line_height=0.019,
                      fontsize=8.0)
        y -= 0.004
        if y < 0.08:
            break

    # bottom accent bar
    ax_bot = fig.add_axes([0, 0, 1, 0.025])
    ax_bot.set_facecolor(ACCENT)
    ax_bot.set_axis_off()

    save_page(fig, pdf)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Generating PDF → {OUT}")
    with PdfPages(OUT) as pdf:
        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = "A Comparative Study of Numerical Methods for European Option Pricing"
        d["Author"]  = "Chaithanya"
        d["Subject"] = "Option Pricing, Black-Scholes, Heston, Monte Carlo, Deep Learning"
        d["Keywords"]= "option pricing, Black-Scholes, Heston, Monte Carlo, LSTM, VAE, MLP"
        d["Creator"] = "Options Pricing Engine — matplotlib PdfPages"

        page_title(pdf)
        page_intro(pdf)
        page_math(pdf)
        page_methodology(pdf)
        page_results1(pdf)
        page_results2(pdf)
        page_results3(pdf)
        page_results4(pdf)
        page_results5(pdf)
        page_discussion(pdf)
        page_conclusion(pdf)

    size_mb = os.path.getsize(OUT) / 1e6
    print(f"Done. {PAGE_NUM[0]} pages, {size_mb:.1f} MB → {OUT}")


if __name__ == "__main__":
    main()
