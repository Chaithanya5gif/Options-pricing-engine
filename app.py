"""
Options Pricing Engine — Streamlit Dashboard
=============================================
Tabs:
  1. Pricing        — All 6 methods side-by-side, live sliders
  2. Greeks         — BS Greeks as bar chart + delta/gamma plots
  3. Vol Smile      — Heston IV smile vs flat BS smile
  4. Model Comparison — comparison_table.csv rendered with colour highlights
"""

import math
import sys
import os
import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Add repo root to path so local modules are importable ─────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from black_scholes import (
    black_scholes,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_vega,
    black_scholes_theta,
    black_scholes_rho,
    implied_vol,
)
from monte_carlo import mc_price, mc_antithetic, mc_control_variate
from crr_binomial_tree import price_option
from heston_mc import heston_mc_price, compute_heston_smile

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Pricing Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Root vars */
    :root {
        --bg-primary:   #0d1117;
        --bg-card:      #161b22;
        --bg-hover:     #1c2128;
        --border:       #30363d;
        --accent-blue:  #38bdf8;
        --accent-pink:  #f472b6;
        --accent-green: #34d399;
        --accent-gold:  #facc15;
        --accent-purple:#a78bfa;
        --accent-orange:#fb923c;
        --text-primary: #e6edf3;
        --text-muted:   #8b949e;
    }

    /* Global background */
    .stApp { background: var(--bg-primary); }
    section[data-testid="stSidebar"] { background: var(--bg-card); border-right: 1px solid var(--border); }

    /* Fonts */
    html, body, .stApp, .stMarkdown, label, p, span { font-family: 'Inter', sans-serif !important; color: var(--text-primary); }

    /* Sidebar labels */
    .stSidebar label { color: var(--text-muted) !important; font-size: 0.78rem !important; font-weight: 500 !important; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 20px;
        transition: border-color 0.2s;
    }
    div[data-testid="metric-container"]:hover { border-color: var(--accent-blue); }
    div[data-testid="metric-container"] label { color: var(--text-muted) !important; font-size: 0.76rem !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 1.5rem !important; font-weight: 600 !important; color: var(--accent-blue) !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: var(--bg-card); border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] { background: transparent; color: var(--text-muted); border-radius: 8px; padding: 8px 20px; font-weight: 500; font-size: 0.88rem; transition: all 0.2s; }
    .stTabs [aria-selected="true"] { background: var(--accent-blue) !important; color: #000 !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 24px; }

    /* Sliders */
    .stSlider [data-testid="stTickBar"] { color: var(--text-muted); }

    /* Selectbox */
    .stSelectbox > div > div { background: var(--bg-card); border: 1px solid var(--border); color: var(--text-primary); }

    /* Dataframe / table */
    .stDataFrame { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

    /* Section headers */
    .section-header {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid var(--border);
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), var(--accent-pink));
    }
    .hero h1 { font-size: 1.7rem; font-weight: 700; margin: 0 0 6px 0; }
    .hero p  { color: var(--text-muted); font-size: 0.9rem; margin: 0; }

    /* Method badge */
    .method-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* Price card */
    .price-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
        cursor: default;
    }
    .price-card:hover { transform: translateY(-2px); border-color: var(--accent-blue); }
    .price-card .label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; margin-bottom: 8px; }
    .price-card .value { font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; font-weight: 700; color: var(--accent-blue); }
    .price-card .sub   { font-size: 0.78rem; color: var(--text-muted); margin-top: 4px; }

    /* Info box */
    .info-box {
        background: rgba(56, 189, 248, 0.06);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.84rem;
        color: var(--text-muted);
        margin-top: 12px;
    }
    .info-box b { color: var(--accent-blue); }

    /* Divider */
    hr { border-color: var(--border) !important; margin: 20px 0; }

    /* Spinner */
    .stSpinner > div { border-top-color: var(--accent-blue) !important; }

    /* Button */
    .stButton button {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        color: #000;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.84rem;
        padding: 8px 20px;
        transition: opacity 0.2s, transform 0.1s;
    }
    .stButton button:hover { opacity: 0.85; transform: translateY(-1px); }

    /* Warning / success */
    .stSuccess { background: rgba(52, 211, 153, 0.08); border: 1px solid rgba(52, 211, 153, 0.3); border-radius: 8px; }
    .stWarning { background: rgba(250, 204, 21, 0.08); border: 1px solid rgba(250, 204, 21, 0.3); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme helper ───────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(family="Inter, sans-serif", color="#e6edf3", size=12),
    xaxis=dict(gridcolor="#30363d", linecolor="#30363d", tickfont=dict(color="#8b949e", size=11)),
    yaxis=dict(gridcolor="#30363d", linecolor="#30363d", tickfont=dict(color="#8b949e", size=11)),
    margin=dict(l=50, r=30, t=50, b=50),
    hoverlabel=dict(bgcolor="#1c2128", bordercolor="#30363d", font=dict(color="#e6edf3")),
)
DEFAULT_LEGEND = dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="#30363d", borderwidth=1,
                      font=dict(color="#e6edf3", size=11))

COLORS = {
    "bs":    "#38bdf8",
    "crr":   "#34d399",
    "mc":    "#f472b6",
    "anti":  "#a78bfa",
    "cv":    "#fb923c",
    "heston":"#facc15",
    "mlp":   "#f87171",
    "delta": "#38bdf8",
    "gamma": "#a78bfa",
    "vega":  "#34d399",
    "theta": "#fb923c",
    "rho":   "#f472b6",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 20px 0;">
        <div style="font-size:1.15rem; font-weight:700; color:#e6edf3; letter-spacing:-0.01em;">📈 Options Engine</div>
        <div style="font-size:0.75rem; color:#8b949e; margin-top:4px;">Interactive Pricing Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Market Parameters</div>', unsafe_allow_html=True)
    S = st.slider("Spot Price  S ($)", min_value=50, max_value=200, value=100, step=1)
    K = st.slider("Strike Price  K ($)", min_value=50, max_value=200, value=100, step=1)
    T = st.slider("Time to Expiry  T (years)", min_value=0.1, max_value=2.0, value=1.0, step=0.05)
    r = st.slider("Risk-free Rate  r (%)", min_value=1, max_value=15, value=5, step=1) / 100.0
    sigma = st.slider("Volatility  σ (%)", min_value=5, max_value=80, value=20, step=1) / 100.0

    st.markdown('<div class="section-header" style="margin-top:20px;">Option Type</div>', unsafe_allow_html=True)
    option_type = st.selectbox("", ["call", "put"], format_func=lambda x: "🟢 Call" if x == "call" else "🔴 Put", label_visibility="collapsed")

    st.markdown('<div class="section-header" style="margin-top:20px;">Monte Carlo</div>', unsafe_allow_html=True)
    n_paths = st.select_slider("Paths", options=[1_000, 10_000, 50_000, 100_000], value=10_000,
                                format_func=lambda x: f"{x:,}")

    st.markdown('<div class="section-header" style="margin-top:20px;">CRR Binomial</div>', unsafe_allow_html=True)
    crr_N = st.slider("Steps  N", min_value=20, max_value=500, value=100, step=10)
    exercise_type = st.selectbox("Exercise", ["european", "american"],
                                 format_func=lambda x: x.capitalize())

    st.markdown('<div class="section-header" style="margin-top:20px;">Heston Parameters</div>', unsafe_allow_html=True)
    h_kappa   = st.slider("κ (mean-reversion)", 0.5, 5.0, 2.0, 0.1)
    h_theta   = st.slider("θ (long-run var)", 0.01, 0.16, 0.04, 0.01, format="%.2f")
    h_sigma_v = st.slider("σᵥ (vol-of-vol)", 0.05, 0.8, 0.3, 0.05)
    h_rho     = st.slider("ρ (spot-vol corr)", -0.99, 0.99, -0.7, 0.01)
    h_v0      = st.slider("v₀ (initial var)", 0.01, 0.16, 0.04, 0.01, format="%.2f")
    h_paths   = st.select_slider("Heston Paths", [1_000, 5_000, 10_000, 20_000], value=5_000,
                                  format_func=lambda x: f"{x:,}")

    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem; color:#8b949e; text-align:center;">Options Pricing Engine · 2026</div>',
                unsafe_allow_html=True)

# ── Compute all prices (cached on params) ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_bs(S, K, T, r, sigma, option_type):
    c, p = black_scholes(S, K, T, r, sigma)
    return c if option_type == "call" else p

@st.cache_data(show_spinner=False)
def compute_crr(S, K, T, r, sigma, N, option_type, exercise):
    t0 = time.perf_counter()
    res = price_option(S, K, T, r, sigma, N, option_type, exercise)
    ms = (time.perf_counter() - t0) * 1000
    return res["price"], res["delta"], res["gamma"], res["theta"], ms

@st.cache_data(show_spinner=False)
def compute_mc_all(S, K, T, r, sigma, n_paths, option_type):
    r1 = mc_price(S, K, T, r, sigma, n_paths, option_type)
    r2 = mc_antithetic(S, K, T, r, sigma, n_paths, option_type)
    r3 = mc_control_variate(S, K, T, r, sigma, n_paths, option_type)
    return r1, r2, r3

@st.cache_data(show_spinner=False)
def compute_heston(S, K, T, r, v0, kappa, theta, sigma_v, rho, n_paths, option_type):
    t0 = time.perf_counter()
    res = heston_mc_price(S, K, T, r, v0, kappa, theta, sigma_v, rho,
                          n_paths=n_paths, n_steps=63, option_type=option_type, seed=42)
    ms = (time.perf_counter() - t0) * 1000
    return res["price"], res["std_error"], ms

@st.cache_data(show_spinner=False)
def compute_heston_smile_cached(S, T, r, v0, kappa, theta, sigma_v, rho, n_paths):
    strikes, h_ivs, bs_ivs = compute_heston_smile(
        S0=S, T=T, r=r, v0=v0, kappa=kappa, theta=theta,
        sigma_v=sigma_v, rho=rho, n_strikes=12, n_paths=n_paths, n_steps=63,
    )
    return strikes, h_ivs, bs_ivs

@st.cache_data(show_spinner=False)
def compute_greeks(S, K, T, r, sigma):
    c_delta, p_delta = black_scholes_delta(S, K, T, r, sigma)
    gamma            = black_scholes_gamma(S, K, T, r, sigma)
    vega             = black_scholes_vega(S, K, T, r, sigma)
    c_theta, p_theta = black_scholes_theta(S, K, T, r, sigma)
    c_rho,   p_rho   = black_scholes_rho(S, K, T, r, sigma)
    return c_delta, p_delta, gamma, vega, c_theta, p_theta, c_rho, p_rho

# ── Pre-compute everything ─────────────────────────────────────────────────────
bs_price = compute_bs(S, K, T, r, sigma, option_type)
with st.spinner("Computing CRR…"):
    crr_price, crr_delta, crr_gamma, crr_theta, crr_ms = compute_crr(
        float(S), float(K), T, r, sigma, crr_N, option_type, exercise_type)
with st.spinner("Computing Monte Carlo…"):
    mc_r1, mc_r2, mc_r3 = compute_mc_all(float(S), float(K), T, r, sigma, n_paths, option_type)
with st.spinner("Computing Heston…"):
    h_price, h_se, h_ms = compute_heston(
        float(S), float(K), T, r, h_v0, h_kappa, h_theta, h_sigma_v, h_rho, h_paths, option_type)
c_delta, p_delta, gamma, vega, c_theta, p_theta, c_rho, p_rho = compute_greeks(S, K, T, r, sigma)

# ── Hero ──────────────────────────────────────────────────────────────────────
moneyness = S / K
mono_label = "ATM" if 0.975 <= moneyness <= 1.025 else ("ITM" if moneyness > 1.025 else "OTM")
mono_color = {"ATM": "#facc15", "ITM": "#34d399", "OTM": "#f472b6"}[mono_label]

st.markdown(f"""
<div class="hero">
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
        <div>
            <h1>📈 Options Pricing Engine</h1>
            <p>6 models · real-time Greeks · Heston vol smile · model comparison table</p>
        </div>
        <div style="display:flex; gap:12px; flex-wrap:wrap;">
            <div style="background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3); border-radius:10px; padding:10px 18px; text-align:center;">
                <div style="font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.06em;">Spot / Strike</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700; color:#38bdf8;">${S} / ${K}</div>
            </div>
            <div style="background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3); border-radius:10px; padding:10px 18px; text-align:center;">
                <div style="font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.06em;">Moneyness</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700; color:{mono_color};">{moneyness:.3f} ({mono_label})</div>
            </div>
            <div style="background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3); border-radius:10px; padding:10px 18px; text-align:center;">
                <div style="font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.06em;">BS Reference</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700; color:#38bdf8;">${bs_price:.4f}</div>
            </div>
            <div style="background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3); border-radius:10px; padding:10px 18px; text-align:center;">
                <div style="font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.06em;">Option Type</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700; color:#{'34d399' if option_type=='call' else 'f472b6'};">{'CALL ▲' if option_type=='call' else 'PUT ▼'}</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Pricing",
    "🧮 Greeks",
    "🌊 Vol Smile",
    "📊 Model Comparison",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICING
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">All 6 Pricing Methods · Live Output</div>', unsafe_allow_html=True)

    # ── Top metric row ──────────────────────────────────────────────────────
    cols = st.columns(6)
    methods_info = [
        ("Black-Scholes",   bs_price,       COLORS["bs"],     "Analytical"),
        ("CRR Binomial",    crr_price,      COLORS["crr"],    f"N={crr_N}"),
        ("Basic MC",        mc_r1["price"], COLORS["mc"],     f"{n_paths:,} paths"),
        ("Antithetic MC",   mc_r2["price"], COLORS["anti"],   f"SE={mc_r2['std_error']:.4f}"),
        ("Control Variate", mc_r3["price"], COLORS["cv"],     f"SE={mc_r3['std_error']:.4f}"),
        ("Heston MC",       h_price,        COLORS["heston"], f"SE={h_se:.4f}"),
    ]

    for col, (name, price, color, sub) in zip(cols, methods_info):
        diff = price - bs_price
        diff_str = f"{diff:+.4f} vs BS"
        with col:
            st.markdown(f"""
            <div class="price-card">
                <div class="label">{name}</div>
                <div class="value" style="color:{color};">${price:.4f}</div>
                <div class="sub">{sub}</div>
                <div style="font-size:0.7rem; color:{'#34d399' if abs(diff)<0.1 else '#f472b6'}; margin-top:6px; font-family:'JetBrains Mono',monospace;">{diff_str}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar chart ───────────────────────────────────────────────────────────
    method_names  = ["Black-Scholes", "CRR Binomial", "Basic MC", "Antithetic MC", "Control Variate", "Heston MC"]
    method_prices = [bs_price, crr_price, mc_r1["price"], mc_r2["price"], mc_r3["price"], h_price]
    method_colors = [COLORS["bs"], COLORS["crr"], COLORS["mc"], COLORS["anti"], COLORS["cv"], COLORS["heston"]]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=method_names,
        y=method_prices,
        marker=dict(color=method_colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        text=[f"${p:.4f}" for p in method_prices],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12, color="#e6edf3"),
        hovertemplate="<b>%{x}</b><br>Price: $%{y:.5f}<extra></extra>",
    ))
    fig_bar.add_hline(y=bs_price, line_dash="dash", line_color="#38bdf8",
                      annotation_text=f"BS = ${bs_price:.4f}", annotation_position="top right",
                      annotation_font=dict(color="#38bdf8", size=11))
    fig_bar.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Option Price by Method", font=dict(size=15, color="#e6edf3"), x=0),
        xaxis_title="Method",
        yaxis_title=f"{'Call' if option_type=='call' else 'Put'} Price ($)",
        showlegend=False,
        height=360,
        bargap=0.28,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── MC Convergence ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Monte Carlo Convergence  (Basic vs Antithetic vs Control Variate)</div>',
                unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def mc_convergence_data(S, K, T, r, sigma, option_type):
        Ns = [500, 1000, 2000, 5000, 10_000, 25_000, 50_000, 100_000]
        b_p, a_p, c_p, b_se, a_se, c_se = [], [], [], [], [], []
        for i, n in enumerate(Ns):
            rb = mc_price(S, K, T, r, sigma, n, option_type, seed=i)
            ra = mc_antithetic(S, K, T, r, sigma, n, option_type, seed=i)
            rc = mc_control_variate(S, K, T, r, sigma, n, option_type, seed=i)
            b_p.append(rb["price"]); a_p.append(ra["price"]); c_p.append(rc["price"])
            b_se.append(rb["std_error"]); a_se.append(ra["std_error"]); c_se.append(rc["std_error"])
        return Ns, b_p, a_p, c_p, b_se, a_se, c_se

    with st.spinner("Computing convergence…"):
        Ns, b_p, a_p, c_p, b_se, a_se, c_se = mc_convergence_data(
            float(S), float(K), T, r, sigma, option_type)

    fig_conv = make_subplots(rows=1, cols=2,
                             subplot_titles=["Price Convergence (log N)", "Std Error vs N (log-log)"])

    for name, prices, color, dash in [
        ("Basic MC", b_p, COLORS["mc"], "solid"),
        ("Antithetic", a_p, COLORS["anti"], "solid"),
        ("Control Variate", c_p, COLORS["cv"], "solid"),
    ]:
        fig_conv.add_trace(go.Scatter(x=Ns, y=prices, mode="lines+markers", name=name,
                                      line=dict(color=color, width=2, dash=dash),
                                      marker=dict(size=6), hovertemplate=f"<b>{name}</b><br>N=%{{x:,}}<br>Price=$%{{y:.5f}}<extra></extra>"),
                           row=1, col=1)

    fig_conv.add_hline(y=bs_price, line_dash="dash", line_color="#facc15",
                       annotation_text=f"BS Ref ${bs_price:.4f}",
                       annotation_font=dict(color="#facc15", size=10), row=1, col=1)

    for name, ses, color in [
        ("Basic MC SE", b_se, COLORS["mc"]),
        ("Antithetic SE", a_se, COLORS["anti"]),
        ("CV SE", c_se, COLORS["cv"]),
    ]:
        fig_conv.add_trace(go.Scatter(x=Ns, y=ses, mode="lines+markers", name=name,
                                      line=dict(color=color, width=2),
                                      marker=dict(size=6), showlegend=False,
                                      hovertemplate=f"<b>{name}</b><br>N=%{{x:,}}<br>SE=%{{y:.6f}}<extra></extra>"),
                           row=1, col=2)

    fig_conv.update_xaxes(type="log", gridcolor="#30363d", linecolor="#30363d",
                          tickfont=dict(color="#8b949e", size=10))
    fig_conv.update_yaxes(gridcolor="#30363d", linecolor="#30363d",
                          tickfont=dict(color="#8b949e", size=10))
    fig_conv.update_xaxes(type="log", row=1, col=2)
    fig_conv.update_yaxes(type="log", row=1, col=2)

    fig_conv.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(22,27,34,0.8)", bordercolor="#30363d", borderwidth=1,
                    font=dict(color="#e6edf3", size=11)),
    )
    fig_conv.update_annotations(font=dict(color="#8b949e", size=12))
    st.plotly_chart(fig_conv, use_container_width=True)

    # ── Summary table ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Summary Table</div>', unsafe_allow_html=True)
    summary_data = {
        "Method":     ["Black-Scholes", "CRR Binomial", "Basic MC", "Antithetic MC", "Control Variate", "Heston MC"],
        "Price ($)":  [f"{p:.5f}" for p in method_prices],
        "vs BS ($)":  [f"{p-bs_price:+.5f}" for p in method_prices],
        "Std Error":  ["—", "—", f"{mc_r1['std_error']:.5f}", f"{mc_r2['std_error']:.5f}", f"{mc_r3['std_error']:.5f}", f"{h_se:.5f}"],
        "Speed":      ["< 1 ms", f"{crr_ms:.1f} ms", f"{mc_r1['elapsed_ms']:.1f} ms", f"{mc_r2['elapsed_ms']:.1f} ms", f"{mc_r3['elapsed_ms']:.1f} ms", f"{h_ms:.1f} ms"],
        "Type":       ["Analytical", "Lattice", "Simulation", "Simulation", "Simulation", "Stoch Vol"],
    }
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="info-box">
        <b>Reading this table:</b> All methods price the same {option_type} option with
        S=${S}, K=${K}, T={T:.2f}yr, r={r:.1%}, σ={sigma:.1%}.
        The BS analytical price <b>${bs_price:.4f}</b> is the gold standard for European options.
        Heston (ρ={h_rho}, σᵥ={h_sigma_v}) adds stochastic volatility, producing a different price
        when vol-of-vol is non-zero.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GREEKS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Black-Scholes Greeks at Current Parameters</div>', unsafe_allow_html=True)

    # ── Current-params bar chart ────────────────────────────────────────────
    greek_names  = ["Δ Delta", "Γ Gamma", "V Vega", "Θ Theta", "ρ Rho"]
    call_vals = [c_delta, gamma, vega / 100, c_theta / 365, c_rho / 100]  # normalise for display
    put_vals  = [p_delta, gamma, vega / 100, p_theta / 365, p_rho / 100]
    greek_colors = [COLORS["delta"], COLORS["gamma"], COLORS["vega"], COLORS["theta"], COLORS["rho"]]

    col_g1, col_g2 = st.columns([1.2, 1])
    with col_g1:
        fig_greek_bar = go.Figure()
        fig_greek_bar.add_trace(go.Bar(
            name="Call", x=greek_names, y=call_vals,
            marker_color=[COLORS["bs"]] * 5,
            text=[f"{v:.4f}" for v in call_vals],
            textposition="outside", textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig_greek_bar.add_trace(go.Bar(
            name="Put", x=greek_names, y=put_vals,
            marker_color=[COLORS["mc"]] * 5,
            text=[f"{v:.4f}" for v in put_vals],
            textposition="outside", textfont=dict(family="JetBrains Mono", size=11),
        ))
        fig_greek_bar.update_layout(
            **PLOTLY_LAYOUT,
            title="Greeks — Call vs Put (normalised for scale)",
            barmode="group",
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                        bgcolor="rgba(22,27,34,0.8)", bordercolor="#30363d", borderwidth=1,
                        font=dict(color="#e6edf3", size=11)),
        )
        st.plotly_chart(fig_greek_bar, use_container_width=True)

    with col_g2:
        greek_rows = [
            ("Δ Delta (Call)",  f"{c_delta:.5f}", "Sensitivity to spot price — fraction of underlying"),
            ("Δ Delta (Put)",   f"{p_delta:.5f}", "Negative; ranges (−1, 0)"),
            ("Γ Gamma",         f"{gamma:.5f}",   "Convexity; rate of change of Delta"),
            ("V Vega",          f"{vega:.5f}",    "Sensitivity to 1-unit change in σ"),
            ("Θ Theta (Call)",  f"{c_theta:.5f}", "Time decay per year; always negative for calls"),
            ("Θ Theta (Put)",   f"{p_theta:.5f}", "Can be positive for deep ITM puts"),
            ("ρ Rho (Call)",    f"{c_rho:.5f}",   "Sensitivity to interest rate"),
            ("ρ Rho (Put)",     f"{p_rho:.5f}",   "Negative — puts lose value as rates rise"),
        ]
        st.markdown("""
        <style>.greek-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
        .greek-table th { color:#8b949e; font-weight:500; text-align:left; padding:6px 10px; border-bottom:1px solid #30363d; }
        .greek-table td { padding:7px 10px; border-bottom:1px solid rgba(48,54,61,0.5); }
        .greek-table tr:hover td { background:rgba(56,189,248,0.04); }
        .greek-val { font-family:'JetBrains Mono',monospace; color:#38bdf8; font-weight:600; }
        </style>
        <table class="greek-table">
          <tr><th>Greek</th><th>Value</th><th>Interpretation</th></tr>
        """, unsafe_allow_html=True)
        for name, val, interp in greek_rows:
            st.markdown(f"<tr><td>{name}</td><td class='greek-val'>{val}</td><td style='color:#8b949e'>{interp}</td></tr>",
                        unsafe_allow_html=True)
        st.markdown("</table>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Delta vs Spot ───────────────────────────────────────────────────────
    spot_range = np.linspace(max(10, S - 80), S + 80, 200)

    @st.cache_data(show_spinner=False)
    def greeks_vs_spot(K, T, r, sigma):
        sp = np.linspace(max(10, K - 80), K + 80, 200)
        cd, pd_, gm, vg, ct, pt = [], [], [], [], [], []
        for s in sp:
            _cd, _pd = black_scholes_delta(s, K, T, r, sigma)
            cd.append(_cd); pd_.append(_pd)
            gm.append(black_scholes_gamma(s, K, T, r, sigma))
            vg.append(black_scholes_vega(s, K, T, r, sigma))
            _ct, _pt = black_scholes_theta(s, K, T, r, sigma)
            ct.append(_ct); pt.append(_pt)
        return sp, cd, pd_, gm, vg, ct, pt

    sp, cd, pd_, gm, vg, ct, pt = greeks_vs_spot(K, T, r, sigma)

    fig_greeks = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Delta vs Spot", "Gamma vs Spot", "Vega vs Spot", "Theta vs Spot"],
        vertical_spacing=0.18, horizontal_spacing=0.12,
    )

    # Delta
    fig_greeks.add_trace(go.Scatter(x=sp, y=cd, name="Call Δ", line=dict(color=COLORS["bs"], width=2),
                                    hovertemplate="S=%{x:.1f}<br>Delta=%{y:.4f}<extra>Call</extra>"), row=1, col=1)
    fig_greeks.add_trace(go.Scatter(x=sp, y=pd_, name="Put Δ", line=dict(color=COLORS["mc"], width=2),
                                    hovertemplate="S=%{x:.1f}<br>Delta=%{y:.4f}<extra>Put</extra>"), row=1, col=1)
    # Gamma
    fig_greeks.add_trace(go.Scatter(x=sp, y=gm, name="Γ Gamma", line=dict(color=COLORS["gamma"], width=2),
                                    showlegend=False, hovertemplate="S=%{x:.1f}<br>Gamma=%{y:.5f}<extra></extra>"), row=1, col=2)
    # Vega
    fig_greeks.add_trace(go.Scatter(x=sp, y=vg, name="V Vega", line=dict(color=COLORS["vega"], width=2),
                                    showlegend=False, hovertemplate="S=%{x:.1f}<br>Vega=%{y:.4f}<extra></extra>"), row=2, col=1)
    # Theta
    fig_greeks.add_trace(go.Scatter(x=sp, y=ct, name="Θ Call", line=dict(color=COLORS["theta"], width=2),
                                    showlegend=False, hovertemplate="S=%{x:.1f}<br>Theta=%{y:.4f}<extra>Call</extra>"), row=2, col=2)
    fig_greeks.add_trace(go.Scatter(x=sp, y=pt, name="Θ Put", line=dict(color=COLORS["rho"], width=2),
                                    showlegend=False, hovertemplate="S=%{x:.1f}<br>Theta=%{y:.4f}<extra>Put</extra>"), row=2, col=2)

    for row, col in [(1,1),(1,2),(2,1),(2,2)]:
        fig_greeks.add_vline(x=K, line_dash="dot", line_color="#facc15", line_width=1.2, row=row, col=col)

    fig_greeks.update_xaxes(gridcolor="#30363d", linecolor="#30363d", tickfont=dict(color="#8b949e", size=10))
    fig_greeks.update_yaxes(gridcolor="#30363d", linecolor="#30363d", tickfont=dict(color="#8b949e", size=10))
    fig_greeks.update_annotations(font=dict(color="#8b949e", size=12))
    fig_greeks.update_layout(
        **PLOTLY_LAYOUT,
        height=560,
        title=dict(text=f"BS Greeks vs Spot  [K={K}, T={T:.2f}yr, r={r:.1%}, σ={sigma:.1%}]",
                   font=dict(size=14, color="#e6edf3"), x=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(22,27,34,0.8)", bordercolor="#30363d", borderwidth=1,
                    font=dict(color="#e6edf3", size=11)),
    )
    st.plotly_chart(fig_greeks, use_container_width=True)

    # ── Vega vs σ ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Vega vs Volatility  &  Delta vs Time to Expiry</div>',
                unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def vega_vs_sigma_data(S, K, T, r):
        sigs = np.linspace(0.05, 0.80, 200)
        return sigs, [black_scholes_vega(S, K, T, r, s) for s in sigs]

    @st.cache_data(show_spinner=False)
    def delta_vs_T_data(S, K, r, sigma):
        times = np.linspace(0.05, 3.0, 200)
        cd = [black_scholes_delta(S, K, t, r, sigma)[0] for t in times]
        pd_ = [black_scholes_delta(S, K, t, r, sigma)[1] for t in times]
        return times, cd, pd_

    sigs, vegas_curve = vega_vs_sigma_data(float(S), float(K), T, r)
    times_arr, cd_t, pd_t = delta_vs_T_data(float(S), float(K), r, sigma)

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_vega = go.Figure()
        fig_vega.add_trace(go.Scatter(x=sigs * 100, y=vegas_curve, mode="lines",
                                      line=dict(color=COLORS["vega"], width=2.5),
                                      fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
                                      hovertemplate="σ=%{x:.1f}%<br>Vega=%{y:.4f}<extra></extra>"))
        fig_vega.add_vline(x=sigma * 100, line_dash="dash", line_color="#facc15", line_width=1.5,
                           annotation_text=f"σ={sigma:.0%}", annotation_font=dict(color="#facc15"))
        fig_vega.update_layout(**PLOTLY_LAYOUT, title="Vega vs Volatility",
                               xaxis_title="Volatility σ (%)", yaxis_title="Vega", height=320)
        st.plotly_chart(fig_vega, use_container_width=True)

    with col_v2:
        fig_delta_t = go.Figure()
        fig_delta_t.add_trace(go.Scatter(x=times_arr, y=cd_t, name="Call Δ",
                                          line=dict(color=COLORS["bs"], width=2.5),
                                          hovertemplate="T=%{x:.2f}yr<br>Δ=%{y:.4f}<extra>Call</extra>"))
        fig_delta_t.add_trace(go.Scatter(x=times_arr, y=pd_t, name="Put Δ",
                                          line=dict(color=COLORS["mc"], width=2.5),
                                          hovertemplate="T=%{x:.2f}yr<br>Δ=%{y:.4f}<extra>Put</extra>"))
        fig_delta_t.add_vline(x=T, line_dash="dash", line_color="#facc15", line_width=1.5,
                               annotation_text=f"T={T:.2f}", annotation_font=dict(color="#facc15"))
        fig_delta_t.update_layout(**PLOTLY_LAYOUT, title="Delta vs Time to Expiry",
                                   xaxis_title="T (years)", yaxis_title="Delta", height=320,
                                   legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="#30363d",
                                               borderwidth=1, font=dict(color="#e6edf3", size=11)))
        st.plotly_chart(fig_delta_t, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VOL SMILE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Heston Stochastic Volatility — Implied Vol Smile vs Flat BS</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box" style="margin-bottom:20px;">
        <b>What this shows:</b> The Heston model (κ={h_kappa}, θ={h_theta:.2f}, σᵥ={h_sigma_v}, ρ={h_rho}, v₀={h_v0:.2f})
        generates <b>different prices for OTM/ITM options</b> versus Black-Scholes.
        Back-solving for the BS implied vol at each Heston price reveals the characteristic
        <b>volatility smile / skew</b> absent from the flat BS model.
        The negative correlation ρ={h_rho} produces left-skew — higher IV for OTM puts.
    </div>
    """, unsafe_allow_html=True)

    n_smile_paths = st.select_slider("Smile Paths (more = smoother, slower)",
                                     [500, 1000, 2000, 5000, 10_000], value=1000,
                                     format_func=lambda x: f"{x:,}",
                                     key="smile_paths")

    with st.spinner(f"Computing Heston smile ({n_smile_paths:,} paths × 12 strikes) …"):
        strikes_smile, h_ivs, bs_ivs = compute_heston_smile_cached(
            float(S), T, r, h_v0, h_kappa, h_theta, h_sigma_v, h_rho, n_smile_paths)

    # Filter valid
    valid = [(strikes_smile[i], h_ivs[i], bs_ivs[i]) for i in range(len(h_ivs)) if h_ivs[i] is not None]
    if valid:
        vK, vh, vbs = zip(*valid)
        moneyness_arr = [k / S for k in vK]

        fig_smile = make_subplots(rows=1, cols=2,
                                  subplot_titles=["Implied Volatility Smile", "Heston vs BS Call Prices"])

        # ── Panel 1: IV Smile ──────────────────────────────────────────────
        fig_smile.add_trace(go.Scatter(
            x=moneyness_arr, y=[iv * 100 for iv in vh],
            name=f"Heston  (κ={h_kappa}, σᵥ={h_sigma_v}, ρ={h_rho})",
            mode="lines+markers",
            line=dict(color=COLORS["heston"], width=2.5),
            marker=dict(size=8, symbol="circle"),
            fill="tonexty" if len(vbs) > 0 else None,
            hovertemplate="K/S=%{x:.3f}<br>IV=%{y:.2f}%<extra>Heston</extra>",
        ), row=1, col=1)

        fig_smile.add_trace(go.Scatter(
            x=moneyness_arr, y=[iv * 100 for iv in vbs],
            name=f"BS flat  (σ=√θ={math.sqrt(h_theta)*100:.0f}%)",
            mode="lines",
            line=dict(color=COLORS["bs"], width=2, dash="dash"),
            hovertemplate="K/S=%{x:.3f}<br>IV=%{y:.2f}%<extra>BS flat</extra>",
        ), row=1, col=1)

        # Fill between
        fig_smile.add_trace(go.Scatter(
            x=list(moneyness_arr) + list(moneyness_arr)[::-1],
            y=[iv * 100 for iv in vh] + [iv * 100 for iv in vbs][::-1],
            fill="toself", fillcolor="rgba(250,204,21,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

        fig_smile.add_vline(x=1.0, line_dash="dot", line_color="#8b949e", line_width=1,
                            annotation_text="ATM", annotation_font=dict(color="#8b949e", size=10),
                            row=1, col=1)

        # ── Panel 2: Prices ────────────────────────────────────────────────
        h_prices_smile  = [heston_mc_price(S, k, T, r, h_v0, h_kappa, h_theta, h_sigma_v, h_rho,
                                            n_paths=n_smile_paths, n_steps=63, seed=int(k))["price"]
                           for k in vK]
        bs_prices_smile = [black_scholes(S, k, T, r, math.sqrt(h_theta))[0] for k in vK]

        fig_smile.add_trace(go.Scatter(
            x=moneyness_arr, y=h_prices_smile, name="Heston price",
            mode="lines+markers", line=dict(color=COLORS["heston"], width=2.5),
            marker=dict(size=7),
            hovertemplate="K/S=%{x:.3f}<br>Price=$%{y:.4f}<extra>Heston</extra>",
        ), row=1, col=2)

        fig_smile.add_trace(go.Scatter(
            x=moneyness_arr, y=bs_prices_smile, name="BS price",
            mode="lines", line=dict(color=COLORS["bs"], width=2, dash="dash"),
            hovertemplate="K/S=%{x:.3f}<br>Price=$%{y:.4f}<extra>BS</extra>",
        ), row=1, col=2)

        fig_smile.add_trace(go.Scatter(
            x=list(moneyness_arr) + list(moneyness_arr)[::-1],
            y=h_prices_smile + bs_prices_smile[::-1],
            fill="toself", fillcolor="rgba(52,211,153,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=2)

        fig_smile.update_xaxes(gridcolor="#30363d", linecolor="#30363d", tickfont=dict(color="#8b949e", size=10))
        fig_smile.update_yaxes(gridcolor="#30363d", linecolor="#30363d", tickfont=dict(color="#8b949e", size=10))
        fig_smile.update_annotations(font=dict(color="#8b949e", size=12))
        fig_smile.update_layout(
            **PLOTLY_LAYOUT,
            height=460,
            title=dict(
                text=f"Heston Stochastic Vol  [S={S}, T={T:.1f}yr, r={r:.1%}, v₀={h_v0:.2f}, {n_smile_paths:,} paths]",
                font=dict(size=14, color="#e6edf3"), x=0,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(22,27,34,0.8)", bordercolor="#30363d", borderwidth=1,
                        font=dict(color="#e6edf3", size=11)),
        )
        st.plotly_chart(fig_smile, use_container_width=True)

        # ── ATM vs wing stats ──────────────────────────────────────────────
        atm_idx  = min(range(len(moneyness_arr)), key=lambda i: abs(moneyness_arr[i] - 1.0))
        wing_idx = 0  # leftmost (OTM put wing)
        atm_iv   = vh[atm_idx] * 100
        wing_iv  = vh[wing_idx] * 100
        skew     = wing_iv - atm_iv

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ATM IV (Heston)", f"{atm_iv:.2f}%", delta=f"{atm_iv - math.sqrt(h_theta)*100:+.2f}% vs flat BS")
        c2.metric("OTM Put Wing IV", f"{wing_iv:.2f}%", delta=f"K/S={moneyness_arr[wing_idx]:.2f}")
        c3.metric("Skew (Wing − ATM)", f"{skew:+.2f}%", delta="left skew from ρ<0" if h_rho < 0 else "right skew")
        c4.metric("BS Flat IV", f"{math.sqrt(h_theta)*100:.1f}%", delta="constant across strikes")
    else:
        st.warning("Could not extract implied vols for these parameters. Try lower σᵥ or fewer paths.")

    # ── IV sensitivity ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Smile Shape vs ρ (spot-vol correlation)</div>',
                unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def rho_sensitivity(S, T, r, v0, kappa, theta, sigma_v, n_paths):
        rhos = [-0.8, -0.4, 0.0, 0.4, 0.8]
        result = {}
        moneyness_grid = np.linspace(0.85, 1.15, 8)
        Ks = (S * moneyness_grid).tolist()
        for rho_val in rhos:
            ivs = []
            for i, K_val in enumerate(Ks):
                res = heston_mc_price(S, K_val, T, r, v0, kappa, theta, sigma_v, rho_val,
                                      n_paths=n_paths, n_steps=63, seed=i+200)
                iv = implied_vol(res["price"], S, K_val, T, r, "call")
                ivs.append(iv * 100 if iv else None)
            result[rho_val] = (moneyness_grid.tolist(), ivs)
        return result

    rho_paths = min(n_smile_paths, 1000)
    with st.spinner("Computing ρ sensitivity…"):
        rho_data = rho_sensitivity(float(S), T, r, h_v0, h_kappa, h_theta, h_sigma_v, rho_paths)

    fig_rho = go.Figure()
    rho_colors_list = ["#f87171", "#fb923c", "#facc15", "#34d399", "#38bdf8"]
    for (rho_val, (mono, ivs)), col in zip(rho_data.items(), rho_colors_list):
        valid_mono = [mono[i] for i, iv in enumerate(ivs) if iv is not None]
        valid_iv   = [iv for iv in ivs if iv is not None]
        fig_rho.add_trace(go.Scatter(
            x=valid_mono, y=valid_iv, name=f"ρ = {rho_val:+.1f}",
            mode="lines+markers", line=dict(color=col, width=2), marker=dict(size=7),
            hovertemplate=f"K/S=%{{x:.3f}}<br>IV=%{{y:.2f}}%<extra>ρ={rho_val:+.1f}</extra>",
        ))
    fig_rho.add_vline(x=1.0, line_dash="dot", line_color="#8b949e", line_width=1)
    fig_rho.update_layout(
        **PLOTLY_LAYOUT,
        title="Heston IV Smile Shape for Different ρ Values",
        xaxis_title="Moneyness  K/S",
        yaxis_title="Implied Volatility (%)",
        height=360,
        legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="#30363d", borderwidth=1,
                    font=dict(color="#e6edf3", size=11)),
    )
    st.plotly_chart(fig_rho, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">SPY Test-Set Results  (200 options · Market Prices vs Model Prices)</div>',
                unsafe_allow_html=True)

    # ── Load CSV ────────────────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), "comparison_table.csv")
    if os.path.exists(csv_path):
        df_cmp = pd.read_csv(csv_path)
    else:
        df_cmp = pd.DataFrame({
            "Method": ["Black-Scholes", "CRR Binomial N=200", "Monte Carlo 100k",
                       "LSTM-BS Hybrid", "MLP Pricer", "VAE-IV → BS", "Heston MC"],
            "MAE (all)": [0.9058, 0.9053, 0.9038, 1.0477, 6.6577, 2.4755, 1.0402],
            "RMSE": [1.4442, 1.4431, 1.4459, 1.4902, 7.3106, 3.1479, 1.5781],
            "% within 5%": [49.0, 47.5, 50.5, 41.0, 12.5, 25.5, 38.5],
            "MAE ATM": [0.3790, 0.3793, 0.3808, 1.2547, 10.9255, 4.2988, 0.4550],
            "MAE OTM": [0.0193, 0.0198, 0.0199, 0.0413, 5.1214, 0.9144, 0.1459],
            "Speed ms/opt": [0.0096, 13.55, 1.23, 0.0094, 0.486, 0.0347, 18.21],
            "Best Use": ["Fast European baseline", "American/early-exercise",
                         "Exotic/path-dependent", "Time-series vol", "Ultra-fast batch",
                         "IV surface interpolation", "Stochastic vol/smile"],
        })

    # ── Colour-coded dataframe ──────────────────────────────────────────────
    numeric_cols = [c for c in df_cmp.columns if df_cmp[c].dtype in [float, "float64"]]

    def highlight_table(row):
        styles = [""] * len(row)
        mae_col = df_cmp.columns.get_loc("MAE (all)") if "MAE (all)" in df_cmp.columns else None
        pct_col = df_cmp.columns.get_loc("% within 5%") if "% within 5%" in df_cmp.columns else None
        if mae_col is not None:
            mae_val = row.iloc[mae_col]
            min_mae = df_cmp["MAE (all)"].min()
            if mae_val == min_mae:
                styles[mae_col] = "background-color: rgba(52,211,153,0.15); color: #34d399; font-weight:600;"
            elif mae_val == df_cmp["MAE (all)"].max():
                styles[mae_col] = "background-color: rgba(248,113,113,0.12); color: #f87171;"
        if pct_col is not None:
            pct_val = row.iloc[pct_col]
            if pct_val == df_cmp["% within 5%"].max():
                styles[pct_col] = "background-color: rgba(52,211,153,0.15); color: #34d399; font-weight:600;"
        return styles

    styled = df_cmp.style.apply(highlight_table, axis=1)\
        .format({c: "{:.4f}" for c in numeric_cols if c != "% within 5%"})\
        .format({"% within 5%": "{:.1f}%"} if "% within 5%" in df_cmp.columns else {})\
        .set_properties(**{"font-family": "JetBrains Mono, monospace", "font-size": "12px"})

    st.dataframe(styled, use_container_width=True, hide_index=True, height=320)

    st.markdown(f"""
    <div class="info-box">
        <b>Colour coding:</b>
        <span style="color:#34d399">■ Green</span> = best in column &nbsp;
        <span style="color:#f87171">■ Red</span> = worst in column.
        All errors are in <b>USD ($)</b> versus SPY market mid-prices.
        Speed is <b>ms per option</b> on the evaluation machine.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── MAE bar chart ───────────────────────────────────────────────────────
    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        fig_mae = go.Figure()
        mae_col_name = "MAE (all)" if "MAE (all)" in df_cmp.columns else df_cmp.columns[1]
        bar_colors = [COLORS["bs"], COLORS["crr"], COLORS["mc"], COLORS["anti"],
                      COLORS["mlp"], COLORS["cv"], COLORS["heston"]][:len(df_cmp)]
        fig_mae.add_trace(go.Bar(
            x=df_cmp["Method"], y=df_cmp[mae_col_name],
            marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)")),
            text=[f"${v:.4f}" for v in df_cmp[mae_col_name]],
            textposition="outside", textfont=dict(family="JetBrains Mono", size=10, color="#e6edf3"),
            hovertemplate="<b>%{x}</b><br>MAE=$%{y:.4f}<extra></extra>",
        ))
        fig_mae.update_layout(
            **PLOTLY_LAYOUT,
            title="MAE vs Market Prices ($)",
            xaxis_title="", yaxis_title="MAE ($)", height=380,
            xaxis_tickangle=-30, showlegend=False, bargap=0.3,
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    with col_ch2:
        if "% within 5%" in df_cmp.columns:
            fig_pct = go.Figure()
            fig_pct.add_trace(go.Bar(
                x=df_cmp["Method"], y=df_cmp["% within 5%"],
                marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.1f}%" for v in df_cmp["% within 5%"]],
                textposition="outside", textfont=dict(family="JetBrains Mono", size=10, color="#e6edf3"),
                hovertemplate="<b>%{x}</b><br>Within 5%%: %{y:.1f}%<extra></extra>",
            ))
            fig_pct.update_layout(
                **PLOTLY_LAYOUT,
                title="% Options Priced Within 5% of Market",
                xaxis_title="", yaxis_title="% within 5%", height=380,
                xaxis_tickangle=-30, showlegend=False, bargap=0.3,
            )
            st.plotly_chart(fig_pct, use_container_width=True)

    # ── Radar chart ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Radar — Multi-Metric Model Profile</div>', unsafe_allow_html=True)

    radar_methods = df_cmp["Method"].tolist()
    if all(c in df_cmp.columns for c in ["MAE (all)", "RMSE", "% within 5%", "MAE ATM"]):
        max_mae   = df_cmp["MAE (all)"].max()
        max_rmse  = df_cmp["RMSE"].max()
        max_pct   = df_cmp["% within 5%"].max()
        max_atm   = df_cmp["MAE ATM"].max()
        max_speed = df_cmp["Speed ms/opt"].max() if "Speed ms/opt" in df_cmp.columns else 1

        fig_radar = go.Figure()
        categories = ["Low MAE", "Low RMSE", "% Within 5%", "Low ATM err", "Speed"]
        for i, (_, row) in enumerate(df_cmp.iterrows()):
            scores = [
                1 - row["MAE (all)"]  / max_mae,
                1 - row["RMSE"]       / max_rmse,
                row["% within 5%"]    / max_pct,
                1 - row["MAE ATM"]    / max_atm,
                1 - (row["Speed ms/opt"] / max_speed) if "Speed ms/opt" in df_cmp.columns else 0.5,
            ]
            color = bar_colors[i % len(bar_colors)]
            fig_radar.add_trace(go.Scatterpolar(
                r=scores + [scores[0]], theta=categories + [categories[0]],
                name=row["Method"],
                line=dict(color=color, width=2),
                fill="toself", fillcolor=color.replace(")", ",0.07)").replace("rgb", "rgba") if color.startswith("rgb") else color + "12",
                hovertemplate="<b>" + row["Method"] + "</b><br>%{theta}: %{r:.3f}<extra></extra>",
            ))

        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            polar=dict(
                bgcolor="#161b22",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363d",
                                tickfont=dict(color="#8b949e", size=9), linecolor="#30363d"),
                angularaxis=dict(gridcolor="#30363d", linecolor="#30363d",
                                 tickfont=dict(color="#e6edf3", size=12)),
            ),
            title=dict(text="Normalised Model Performance (higher = better)", font=dict(size=14, color="#e6edf3"), x=0),
            height=480,
            legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="#30363d", borderwidth=1,
                        font=dict(color="#e6edf3", size=11), orientation="v"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Speed vs accuracy scatter ───────────────────────────────────────────
    if "Speed ms/opt" in df_cmp.columns:
        st.markdown('<div class="section-header">Speed vs Accuracy Trade-off</div>', unsafe_allow_html=True)
        fig_scatter = go.Figure()
        for i, (_, row) in enumerate(df_cmp.iterrows()):
            color = bar_colors[i % len(bar_colors)]
            fig_scatter.add_trace(go.Scatter(
                x=[row["Speed ms/opt"]], y=[row["MAE (all)"]],
                mode="markers+text",
                name=row["Method"],
                marker=dict(size=18, color=color, line=dict(width=2, color="rgba(255,255,255,0.3)")),
                text=[row["Method"].split()[0]],
                textposition="top center",
                textfont=dict(color=color, size=10, family="Inter"),
                hovertemplate=f"<b>{row['Method']}</b><br>Speed: %{{x:.3f}} ms/opt<br>MAE: $%{{y:.4f}}<extra></extra>",
            ))
        fig_scatter.update_layout(
            **PLOTLY_LAYOUT,
            title="Speed vs MAE — Pareto Front",
            xaxis_title="Speed (ms per option)  →  Faster is right",
            yaxis_title="MAE ($)  →  More accurate is bottom",
            xaxis=dict(type="log", gridcolor="#30363d", linecolor="#30363d", tickfont=dict(color="#8b949e", size=10)),
            yaxis=dict(gridcolor="#30363d", linecolor="#30363d", tickfont=dict(color="#8b949e", size=10)),
            height=400, showlegend=False,
        )
        # Annotate ideal
        fig_scatter.add_annotation(x=0, y=0, text="Ideal →", showarrow=False,
                                   font=dict(color="#34d399", size=11), xref="paper", yref="paper")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("""
        <div class="info-box">
            <b>Pareto-optimal models</b> sit in the <span style="color:#34d399">bottom-left</span> corner (fast AND accurate).
            Black-Scholes and CRR dominate on speed; Monte Carlo offers the best accuracy among simulation methods;
            MLP is extremely fast but sacrifices accuracy due to OOD generalisation limits.
            Heston trades speed for smile-consistent pricing.
        </div>
        """, unsafe_allow_html=True)
