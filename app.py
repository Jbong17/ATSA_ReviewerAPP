"""
ATSA Comprehensive Exam Part 1 — Interactive Reviewer (Streamlit / PWA-ready)

A mobile-friendly companion app for the BSDS-edition reviewer.
Provides interactive simulations for the visual concepts that benefit
most from live data: stationarity, ACF/PACF identification, differencing,
unit-circle root location, forecasting, and spectral analysis.

Author     : Prepared for J. B. Bongalos
Exam date  : 02 May 2026, 12:30–16:30
Run locally:  streamlit run app.py
"""

# ============================================================================
# Imports
# ============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller, kpss

# ============================================================================
# Page config (must be the FIRST Streamlit call)
# ============================================================================
st.set_page_config(
    page_title="ATSA Reviewer",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        "About": (
            "ATSA Comp Exam · Part 1 (Closed-Book, 25%) · BSDS Edition\n\n"
            "Interactive companion with live data simulations."
        )
    },
)

# ============================================================================
# Theme — match the printed reviewer's editorial aesthetic
# ============================================================================
COL_BG     = "#f7f3ec"   # warm cream
COL_CARD   = "#fffaf0"   # lighter card
COL_INK    = "#1c1814"   # near-black ink
COL_MUTED  = "#6b6259"
COL_RULE   = "#d9cfc0"
COL_ACCENT = "#8b1d1d"   # oxblood
COL_SOFT   = "#b8553a"   # terracotta
COL_GREEN  = "#7a9466"
COL_BLUE   = "#6b8caf"
COL_YELLOW = "#d4a373"

# Matplotlib style block — applied to every figure
mpl.rcParams.update({
    "axes.facecolor": COL_BG,
    "figure.facecolor": COL_BG,
    "savefig.facecolor": COL_BG,
    "axes.edgecolor": COL_INK,
    "axes.labelcolor": COL_INK,
    "axes.titlecolor": COL_INK,
    "text.color": COL_INK,
    "xtick.color": COL_INK,
    "ytick.color": COL_INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": COL_RULE,
    "grid.alpha": 0.6,
    "grid.linewidth": 0.5,
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
})

# ============================================================================
# PWA / mobile meta tags (best effort — works for most "Add to Home Screen")
# ============================================================================
st.markdown(
    """
<meta name="theme-color" content="#8b1d1d">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="ATSA Reviewer">
<meta name="mobile-web-app-capable" content="yes">
<meta name="application-name" content="ATSA Reviewer">
""",
    unsafe_allow_html=True,
)

# ============================================================================
# Custom CSS — typography + colored callouts
# ============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Crimson+Pro:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, .stTabs {
    font-family: 'Crimson Pro', Georgia, serif !important;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Fraunces', Georgia, serif !important;
    letter-spacing: -0.005em;
}
h1 { font-weight: 600; }
h2, h3 { font-weight: 600; }

/* App background */
.stApp { background: #f7f3ec; }

/* Sidebar tweaks */
section[data-testid="stSidebar"] {
    background: #fffaf0;
    border-right: 1px solid #d9cfc0;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #8b1d1d;
}

/* Eyebrow / section number */
.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #8b1d1d;
    margin-bottom: 4px;
}

/* Tier pill */
.pill {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 8px;
    background: #8b1d1d;
    color: #f7f3ec;
    border-radius: 2px;
    margin-left: 6px;
    vertical-align: middle;
}

/* Hide the Streamlit hamburger and footer for cleaner mobile view */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Tighter mobile spacing */
.block-container { padding-top: 1.5rem; padding-bottom: 4rem; }
@media (max-width: 600px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    h1 { font-size: 1.7rem !important; }
    h2 { font-size: 1.35rem !important; }
}

/* Math styling */
.katex { font-size: 1.02em; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# Reusable helpers
# ============================================================================
CALLOUT_STYLES = {
    "def":   ("#fdf6dd", "#d4a373", "#8a6a1a"),
    "thm":   ("#e8eef5", "#6b8caf", "#2c4a6e"),
    "plain": ("#f0e6f0", "#8a5a8a", "#6a3f6a"),
    "key":   ("#e6efe1", "#7a9466", "#4a6638"),
    "warn":  ("#fbe8e0", "#c0613f", "#8a3a1c"),
}


def callout(kind: str, label: str, body_md: str):
    """Render a colored callout box. body_md is rendered as markdown.

    kind ∈ {def, thm, plain, key, warn}
    """
    bg, border, txt = CALLOUT_STYLES.get(kind, CALLOUT_STYLES["plain"])
    st.markdown(
        f"""
<div style="background:{bg};border-left:3px solid {border};padding:14px 18px;margin:14px 0 18px;border-radius:2px;">
<div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;letter-spacing:0.18em;
            text-transform:uppercase;color:{txt};margin-bottom:6px;font-weight:500;">{label}</div>
<div style="font-family:'Crimson Pro',Georgia,serif;font-size:16px;line-height:1.55;">{body_md}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def section_header(num: str, title: str, lede: str, tier: str = "T1"):
    """Render a consistent section header."""
    st.markdown(
        f"""
<div class="eyebrow">Section {num}</div>
<h1 style="margin-top:0;margin-bottom:6px;">{title} <span class="pill">{tier}</span></h1>
<p style="font-style:italic;color:{COL_MUTED};margin:0 0 22px;border-bottom:1px solid {COL_RULE};
          padding-bottom:14px;">{lede}</p>
""",
        unsafe_allow_html=True,
    )


# ----- Data generators ------------------------------------------------------
@st.cache_data
def gen_ar1(phi: float, n: int = 400, seed: int = 42) -> np.ndarray:
    """Generate an AR(1) realization. φ controls persistence."""
    rng = np.random.default_rng(seed)
    if abs(phi) >= 1:
        # Use direct recursion so we can also generate explosive/random walk
        x = np.zeros(n)
        z = rng.standard_normal(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + z[t]
        return x
    ar = ArmaProcess(np.r_[1, -phi], np.r_[1])
    return ar.generate_sample(nsample=n, distrvs=rng.standard_normal)


@st.cache_data
def gen_arma(ar_coefs: tuple, ma_coefs: tuple, n: int = 400, seed: int = 42) -> np.ndarray:
    """Generate ARMA(p, q). Coefficients given in 'natural' form: 
    X_t = ar_coefs[0]·X_{t-1} + ... + Z_t + ma_coefs[0]·Z_{t-1} + ..."""
    rng = np.random.default_rng(seed)
    ar = np.r_[1, -np.array(ar_coefs)]      # statsmodels uses signs flipped
    ma = np.r_[1, np.array(ma_coefs)]
    proc = ArmaProcess(ar, ma)
    return proc.generate_sample(nsample=n, distrvs=rng.standard_normal)


@st.cache_data
def gen_random_walk(n: int = 400, seed: int = 42, drift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(drift + rng.standard_normal(n))


@st.cache_data
def gen_trend_stationary(n: int = 400, seed: int = 42, slope: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return slope * np.arange(n) + rng.standard_normal(n)


# ----- Test runner ----------------------------------------------------------
def run_tests(x: np.ndarray) -> pd.DataFrame:
    """Run ADF and KPSS, return a small results table."""
    rows = []
    try:
        adf = adfuller(x, autolag="AIC")
        rows.append(("ADF", "Unit root (non-stationary)",
                     f"{adf[0]:.3f}", f"{adf[1]:.3f}",
                     "✓ stationary" if adf[1] < 0.05 else "✗ cannot reject unit root"))
    except Exception as e:
        rows.append(("ADF", "—", "err", "err", str(e)))
    try:
        kp = kpss(x, regression="c", nlags="auto")
        rows.append(("KPSS", "Stationary (level)",
                     f"{kp[0]:.3f}", f"{kp[1]:.3f}",
                     "✗ rejects stationarity" if kp[1] < 0.05 else "✓ cannot reject stationarity"))
    except Exception as e:
        rows.append(("KPSS", "—", "err", "err", str(e)))
    return pd.DataFrame(rows, columns=["Test", "Null H₀", "Statistic", "p-value", "Verdict"])


# ----- Plot helpers ---------------------------------------------------------
def plot_series(x: np.ndarray, title: str = "", color=None, mean_line=True, ax=None):
    """Plot a single series with optional mean overlay."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 2.6))
    else:
        fig = ax.figure
    ax.plot(x, color=color or COL_INK, linewidth=0.9)
    if mean_line:
        ax.axhline(np.mean(x), color=COL_ACCENT, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(title, loc="left", color=COL_INK)
    ax.set_xlabel("time t")
    fig.tight_layout()
    return fig


def plot_rolling_stats(x: np.ndarray, window: int, title: str = "") -> plt.Figure:
    """Plot series with rolling mean and ±1 rolling std envelope."""
    s = pd.Series(x)
    rmean = s.rolling(window).mean()
    rstd  = s.rolling(window).std()

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(x, color=COL_INK, linewidth=0.8, alpha=0.85, label="series")
    ax.plot(rmean, color=COL_ACCENT, linewidth=1.3, label=f"rolling mean (w={window})")
    ax.fill_between(s.index, rmean - rstd, rmean + rstd,
                    color=COL_SOFT, alpha=0.18, label="±1 rolling std")
    ax.set_title(title, loc="left")
    ax.set_xlabel("time t")
    ax.legend(loc="upper left", framealpha=0)
    fig.tight_layout()
    return fig


def plot_acf_pacf_pair(x: np.ndarray, lags: int = 30) -> plt.Figure:
    """Side-by-side ACF and PACF with shared y-limits."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    plot_acf(x, lags=lags, ax=axes[0], alpha=0.05, color=COL_INK,
             vlines_kwargs={"colors": COL_INK})
    plot_pacf(x, lags=lags, ax=axes[1], alpha=0.05, method="ywm", color=COL_INK,
              vlines_kwargs={"colors": COL_INK})
    axes[0].set_title("ACF", loc="left")
    axes[1].set_title("PACF", loc="left")
    for a in axes:
        a.set_ylim(-1.05, 1.05)
        a.set_xlabel("lag")
    fig.tight_layout()
    return fig


def plot_unit_circle(roots: list[complex], title: str = "Roots in the complex plane") -> plt.Figure:
    """Visualize roots relative to the unit circle."""
    fig, ax = plt.subplots(figsize=(5, 5))
    # forbidden region
    forbidden = Circle((0, 0), 1.0, color=COL_SOFT, alpha=0.15, zorder=0)
    ax.add_patch(forbidden)
    # unit circle outline
    theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(np.cos(theta), np.sin(theta), color=COL_ACCENT, linewidth=1.6)
    # axes
    ax.axhline(0, color=COL_INK, linewidth=0.7)
    ax.axvline(0, color=COL_INK, linewidth=0.7)
    # plot roots
    for r in roots:
        good = abs(r) > 1
        ax.plot(r.real, r.imag, "o",
                color=COL_GREEN if good else COL_ACCENT,
                markersize=10, markeredgecolor=COL_INK, markeredgewidth=0.8)
        ax.annotate(f"|z|={abs(r):.2f}",
                    (r.real, r.imag), xytext=(8, 8),
                    textcoords="offset points", fontsize=9,
                    color=COL_GREEN if good else COL_ACCENT, fontweight="bold")
    # cosmetics
    lim = max(2.0, max([abs(r) for r in roots] + [1.5]) * 1.15)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(z)"); ax.set_ylabel("Im(z)")
    ax.set_title(title, loc="left")
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_GREEN,
               markeredgecolor=COL_INK, markersize=10, label="|z| > 1  ✓ causal/invertible"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_ACCENT,
               markeredgecolor=COL_INK, markersize=10, label="|z| ≤ 1  ✗ violates"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=8.5,
              framealpha=0.85, edgecolor=COL_RULE)
    fig.tight_layout()
    return fig


def plot_periodogram(x: np.ndarray, fs: float = 1.0) -> plt.Figure:
    """Plot the raw periodogram and a smoothed estimate."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fx = np.fft.rfft(x - np.mean(x))
    pgram = (np.abs(fx) ** 2) / n

    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
    axes[0].plot(np.arange(n), x, color=COL_INK, linewidth=0.7)
    axes[0].set_title("series", loc="left"); axes[0].set_xlabel("")

    axes[1].plot(freqs, pgram, color=COL_INK, linewidth=0.6, alpha=0.55, label="raw")
    # Daniell-style smooth (rolling mean window 7)
    if len(pgram) > 9:
        smooth = pd.Series(pgram).rolling(7, center=True, min_periods=1).mean()
        axes[1].plot(freqs, smooth, color=COL_ACCENT, linewidth=1.4, label="smoothed (Daniell)")
    axes[1].set_title("periodogram I(ω)", loc="left")
    axes[1].set_xlabel("frequency (cycles per time step)")
    axes[1].legend(framealpha=0)
    fig.tight_layout()
    return fig


def plot_forecast(x: np.ndarray, model: str, horizon: int = 30, sigma: float = 1.0) -> plt.Figure:
    """Plot the series with point forecasts and 95% prediction intervals.
    Implements the analytic AR(1) and random-walk forecasts directly."""
    n = len(x)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(np.arange(n), x, color=COL_INK, linewidth=0.9, label="observed")

    fut_idx = np.arange(n, n + horizon)

    if model == "AR(1)":
        # Estimate phi quickly
        phi = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        mean_ = np.mean(x)
        forecasts = mean_ + (x[-1] - mean_) * phi ** np.arange(1, horizon + 1)
        # Variance growth
        psi = phi ** np.arange(horizon)
        var_h = sigma ** 2 * np.cumsum(psi ** 2)
        se = np.sqrt(var_h)
        ax.set_title(f"AR(1) forecast (estimated φ ≈ {phi:.2f}) — mean reverts", loc="left")
    else:  # Random walk
        forecasts = np.full(horizon, x[-1])
        var_h = sigma ** 2 * np.arange(1, horizon + 1)  # grows linearly
        se = np.sqrt(var_h)
        ax.set_title("Random-walk forecast — flat point, growing intervals", loc="left")

    ax.plot(fut_idx, forecasts, color=COL_ACCENT, linewidth=1.6, label="forecast")
    ax.fill_between(fut_idx, forecasts - 1.96 * se, forecasts + 1.96 * se,
                    color=COL_SOFT, alpha=0.25, label="95% PI")
    ax.axvline(n - 0.5, color=COL_MUTED, linestyle=":", linewidth=1)
    ax.set_xlabel("time t")
    ax.legend(framealpha=0, loc="upper left")
    fig.tight_layout()
    return fig


# ============================================================================
# Page: Home
# ============================================================================
def page_home():
    st.markdown('<div class="eyebrow">ATSA Comprehensive Exam · Part 1 of 2</div>',
                unsafe_allow_html=True)
    st.markdown("# Conceptual & Pipeline Fundamentals")
    st.markdown(
        f'<p style="font-style:italic;color:{COL_MUTED};font-size:18px;margin-top:0;">'
        "An interactive companion for the closed-book theory section.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Candidate:** Jerald B. Bongalos")
        st.markdown(f"**Exam date:** 02 May 2026, 12:30–16:30")
        st.markdown(f"**Duration:** 4 hours total")
    with col2:
        st.markdown(f"**Pass mark:** ≥ 80% accumulated")
        st.markdown(f"**Part 1 weight:** 25% · *closed everything*")
        st.markdown(f"**Part 2 weight:** 75% · *open everything*")

    callout(
        "key", "How to use this app",
        "Open one section per study block. Sections marked ★ have <strong>live data simulations</strong> "
        "you can play with — change a slider, watch the ACF and the test statistics change in real time. "
        "End every session with the <em>Cheat Sheet</em> for retention."
    )

    st.markdown("### Sections")
    rows = [
        ("A", "Stationarity ★",          "Three conditions, ergodicity, Wold theorem", "T1"),
        ("B", "Causality & Invertibility ★", "Unit-circle root condition, worked checks", "T1"),
        ("C", "Box–Jenkins Pipeline",    "The eight-step protocol with decisions",      "T1"),
        ("D", "ACF/PACF Identification ★","Read the plots, name the model",             "T1"),
        ("E", "Forecasting ★",            "MMSE, prediction intervals, validation",     "T2"),
        ("F", "VAR & Cointegration",     "Multivariate stability, VECM",                "T2"),
        ("G", "ARCH/GARCH",              "Volatility clustering",                        "T3"),
        ("H", "Spectral Analysis ★",     "Wiener–Khinchin, periodogram",                 "T2"),
        ("I", "Granger vs CCM",          "Linear-stochastic vs nonlinear-deterministic", "T2"),
        ("J", "Simplex & S-map",         "Empirical dynamic modeling",                  "T2"),
        ("K", "State Space & Kalman",    "Predict & update",                            "T3"),
        ("★", "Cheat Sheet",             "All must-knows, mobile-readable",              "—"),
        ("🎯","Recall Drill",            "Self-test flashcards",                         "—"),
    ]
    df = pd.DataFrame(rows, columns=["Sec", "Topic", "What it covers", "Tier"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    callout(
        "plain", "★ = interactive simulation",
        "Sections A, B, D, E, and H let you generate live time-series data, change parameters, "
        "and see ACF/PACF, unit-circle, periodogram, and forecast plots update on the spot. "
        "These are the most useful for building intuition the night before."
    )

    st.markdown("### Install on your phone")
    st.markdown(
        "1. Open this app's URL in your phone browser (Chrome on Android, Safari on iOS).\n"
        "2. **Android (Chrome):** menu → *Add to Home screen* → confirm.\n"
        "3. **iOS (Safari):** Share → *Add to Home Screen* → confirm.\n"
        "4. The app will open fullscreen-style from the new home-screen icon."
    )


# ============================================================================
# Page A: Stationarity (with multiple interactive demos)
# ============================================================================
def page_stationarity():
    section_header(
        "A", "Stationarity & Building Blocks",
        "The most important word in this course — and the assumption that nearly every "
        "ATSA tool requires. Let's build intuition with live data.",
        "T1",
    )

    tab_def, tab_compare, tab_three, tab_diff, tab_test = st.tabs([
        "📚 Definitions",
        "🆚 Stationary vs not",
        "🎯 Three conditions",
        "✂️ Differencing demo",
        "🧪 ADF & KPSS",
    ])

    # -------- DEFINITIONS TAB --------
    with tab_def:
        st.markdown("#### Stochastic process vs time series")
        st.markdown(
            "A **stochastic process** $\\{X_t : t \\in T\\}$ is a sequence of random variables "
            "indexed by time. A **time series** is one realization of such a process — the actual "
            "data we observe."
        )
        callout("plain", "In plain English",
                "The 'process' is the recipe; the 'time series' is one cake that came out of the oven. "
                "We taste the cake and try to figure out the recipe.")

        st.markdown("#### White noise WN(0, σ²)")
        st.latex(r"E[Z_t] = 0,\quad \mathrm{Var}(Z_t) = \sigma^2,\quad \mathrm{Cov}(Z_s, Z_t) = 0\ \text{for } s \neq t.")
        callout("plain", "Hierarchy (weakest to strongest)",
                "<strong>WN</strong> (uncorrelated) → <strong>Martingale difference</strong> "
                "(<em>E[Z<sub>t</sub> | ℱ<sub>t-1</sub>] = 0</em>) → <strong>IID</strong> → <strong>Gaussian WN</strong> (IID + Normal).")

        st.markdown("#### Strict vs. weak stationarity")
        callout("def", "Strict stationarity",
                "Joint distribution <em>(X<sub>t<sub>1</sub></sub>, …, X<sub>t<sub>k</sub></sub>) =<sup>d</sup> "
                "(X<sub>t<sub>1</sub>+h</sub>, …, X<sub>t<sub>k</sub>+h</sub>)</em> for every <em>k</em>, every shift <em>h</em>.")
        callout("def", "Weak (covariance) stationarity",
                "Three conditions all hold simultaneously: (1) constant mean <em>E[X<sub>t</sub>] = μ</em>, "
                "(2) finite constant variance <em>Var(X<sub>t</sub>) = γ(0) &lt; ∞</em>, "
                "(3) covariance depends only on the lag, "
                "<em>Cov(X<sub>t</sub>, X<sub>t+h</sub>) = γ(h)</em>.")
        callout("warn", "How they relate",
                "Strict + finite variance ⇒ weak. The reverse is not true in general. "
                "For Gaussian processes the two are equivalent.")

        st.markdown("#### Ergodicity")
        st.latex(r"\bar X_n = \frac{1}{n}\sum_{t=1}^n X_t \;\xrightarrow{n\to\infty}\; \mu.")
        callout("plain", "In plain English",
                "Watching one coin flipped many times tells you the same thing as watching one flip "
                "from each of many coins. Sufficient: <em>γ(h)→ 0</em> as <em>h→∞</em>.")
        callout("warn", "Stationary but NOT ergodic — classic example",
                "Draw <em>A ~ N(0,1)</em> once, set <em>X<sub>t</sub> = A</em> forever. Strictly stationary, but the "
                "sample mean equals <em>A</em> — converges to a random value, not to <em>μ = 0</em>.")

        st.markdown("#### Wold decomposition")
        st.latex(r"X_t = \sum_{j=0}^{\infty}\psi_j Z_{t-j} + V_t,\quad \sum\psi_j^2<\infty.")
        callout("plain", "Why Wold matters",
                "Every covariance-stationary process is, deep down, an MA(∞) of past shocks plus a "
                "deterministic part. ARMA exists because we can approximate that infinite sum with a "
                "few parameters. <em>ARMA is general because Wold is general.</em>")

    # -------- STATIONARY VS NOT TAB --------
    with tab_compare:
        st.markdown("#### Compare two series side by side")
        st.markdown("Pick parameters and watch how a stationary AR(1) "
                    "and a non-stationary random walk differ in mean, variance, and ACF.")

        c1, c2 = st.columns(2)
        with c1:
            phi = st.slider("AR(1) coefficient φ (stationary if |φ|<1)",
                            -0.95, 0.95, 0.7, 0.05, key="A_phi")
            n = st.slider("Sample size n", 100, 1500, 500, 50, key="A_n")
        with c2:
            seed = st.number_input("Random seed", value=42, key="A_seed")
            show_rolling = st.checkbox("Show rolling mean & ±1σ band", value=True, key="A_roll")

        x_stat = gen_ar1(phi, n=int(n), seed=int(seed))
        x_rw   = gen_random_walk(n=int(n), seed=int(seed))

        if show_rolling:
            st.pyplot(plot_rolling_stats(
                x_stat, window=max(20, int(n * 0.1)),
                title=f"Stationary AR(1) with φ = {phi}"))
            st.pyplot(plot_rolling_stats(
                x_rw, window=max(20, int(n * 0.1)),
                title="Non-stationary random walk"))
        else:
            st.pyplot(plot_series(x_stat, title=f"Stationary AR(1), φ = {phi}"))
            st.pyplot(plot_series(x_rw, title="Random walk"))

        st.markdown("#### Their ACFs tell the story")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f"**AR(1), φ = {phi}**")
            st.pyplot(plot_acf_pacf_pair(x_stat, lags=30))
        with c4:
            st.markdown("**Random walk**")
            st.pyplot(plot_acf_pacf_pair(x_rw, lags=30))

        callout("key", "What you should see",
                "The stationary AR(1)'s ACF decays geometrically toward zero. The random walk's ACF "
                "stays near 1 for many lags — the dependence never dies out. Slow ACF decay near 1 "
                "is the visual signature of an I(1) series.")

    # -------- THREE CONDITIONS TAB --------
    with tab_three:
        st.markdown("#### Check the three conditions on real series")
        choice = st.radio(
            "Which series to inspect?",
            ["Stationary AR(1) (φ = 0.6)", "Random walk", "Trend-stationary (slope = 0.05)",
             "Mean-shift (level break at midpoint)"],
            key="A3_choice",
        )
        n3 = st.slider("Sample size", 200, 1500, 600, 50, key="A3_n")

        if choice.startswith("Stationary"):
            x = gen_ar1(0.6, n=int(n3))
            verdict = "✓ All three conditions are satisfied."
        elif choice.startswith("Random walk"):
            x = gen_random_walk(n=int(n3))
            verdict = "✗ Variance grows with t (Var(X_t) = t·σ²) — fails condition 2."
        elif choice.startswith("Trend"):
            x = gen_trend_stationary(n=int(n3))
            verdict = "✗ Mean drifts linearly — fails condition 1."
        else:
            x = gen_ar1(0.4, n=int(n3))
            x[int(n3) // 2:] += 4.0
            verdict = "✗ Mean shifts at the break — fails condition 1."

        # Top: series with rolling stats
        win = max(20, int(n3 * 0.1))
        st.pyplot(plot_rolling_stats(x, window=win, title=choice))

        # Bottom: rolling mean and rolling std as separate diagnostics
        s = pd.Series(x)
        rmean = s.rolling(win).mean()
        rstd  = s.rolling(win).std()
        fig, axes = plt.subplots(1, 2, figsize=(8, 2.6))
        axes[0].plot(rmean, color=COL_ACCENT, linewidth=1.3)
        axes[0].axhline(np.mean(x), color=COL_INK, linestyle="--", linewidth=0.8, alpha=0.5)
        axes[0].set_title("Condition 1 — rolling mean", loc="left")
        axes[0].set_xlabel("time t")
        axes[1].plot(rstd, color=COL_GREEN, linewidth=1.3)
        axes[1].axhline(np.std(x), color=COL_INK, linestyle="--", linewidth=0.8, alpha=0.5)
        axes[1].set_title("Condition 2 — rolling std", loc="left")
        axes[1].set_xlabel("time t")
        fig.tight_layout()
        st.pyplot(fig)

        callout("key", "Verdict", verdict)

    # -------- DIFFERENCING TAB --------
    with tab_diff:
        st.markdown("#### Differencing: turn an I(1) series into a stationary one")
        c1, c2 = st.columns(2)
        with c1:
            d_choice = st.radio(
                "Source series",
                ["Random walk", "Random walk + drift", "AR(1), φ = 0.99 (near unit root)"],
                key="DIFF_choice",
            )
        with c2:
            d_n = st.slider("Sample size", 200, 1500, 500, 50, key="DIFF_n")

        if d_choice.startswith("Random walk +"):
            x = gen_random_walk(n=int(d_n), drift=0.1)
        elif d_choice.startswith("Random walk"):
            x = gen_random_walk(n=int(d_n))
        else:
            x = gen_ar1(0.99, n=int(d_n))
        dx = np.diff(x)

        st.pyplot(plot_series(x,  title=f"Original — {d_choice}"))
        st.pyplot(plot_series(dx, title="After first difference  ∇X_t = X_t − X_{t−1}"))

        st.markdown("#### What happened to the ACF?")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Original**")
            st.pyplot(plot_acf_pacf_pair(x, lags=30))
        with c4:
            st.markdown("**Differenced**")
            st.pyplot(plot_acf_pacf_pair(dx, lags=30))

        callout("key", "What you should see",
                "The original ACF stays high for many lags — slow decay near 1. After differencing, "
                "the ACF collapses to white-noise-like values. The unit root has been removed.")

    # -------- TESTS TAB --------
    with tab_test:
        st.markdown("#### Run ADF and KPSS on a chosen series")
        t_choice = st.selectbox(
            "Pick a series",
            [
                "Stationary AR(1), φ = 0.6",
                "Stationary AR(1), φ = 0.95 (near unit root)",
                "Random walk",
                "Random walk, then differenced",
                "Trend-stationary",
                "White noise",
            ],
            key="T_choice",
        )
        t_n = st.slider("Sample size", 200, 1500, 500, 50, key="T_n")

        if "0.6" in t_choice:
            x = gen_ar1(0.6, n=int(t_n))
        elif "0.95" in t_choice:
            x = gen_ar1(0.95, n=int(t_n))
        elif "differenced" in t_choice:
            x = np.diff(gen_random_walk(n=int(t_n)))
        elif "Random walk" in t_choice:
            x = gen_random_walk(n=int(t_n))
        elif "Trend" in t_choice:
            x = gen_trend_stationary(n=int(t_n))
        else:
            x = np.random.default_rng(42).standard_normal(int(t_n))

        st.pyplot(plot_series(x, title=t_choice))
        st.dataframe(run_tests(x), use_container_width=True, hide_index=True)

        callout("plain", "How to read the table",
                "ADF rejects ⇒ stationary. KPSS rejects ⇒ non-stationary. They have <em>opposite</em> "
                "nulls — so for a clean stationary series you want ADF to reject AND KPSS to NOT reject.")


# ============================================================================
# Page B: Causality & Invertibility (interactive unit circle)
# ============================================================================
def page_causality():
    section_header(
        "B", "Causality & Invertibility",
        "Two conditions on the roots of the AR and MA polynomials. Probably the single most "
        "computable closed-book question.",
        "T1",
    )

    tab_def, tab_unit, tab_examples = st.tabs([
        "📚 Definitions", "🎯 Live unit-circle demo", "✏️ Worked examples",
    ])

    with tab_def:
        st.markdown("#### The ARMA(p, q) model")
        st.latex(r"\phi(B)\, X_t = \theta(B)\, Z_t,\quad \{Z_t\}\sim\text{WN}(0,\sigma^2),")
        st.latex(r"\phi(B) = 1-\phi_1 B-\cdots-\phi_p B^p,\quad \theta(B) = 1+\theta_1 B+\cdots+\theta_q B^q.")

        callout("def", "Causality",
                "The process can be written as <em>X<sub>t</sub> = Σ<sub>j=0</sub>^∞ ψ<sub>j</sub> Z<sub>t-j</sub></em> with "
                "<em>Σ |ψ<sub>j</sub>| < ∞</em> — present and past innovations only.")
        callout("plain", "In plain English — causality",
                "Today's value depends only on shocks that have already happened. It cannot depend on "
                "future shocks. We don't have access to the future when we forecast.")

        callout("def", "Invertibility",
                "<em>Z<sub>t</sub> = Σ<sub>j=0</sub><sup>∞</sup> π<sub>j</sub> X<sub>t-j</sub></em> with <em>Σ |π<sub>j</sub>| &lt; ∞</em> — innovations "
                "recoverable from past observations.")
        callout("plain", "In plain English — invertibility",
                "Looking at the data, you can work backwards and figure out what the random shocks were. "
                "Without it, the model is unusable for forecasting.")

        st.markdown("#### The root condition")
        callout("thm", "Causality",
                "An ARMA process is causal iff every root of <em>φ(z)=0</em> has <em>|z| > 1</em> — outside the unit circle.")
        callout("thm", "Invertibility",
                "Same picture for <em>θ(z)</em>: every root of <em>θ(z)=0</em> must have <em>|z|>1</em>.")

        st.markdown("#### Why outside the unit circle?")
        st.markdown(
            "For an AR(1), $\\phi(B) = 1 - \\phi B$. To rewrite as a one-sided sum we invert via "
            "the geometric series:"
        )
        st.latex(r"\frac{1}{1-\phi B} = \sum_{j=0}^{\infty}\phi^j B^j,")
        st.markdown(
            "which converges only when $|\\phi|<1$, i.e. when the root $z=1/\\phi$ has $|z|>1$. "
            "If the root were inside, the series would blow up — no causal MA(∞) representation exists."
        )

    with tab_unit:
        st.markdown("#### Move the AR(1) coefficient and watch the root")
        st.markdown(
            "For AR(1) the polynomial is $\\phi(z) = 1 - \\phi_1 z$, so the single root is $z = 1/\\phi_1$. "
            "Drag the slider through and through 1 to see the boundary."
        )

        phi1 = st.slider("φ₁ (AR(1) coefficient)", -1.5, 1.5, 0.7, 0.05, key="B_phi1")
        # Avoid divide-by-zero
        root = complex(1.0 / phi1) if abs(phi1) > 1e-3 else complex(1e6)

        c1, c2 = st.columns([1.1, 1.0])
        with c1:
            st.pyplot(plot_unit_circle([root],
                                       title=f"AR(1) root — φ₁ = {phi1}"))
        with c2:
            st.markdown(f"**Root location:** $z = 1/\\phi_1 = {root.real:.3f}$")
            st.markdown(f"**$|z|$:** {abs(root):.3f}")
            if abs(root) > 1:
                st.markdown(f"<span style='color:{COL_GREEN};font-weight:600'>✓ Causal</span>",
                            unsafe_allow_html=True)
            elif abs(root) == 1:
                st.markdown(f"<span style='color:{COL_ACCENT};font-weight:600'>= unit root (boundary, non-stationary)</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:{COL_ACCENT};font-weight:600'>✗ Non-causal (explosive)</span>",
                            unsafe_allow_html=True)
            st.markdown("---")
            x = gen_ar1(phi1, n=400)
            st.pyplot(plot_series(x, title="Simulated path", mean_line=False))

        callout("key", "What to notice",
                "When |φ₁| < 1 the root sits outside the unit circle (green) and the series wobbles around a "
                "constant level. When |φ₁| crosses 1 the root moves inside the circle (red) and the series "
                "explodes. At φ₁ = 1 you get the random walk — boundary case, non-stationary.")

        st.markdown("#### MA(1) version")
        theta1 = st.slider("θ₁ (MA(1) coefficient)", -1.5, 1.5, 0.5, 0.05, key="B_theta1")
        root_ma = complex(-1.0 / theta1) if abs(theta1) > 1e-3 else complex(-1e6)
        st.pyplot(plot_unit_circle([root_ma], title=f"MA(1) root — θ₁ = {theta1}"))
        if abs(root_ma) > 1:
            st.success(f"|z| = {abs(root_ma):.3f}  ✓  invertible")
        else:
            st.warning(f"|z| = {abs(root_ma):.3f}  ✗  not invertible — reflect the root to fix")

    with tab_examples:
        st.markdown("#### Quick rules")
        callout("plain", "AR(1) and MA(1)",
                "AR(1) is causal iff <em>|φ<sub>1</sub>| < 1</em>. MA(1) is invertible iff <em>|θ<sub>1</sub>|<1</em>.")
        callout("plain", "AR(2) causal triangle",
                "For <em>φ(z)=1-φ<sub>1</sub> z-φ<sub>2</sub> z<sup>2</sup></em>, the causal region is "
                "<em>φ<sub>1</sub>+φ<sub>2</sub>&lt;1, &nbsp;φ<sub>2</sub>-φ<sub>1</sub>&lt;1, &nbsp;|φ<sub>2</sub>|&lt;1</em>.")

        st.markdown("##### Example B.1 — AR(1) causality")
        st.markdown("$X_t = 0.7\\,X_{t-1} + Z_t$. Root: $z = 1/0.7 \\approx 1.43 > 1$ ⇒ causal ✓")
        st.markdown("Counter-example: $X_t = 1.5\\,X_{t-1} + Z_t$. Root $z = 0.67 < 1$ ⇒ non-causal ✗")

        st.markdown("##### Example B.2 — MA(1) invertibility")
        st.markdown("$X_t = Z_t + 0.5\\,Z_{t-1}$. Root: $z = -2$, $|z|=2>1$ ⇒ invertible ✓")

        st.markdown("##### Example B.3 — Reflecting a non-invertible MA")
        st.markdown(
            "$X_t = Z_t + 2\\,Z_{t-1}$ has root $z=-0.5$ (inside) ⇒ not invertible. "
            "Reflect: replace $\\theta_1 = 2$ with $\\theta'_1 = 1/2$ and adjust $\\sigma'^2 = 4\\sigma^2$. "
            "Same ACF, but now invertible. Identification always picks the invertible representation."
        )

        callout("warn", "Why we want both at once",
                "Without invertibility, multiple MA models share the same ACF — identification is ambiguous. "
                "Without causality, the model needs future shocks. Together: <em>uniquely identifiable and "
                "physically usable.</em>")


# ============================================================================
# Page C: Box-Jenkins Pipeline
# ============================================================================
def page_box_jenkins():
    section_header(
        "C", "The Box–Jenkins Pipeline",
        "An eight-step recipe with two decision-point gates: stationarity (step 3) and "
        "white residuals (step 6).",
        "T1",
    )

    steps = [
        ("1", "Plot & inspect",
         "Look for trend, seasonality, structural breaks, outliers, growing/shrinking variance. "
         "Plot ACF and PACF as a first read."),
        ("2", "Stabilize variance, then mean",
         "Variance: log or Box–Cox if it grows with the level. Mean: regression for deterministic trend, "
         "differencing $\\nabla X_t = X_t - X_{t-1}$ for stochastic. Seasonal differencing $\\nabla_s$ if needed."),
        ("3", "Test for stationarity ◆",
         "ADF (H₀ unit root), KPSS (H₀ stationary), PP. Decision rule: ADF rejects + KPSS does NOT reject "
         "⇒ stationary, proceed. Otherwise difference and re-test."),
        ("4", "Identify orders (p, d, q)",
         "Read ACF/PACF cutoff/decay patterns. Entertain several plausible candidates."),
        ("5", "Estimate parameters",
         "Yule–Walker (AR-only, fast), CSS (full ARMA), or **MLE** (the standard, asymptotically efficient)."),
        ("6", "Diagnose residuals ◆",
         "Residuals must look like WN. Run residual ACF, Ljung–Box, JB, ARCH-LM. If anything fails, return to step 4."),
        ("7", "Select among candidates",
         "Lower IC is better. **AIC** = $-2\\ln L + 2k$, **BIC** = $-2\\ln L + k\\ln n$, "
         "**HQIC** = $-2\\ln L + 2k\\ln\\ln n$."),
        ("8", "Forecast & validate",
         "Point + interval forecasts; rolling-origin CV; metrics MAE/RMSE/MAPE/MASE. "
         "Never random k-fold on time series."),
    ]
    for num, name, body in steps:
        st.markdown(
            f"""
<div style="display:grid;grid-template-columns:48px 1fr;gap:14px;margin:14px 0;
            padding:12px 14px;background:{COL_CARD};border-left:3px solid {COL_ACCENT};
            border-radius:2px;">
<div style="font-family:'Fraunces',serif;font-weight:600;font-size:28px;line-height:1;color:{COL_ACCENT};">{num}</div>
<div>
<div style="font-family:'Fraunces',serif;font-weight:600;font-size:17px;margin-bottom:3px;">{name}</div>
<div style="font-size:15px;line-height:1.5;">{body}</div>
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    callout("key", "The pipeline as one sentence",
            "<em>Plot → stabilize → test → identify → estimate → diagnose → select → forecast.</em>")

    st.markdown("#### Tests reference table")
    tests_df = pd.DataFrame([
        ("ADF",            "Unit root (non-stationary)",  "Stationarity",  "Reject ⇒ stationary"),
        ("KPSS",           "Stationary",                  "Stationarity",  "Reject ⇒ non-stationary"),
        ("PP",             "Unit root",                   "Stationarity (HAC-robust)", "Reject ⇒ stationary"),
        ("Ljung–Box Q(m)", "Residuals are WN",            "Diagnostic",    "Reject ⇒ residuals not white"),
        ("Jarque–Bera",    "Residuals normal",            "Diagnostic",    "Reject ⇒ skew/kurtosis problem"),
        ("ARCH-LM",        "No conditional heterosked.",  "Diagnostic",    "Reject ⇒ fit a GARCH"),
        ("Engle–Granger / Johansen", "No cointegration", "Cointegration",  "Reject ⇒ cointegrated"),
    ], columns=["Test", "H₀", "Use for", "Decision"])
    st.dataframe(tests_df, use_container_width=True, hide_index=True)

    st.latex(r"Q(m) = n(n+2)\sum_{k=1}^{m}\frac{\hat\rho_k^2}{n-k}\sim \chi^2_{m-p-q}\text{ under } H_0.")


# ============================================================================
# Page D: ACF/PACF Identification (the most useful interactive tool)
# ============================================================================
def page_acf_pacf():
    section_header(
        "D", "Reading ACF & PACF Plots",
        "Look at two plots, name a model. The most actionable rote skill in time series.",
        "T1",
    )

    tab_lab, tab_table, tab_examples = st.tabs([
        "🔬 Identification lab", "📋 Reference table", "✏️ Walk-throughs",
    ])

    with tab_lab:
        st.markdown("#### Pick a model, watch its ACF/PACF signature")
        model = st.selectbox(
            "Model",
            ["AR(1)", "AR(2)", "MA(1)", "MA(2)", "ARMA(1,1)", "Random walk", "White noise"],
            key="D_model",
        )
        c1, c2 = st.columns(2)
        with c1:
            n_d = st.slider("Sample size n", 200, 2000, 600, 100, key="D_n")
            seed_d = st.number_input("Seed", value=42, key="D_seed")
        with c2:
            lags_d = st.slider("Lags shown on ACF/PACF", 10, 60, 30, 5, key="D_lags")

        # Parameter widgets per model
        if model == "AR(1)":
            phi = st.slider("φ₁", -0.95, 0.95, 0.7, 0.05, key="D_phi1")
            x = gen_arma((phi,), (), n=int(n_d), seed=int(seed_d))
            expected = "ACF tails off geometrically; PACF cuts off cleanly after lag 1."
        elif model == "AR(2)":
            phi1 = st.slider("φ₁", -1.5, 1.5, 0.6, 0.05, key="D_p1")
            phi2 = st.slider("φ₂", -0.95, 0.95, -0.3, 0.05, key="D_p2")
            x = gen_arma((phi1, phi2), (), n=int(n_d), seed=int(seed_d))
            expected = "ACF tails off (often as a damped sinusoid); PACF cuts off after lag 2."
        elif model == "MA(1)":
            theta = st.slider("θ₁", -0.95, 0.95, 0.6, 0.05, key="D_th1")
            x = gen_arma((), (theta,), n=int(n_d), seed=int(seed_d))
            expected = "ACF cuts off cleanly after lag 1; PACF tails off."
        elif model == "MA(2)":
            t1 = st.slider("θ₁", -0.95, 0.95, 0.5, 0.05, key="D_th2_1")
            t2 = st.slider("θ₂", -0.95, 0.95, 0.3, 0.05, key="D_th2_2")
            x = gen_arma((), (t1, t2), n=int(n_d), seed=int(seed_d))
            expected = "ACF cuts off after lag 2; PACF tails off."
        elif model == "ARMA(1,1)":
            phi  = st.slider("φ₁", -0.95, 0.95, 0.7, 0.05, key="D_pa")
            thet = st.slider("θ₁", -0.95, 0.95, 0.4, 0.05, key="D_pa_t")
            x = gen_arma((phi,), (thet,), n=int(n_d), seed=int(seed_d))
            expected = "Both ACF and PACF tail off — try small (p,q) and let IC choose."
        elif model == "Random walk":
            x = gen_random_walk(n=int(n_d), seed=int(seed_d))
            expected = "ACF stays near 1 for many lags (slow decay); PACF spikes at lag 1, then ≈ 0."
        else:  # White noise
            x = np.random.default_rng(int(seed_d)).standard_normal(int(n_d))
            expected = "Both ACF and PACF ≈ 0 at every lag (within ±1.96/√n bands)."

        st.pyplot(plot_series(x, title=f"Simulated {model}"))
        st.pyplot(plot_acf_pacf_pair(x, lags=int(lags_d)))

        callout("key", f"Expected pattern for {model}", expected)

    with tab_table:
        st.markdown("#### Identification cheat table")
        df = pd.DataFrame([
            ("White noise",    "≈ 0 all lags",               "≈ 0 all lags"),
            ("AR(p)",          "tails off",                  "**cuts at p**"),
            ("MA(q)",          "**cuts at q**",              "tails off"),
            ("ARMA(p, q)",     "tails off",                  "tails off"),
            ("Random walk / I(1)", "slow decay near 1",      "spike at 1, then ≈ 0"),
            ("Seasonal I(1)ₛ", "spikes at s, 2s, 3s, …",     "spikes at multiples of s"),
        ], columns=["Process", "ACF", "PACF"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        callout("plain", "Confidence bands",
                "Sample plots draw <em>\± 1.96/sqrt{n}</em> bands assuming WN. Spikes outside the bands "
                "are statistically significant at 5%.")

        st.markdown("#### 30-second reading procedure")
        st.markdown(
            "1. Look at ACF first. Slow decay near 1? — **non-stationary**, difference and re-plot.\n"
            "2. ACF cuts off cleanly at lag $q$, PACF tails off → **MA(q)**.\n"
            "3. PACF cuts off cleanly at lag $p$, ACF tails off → **AR(p)**.\n"
            "4. Both tail off → **ARMA(p, q)** — try small candidates, let AIC/BIC pick.\n"
            "5. Spikes at multiples of period $s$? → seasonal component, consider **SARIMA**."
        )

    with tab_examples:
        st.markdown("##### Example D.1")
        st.markdown(
            "Plot shows ACF significant at lag 1, ≈ 0 from lag 2 onward; "
            "PACF a damped oscillation extending to lag 6.\n\n"
            "**Diagnosis:** ACF cuts off at lag 1, PACF tails off ⇒ **MA(1)**."
        )
        st.markdown("##### Example D.2")
        st.markdown(
            "ACF decays geometrically over 8+ lags; PACF significant at lags 1 and 2 only.\n\n"
            "**Diagnosis:** ACF tails off, PACF cuts off at lag 2 ⇒ **AR(2)**."
        )


# ============================================================================
# Page E: Forecasting (interactive forecast + intervals)
# ============================================================================
def page_forecasting():
    section_header(
        "E", "Forecasting Basics",
        "Optimal forecasts, prediction intervals, and why long-horizon random-walk forecasts have "
        "such enormous uncertainty.",
        "T2",
    )

    callout("def", "MMSE forecast",
            "<em>X̂<sub>n+h</sub> = E[X<sub>n+h</sub> | ℱ<sub>n</sub>]</em> — the conditional expectation minimizes "
            "expected squared error.")
    callout("plain", "In plain English",
            "The 'best' point forecast is the conditional average of the future given the past. "
            "Among <em>linear</em> predictors specifically, this is the linear projection — the BLUP.")

    st.markdown("#### Forecast variance grows with horizon")
    st.latex(r"\mathrm{Var}(X_{n+h}-\hat X_{n+h}) = \sigma^2\sum_{j=0}^{h-1}\psi_j^2.")
    st.markdown(
        "For stationary ARMA the sum converges (variance approaches the unconditional variance). "
        "For random walks, $\\psi_j=1$ always — variance grows **linearly** in $h$ without bound."
    )

    st.markdown("#### Live demo — compare AR(1) and random-walk forecasts")
    c1, c2 = st.columns(2)
    with c1:
        e_phi = st.slider("AR(1) coefficient (for the AR(1) panel)",
                          -0.95, 0.95, 0.6, 0.05, key="E_phi")
        e_n = st.slider("Sample size", 100, 800, 300, 50, key="E_n")
    with c2:
        e_h = st.slider("Forecast horizon", 5, 100, 30, 5, key="E_h")
        e_seed = st.number_input("Seed", value=42, key="E_seed")

    x_ar = gen_ar1(e_phi, n=int(e_n), seed=int(e_seed))
    x_rw = gen_random_walk(n=int(e_n), seed=int(e_seed))

    st.pyplot(plot_forecast(x_ar, "AR(1)", horizon=int(e_h)))
    st.pyplot(plot_forecast(x_rw, "Random walk", horizon=int(e_h)))

    callout("key", "What you should see",
            "<strong>AR(1):</strong> point forecast decays geometrically toward the unconditional mean; "
            "interval width converges to the unconditional std. <strong>Random walk:</strong> point forecast "
            "is flat at the last value; interval width grows like <em>sqrt{h}</em> — at <em>h=100</em>, it's a 10× "
            "expansion over <em>h=1</em>.")

    st.markdown("#### Validation strategy")
    st.markdown(
        "- **Train/test split** — fit on early part, evaluate on a held-out tail. Honest but data-greedy.\n"
        "- **Rolling-origin CV** — repeatedly extend the training window, refit, forecast 1..h ahead, "
        "average errors. The standard for time series.\n"
        "- **Avoid random k-fold CV** — destroys temporal ordering and lets the model 'see the future'."
    )

    st.markdown("#### Standard accuracy metrics")
    metrics_df = pd.DataFrame([
        ("MAE",  r"$\frac{1}{h}\sum |X_{n+i}-\hat X_{n+i}|$",  "Mean absolute error — robust"),
        ("RMSE", r"$\sqrt{\frac{1}{h}\sum (X_{n+i}-\hat X_{n+i})^2}$", "Penalizes large errors more"),
        ("MAPE", r"$\frac{100}{h}\sum |\frac{X_{n+i}-\hat X_{n+i}}{X_{n+i}}|$", "Scale-free; breaks near zero"),
        ("MASE", "scaled by naïve seasonal benchmark", "Robust scale-free alternative"),
    ], columns=["Metric", "Formula", "Note"])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("#### Exponential smoothing — quick mention")
    st.latex(r"\hat X_{n+1} = \alpha X_n + (1-\alpha)\hat X_n,\quad 0<\alpha<1.")
    st.markdown(
        "Equivalent to ARIMA(0, 1, 1). **Holt's** adds a trend component; **Holt–Winters** adds "
        "seasonality. Full taxonomy is the **ETS** state-space framework."
    )


# ============================================================================
# Page F: VAR & Cointegration
# ============================================================================
def page_var():
    section_header(
        "F", "VAR & Cointegration",
        "Multivariate time series. Closed-book questions hit definitions and stability.",
        "T2",
    )

    st.markdown("#### Vector Autoregression VAR(p)")
    st.latex(r"\mathbf{X}_t = \mathbf{c} + \boldsymbol{\Phi}_1\mathbf{X}_{t-1} + \cdots + "
             r"\boldsymbol{\Phi}_p\mathbf{X}_{t-p} + \mathbf{Z}_t,\quad \mathbf{Z}_t\sim \text{WN}(0,\boldsymbol{\Sigma}).")
    callout("plain", "In plain English",
            "Each variable depends linearly on past values of itself and every other variable. "
            "Lag length <em>p</em> is chosen by the same IC criteria as univariate ARMA.")

    callout("thm", "VAR Stability",
            "A VAR(p) is stable (and stationary) iff all eigenvalues of the <strong>companion matrix</strong> "
            "have modulus less than one. Equivalently, all roots of "
            "<em>det(<b>I</b> - <b>Φ</b><sub>1</sub> z - … - <b>Φ</b><sub>p</sub> z<sup>p</sup>) = 0</em> "
            "lie outside the unit circle.")

    st.markdown("#### Granger causality (preview)")
    st.markdown(
        "$X_2$ Granger-causes $X_1$ if past $X_2$ improves the forecast of $X_1$ beyond what past "
        "$X_1$ provides. Tested by an F or Wald test on the lag coefficients of $X_2$ in the equation for $X_1$."
    )
    callout("warn", "What Granger causality is NOT",
            "It is <em>predictive</em> causality, not structural. And it is linear-only — it can miss "
            "nonlinear couplings, which is what CCM addresses.")

    st.markdown("#### Integration & cointegration")
    st.markdown(
        "- $X_t \\sim I(d)$ if $\\nabla^d X_t$ is stationary but $\\nabla^{d-1} X_t$ is not. Random walks are $I(1)$.\n"
        "- Two $I(1)$ series are **cointegrated** if some $\\beta\\neq 0$ makes $Y_t-\\beta X_t$ stationary $I(0)$ "
        "— they share a common stochastic trend."
    )
    callout("plain", "In plain English",
            "Each series wanders on its own (non-stationary), but a particular linear combination is "
            "well-behaved. Think: stock prices and dividends — both drift, but their ratio mean-reverts.")

    callout("key", "Why cointegration matters — VECM",
            "If two <em>I(1)</em> series are cointegrated, just differencing them in a VAR throws away the "
            "long-run equilibrium. Use the <strong>Vector Error Correction Model</strong> instead: "
            "<em>ΔX<sub>t</sub> = αβ′X<sub>t-1</sub> + Σ<sub>i</sub> Γ<sub>i</sub> ΔX<sub>t-i</sub> + Z<sub>t</sub></em>, "
            "where <em>β′X<sub>t-1</sub></em> is the cointegrating combination and <em>α</em> is the speed of adjustment.")

    st.markdown(
        "Tests: **Engle–Granger** two-step (cointegrating regression then ADF on residuals); "
        "**Johansen** trace and max-eigenvalue tests in the VECM (also identifies the rank of the "
        "cointegrating space)."
    )

    callout("warn", "Spurious regression trap",
            "Regressing one <em>I(1)</em> series on another <em>independent</em> <em>I(1)</em> series usually produces "
            "high <em>R<sup>2</sup></em> and significant <em>t</em>-statistics — the famous Granger–Newbold result. Always test "
            "for unit roots before running level regressions on time series.")


# ============================================================================
# Page G: ARCH/GARCH
# ============================================================================
def page_garch():
    section_header(
        "G", "ARCH / GARCH — Volatility Models",
        "Models the conditional variance, not the conditional mean. Likely a one-line definition question.",
        "T3",
    )

    st.markdown("#### Stylized facts")
    st.markdown(
        "- **Volatility clustering** — large changes follow large changes, small follow small.\n"
        "- **Heavy tails** — big moves more frequent than a Normal would predict.\n"
        "- **Leverage effects** — asymmetric volatility response to good vs bad news."
    )
    st.markdown("An ARMA models the conditional *mean*; ARCH/GARCH models the conditional *variance*.")

    st.markdown("#### ARCH(q) — Engle (1982)")
    st.markdown("With $Z_t = \\sigma_t\\,\\varepsilon_t$ and $\\varepsilon_t \\sim$ IID(0, 1):")
    st.latex(r"\sigma_t^2 = \alpha_0 + \alpha_1 Z_{t-1}^2 + \cdots + \alpha_q Z_{t-q}^2.")

    st.markdown("#### GARCH(p, q) — Bollerslev (1986)")
    st.latex(r"\sigma_t^2 = \alpha_0 + \sum_{i=1}^q \alpha_i Z_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2.")
    callout("key", "GARCH(1, 1) — the workhorse",
            "Extremely parsimonious yet surprisingly robust across financial applications. "
            "Parameters <em>α<sub>0</sub>, α<sub>1</sub>, β<sub>1</sub></em> — that's it.")

    st.markdown("#### Diagnostic")
    callout("plain", "ARCH-LM test",
            "Regress squared residuals on their own lags. Test joint significance: "
            "<em>nR<sup>2</sup> ∼ χ<sup>2</sup><sub>q</sub></em> under H₀ of no ARCH effect. If significant, fit ARCH/GARCH on top of the mean model.")

    st.markdown("#### Visual demo — volatility clustering")
    if st.button("Generate a GARCH-like series", key="G_gen"):
        rng = np.random.default_rng(42)
        n = 600
        sigma2 = np.zeros(n); z = np.zeros(n)
        sigma2[0] = 1.0
        a0, a1, b1 = 0.05, 0.1, 0.85
        for t in range(1, n):
            sigma2[t] = a0 + a1 * z[t-1]**2 + b1 * sigma2[t-1]
            z[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
        fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        axes[0].plot(z, color=COL_INK, linewidth=0.6)
        axes[0].set_title("simulated GARCH(1,1) series — volatility clustering visible", loc="left")
        axes[1].plot(np.sqrt(sigma2), color=COL_ACCENT, linewidth=1.0)
        axes[1].set_title("conditional std σ_t — the latent volatility process", loc="left")
        axes[1].set_xlabel("time t")
        fig.tight_layout()
        st.pyplot(fig)


# ============================================================================
# Page H: Spectral Analysis (with periodogram demo)
# ============================================================================
def page_spectral():
    section_header(
        "H", "Spectral Analysis",
        "The frequency-domain view. Closed-book questions hit definitions and Wiener–Khinchin.",
        "T2",
    )

    st.markdown("#### Two views of the same information")
    st.markdown(
        "- **Time domain** — autocovariance $\\gamma(h)$, ACF, ARMA models.\n"
        "- **Frequency domain** — spectral density $f(\\omega)$, periodogram, Fourier representation.\n"
        "The Fourier transform is the bridge."
    )

    callout("thm", "Wiener–Khinchin",
            "<em>f(ω) = (1 / 2π) · Σ<sub>h</sub> γ(h) e<sup>-iωh</sup></em> &nbsp; "
            "and inversely &nbsp; <em>γ(h) = ∫<sub>-π</sub><sup>π</sup> f(ω) e<sup>iωh</sup> dω</em>.")
    st.markdown("**Variance decomposition** — the total variance equals the integral of the spectrum:")
    st.latex(r"\gamma(0) = \int_{-\pi}^{\pi} f(\omega)\, d\omega.")

    st.markdown("#### Reading a spectrum")
    st.markdown(
        "- Peak at low $\\omega$ → slow trend / long memory.\n"
        "- Peak at $\\omega_0$ → period $2\\pi/\\omega_0$. Monthly data with annual seasonality has a peak at $\\omega = 2\\pi/12$.\n"
        "- Flat spectrum → white noise.\n"
        "- Peak at high $\\omega$ → rapid oscillation, anti-persistence."
    )

    callout("def", "Periodogram",
            "<em>I(ω<sub>j</sub>) = (1/n) · |Σ<sub>t</sub> X<sub>t</sub> e<sup>-iω<sub>j</sub>t</sup>|<sup>2</sup></em> &nbsp; "
            "for Fourier frequencies <em>ω<sub>j</sub> = 2πj / n</em>.")
    callout("warn", "Critical caveat",
            "The raw periodogram is asymptotically <strong>unbiased</strong> but "
            "<strong>inconsistent</strong> — its variance does NOT shrink with <em>n</em>. To get a "
            "consistent estimator, you must <em>smooth</em> (Bartlett, Parzen, Daniell, Welch).")

    st.markdown("#### Live demo — find the hidden frequencies")
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.slider("Frequency 1 (cycles per step)", 0.02, 0.45, 0.10, 0.005, key="H_f1")
        a1 = st.slider("Amplitude 1", 0.0, 5.0, 2.0, 0.1, key="H_a1")
    with c2:
        f2 = st.slider("Frequency 2", 0.02, 0.45, 0.25, 0.005, key="H_f2")
        a2 = st.slider("Amplitude 2", 0.0, 5.0, 1.0, 0.1, key="H_a2")
    noise_sd = st.slider("Noise σ", 0.0, 3.0, 1.0, 0.1, key="H_noise")
    n_h = st.slider("Sample size", 200, 2000, 500, 100, key="H_n")

    rng = np.random.default_rng(7)
    t = np.arange(n_h)
    x_h = (a1 * np.sin(2 * np.pi * f1 * t)
           + a2 * np.sin(2 * np.pi * f2 * t + 0.7)
           + noise_sd * rng.standard_normal(n_h))
    st.pyplot(plot_periodogram(x_h, fs=1.0))

    callout("key", "What to notice",
            "Two clean spikes appear in the periodogram exactly at the frequencies you set. "
            "As you increase the noise σ, the spectrum baseline rises and the peaks become harder to see — "
            "but smoothing (orange line) keeps them detectable.")

    st.markdown("#### Cross-spectrum tools (two series)")
    st.markdown(
        "For jointly stationary $\\{X_t, Y_t\\}$ the cross-covariance $\\gamma_{XY}(h) = "
        "\\mathrm{Cov}(X_t, Y_{t+h})$ has Fourier transform $f_{XY}(\\omega)$ (complex). Useful pieces:"
    )
    st.markdown(
        "- **Coherence**: $C_{XY}(\\omega) = |f_{XY}|^2/(f_X f_Y) \\in [0,1]$ — frequency-resolved squared correlation.\n"
        "- **Phase**: $\\arg f_{XY}(\\omega)$ — lead/lag at frequency $\\omega$.\n"
        "- **Gain**: $|f_{XY}|/f_X$ — frequency-domain regression coefficient."
    )


# ============================================================================
# Page I: Granger vs CCM
# ============================================================================
def page_granger_ccm():
    section_header(
        "I", "Granger vs CCM",
        "Two paradigms for inferring causal direction. The contrast is the most likely closed-book question.",
        "T2",
    )

    st.markdown("#### Granger causality")
    st.markdown(
        "Linear, predictive, separability assumed. Tests reduce to F or Wald on cross-coefficients in a VAR. "
        "Works for **stochastic** systems where shocks dominate."
    )

    st.markdown("#### Convergent Cross-Mapping (CCM, Sugihara 2012)")
    callout("thm", "Takens' embedding (1981)",
            "For a smooth deterministic dynamical system on a compact attractor, almost every delay "
            "embedding <em><b>x</b><sub>t</sub> = (X<sub>t</sub>, X<sub>t-τ</sub>, …, X<sub>t-(E-1)τ</sub>)</em> embeds the "
            "attractor diffeomorphically when <em>E ≥ 2d+1</em> (where <em>d</em> is the box-counting dimension). "
            "Dynamical structure is recoverable from a single observable.")
    callout("plain", "In plain English",
            "If you can only watch one variable of a chaotic system, you can still reconstruct a faithful "
            "'shadow' of its full state space by stacking time-shifted copies. Mind-bending, but it works.")

    st.markdown("#### CCM in four steps")
    st.markdown(
        "1. Build delay-embedded shadow manifolds $M_X$ and $M_Y$ from the observed series.\n"
        "2. For a target time $t$ on $M_Y$, find its $E+1$ nearest neighbors and use their corresponding "
        "$X$-values to cross-map $\\hat X_t \\mid M_Y$.\n"
        "3. Compare $\\hat X_t$ to $X_t$ — measure skill by $\\rho$ (correlation).\n"
        "4. Increase library size $L$. **If $\\rho$ converges to a positive limit as $L$ grows, "
        "$X$ causes $Y$** — because $Y$'s manifold contains information about $X$'s state."
    )
    callout("key", "Direction logic — counter-intuitive but key",
            "'<em>X</em> causes <em>Y</em>' means information about <em>X</em> has been transmitted into <em>Y</em>'s dynamics, "
            "so <em>Y</em>'s reconstructed manifold <em>M<sub>Y</sub></em> contains traces of <em>X</em>. Hence we cross-map "
            "<em>from <em>Y</em> to <em>X</em></em>, and convergence of that cross-map is the signature of <em>X → Y</em>.")

    st.markdown("#### Side-by-side comparison")
    df = pd.DataFrame([
        ("System type",      "Stochastic, linear",                   "Deterministic, nonlinear, weakly coupled"),
        ("Separability",     "Required",                              "Not required"),
        ("Statistic",        "F / Wald on lag coefficients",          "ρ vs library size L (convergence)"),
        ("Math basis",       "VAR / regression",                      "Takens' embedding theorem"),
        ("Failure mode",     "Misses nonlinear couplings",            "Fails under heavy noise / pure stochasticity"),
    ], columns=["Aspect", "Granger", "CCM"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    callout("plain", "When in doubt: run both",
            "Real data is rarely purely linear or purely deterministic. Agreement between the two "
            "methods strengthens any causal claim.")


# ============================================================================
# Page J: Simplex & S-map
# ============================================================================
def page_simplex():
    section_header(
        "J", "Simplex & S-map — Empirical Dynamic Modeling",
        "Two model-free forecasting tools built on Takens' embedding.",
        "T2",
    )

    st.markdown("#### Setup")
    st.markdown("Embed scalar $\\{X_t\\}$ into reconstructed state space:")
    st.latex(r"\mathbf{x}_t = (X_t, X_{t-\tau}, X_{t-2\tau}, \ldots, X_{t-(E-1)\tau}).")
    st.markdown(
        "Two parameters: embedding dimension $E$ and delay $\\tau$. Choose $\\tau$ near the first zero "
        "of the ACF (or first minimum of mutual information). Choose $E$ to maximize forecast skill — "
        "the Simplex step itself selects $E$."
    )
    callout("plain", "In plain English",
            "Don't fit a parametric model — treat the embedded points as samples from the system's "
            "geometry, and forecast by looking at what nearby points did next.")

    st.markdown("#### Simplex projection")
    st.markdown(
        "Forecast $X_{t+T_p}$ given $\\mathbf{x}_t$:\n"
        "1. Find the $E+1$ nearest neighbors of $\\mathbf{x}_t$ in the library.\n"
        "2. Weight each by $w_i = \\exp(-d_i / d_{\\min})$.\n"
        "3. Forecast $\\hat X_{t+T_p} = \\frac{\\sum_i w_i X_{t_i+T_p}}{\\sum_i w_i}$."
    )
    callout("key", "What Simplex tells you",
            "Plot forecast skill ρ vs. <em>E</em>. <strong>Peak at moderate <em>E</em> ⇒ deterministic dynamics.</strong> "
            "Flat low curve ⇒ noise-dominated.")

    st.markdown("#### S-map (Sequential locally-weighted global linear maps)")
    st.markdown(
        "Where Simplex uses a fixed neighborhood, S-map fits a locally-weighted linear regression at "
        "each forecast step. Locality controlled by $\\theta \\geq 0$:"
    )
    st.latex(r"w_i = \exp\!\left(-\theta\,\frac{d_i}{\bar d}\right).")
    st.markdown(
        "- $\\theta = 0$: equal weights → reduces to a global linear model (essentially AR forecasting).\n"
        "- $\\theta > 0$: closer points dominate → local approximation of nonlinear dynamics."
    )
    callout("key", "The nonlinearity test",
            "Plot forecast skill ρ vs. <em>θ</em>. "
            "<strong>If skill increases with <em>θ</em>, the system has state-dependent "
            "(nonlinear) dynamics.</strong> Flat or decreasing ⇒ a single global linear model is enough.")


# ============================================================================
# Page K: State Space & Kalman
# ============================================================================
def page_state_space():
    section_header(
        "K", "State Space & Kalman Filter",
        "A unifying framework. Likely a definition / 'name the steps' question.",
        "T3",
    )

    st.markdown("#### Linear-Gaussian state space form")
    st.latex(r"\boldsymbol{\alpha}_t = \mathbf{T}_t\,\boldsymbol{\alpha}_{t-1} + \mathbf{R}_t\,\boldsymbol{\eta}_t,\quad \boldsymbol{\eta}_t\sim \mathcal{N}(0, \mathbf{Q}_t),")
    st.latex(r"\mathbf{y}_t = \mathbf{Z}_t\,\boldsymbol{\alpha}_t + \boldsymbol{\varepsilon}_t,\quad \boldsymbol{\varepsilon}_t\sim \mathcal{N}(0, \mathbf{H}_t).")
    st.markdown(
        "First equation: **state (transition) equation**. Second: **observation (measurement) equation**. "
        "ARMA, ARIMA, structural time series (level + trend + seasonal + cycle), and dynamic factor "
        "models are all special cases."
    )
    callout("plain", "In plain English",
            "Imagine the system has a 'true' hidden state evolving over time, but we only see noisy, "
            "partial measurements. The Kalman filter keeps a best guess of that hidden state, updating "
            "it whenever a new measurement arrives.")

    st.markdown("#### The Kalman filter — two steps per time point")
    st.markdown("**Predict** — propagate the state distribution forward:")
    st.latex(r"\boldsymbol{\alpha}_{t\mid t-1} = \mathbf{T}_t\,\boldsymbol{\alpha}_{t-1\mid t-1},\quad \mathbf{P}_{t\mid t-1} = \mathbf{T}_t\,\mathbf{P}_{t-1\mid t-1}\,\mathbf{T}_t' + \mathbf{R}_t\,\mathbf{Q}_t\,\mathbf{R}_t'.")
    st.markdown("**Update** — incorporate the new observation $\\mathbf{y}_t$:")
    st.latex(r"\mathbf{K}_t = \mathbf{P}_{t\mid t-1}\,\mathbf{Z}_t'(\mathbf{Z}_t\,\mathbf{P}_{t\mid t-1}\,\mathbf{Z}_t' + \mathbf{H}_t)^{-1},")
    st.latex(r"\boldsymbol{\alpha}_{t\mid t} = \boldsymbol{\alpha}_{t\mid t-1} + \mathbf{K}_t(\mathbf{y}_t - \mathbf{Z}_t\,\boldsymbol{\alpha}_{t\mid t-1}),")
    st.latex(r"\mathbf{P}_{t\mid t} = (\mathbf{I} - \mathbf{K}_t\,\mathbf{Z}_t)\,\mathbf{P}_{t\mid t-1}.")
    st.markdown(
        "$\\mathbf{K}_t$ is the **Kalman gain** — the optimal weight on the new innovation given current "
        "uncertainty. Under the linear-Gaussian assumptions, the filter delivers the exact posterior, "
        "which is also the MMSE estimator."
    )

    callout("key", "Why it matters",
            "Handles missing data and irregular sampling naturally. Provides exact likelihood for ML "
            "estimation of any state-space model. Generalizes: EKF (extended) for nonlinear, UKF "
            "(unscented) for strongly nonlinear, particle filter for non-Gaussian.")


# ============================================================================
# Page: Cheat Sheet
# ============================================================================
def page_cheat():
    st.markdown('<div class="eyebrow">★ Final Review</div>', unsafe_allow_html=True)
    st.markdown("# Cheat Sheet")
    st.markdown(
        f'<p style="font-style:italic;color:{COL_MUTED};">'
        "If you have 30 minutes left before walking in, read only this.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("### Definitions to write from memory")
    cheat = pd.DataFrame([
        ("White noise",       "E[Z]=0, Var=σ², Cov=0 for s≠t. Uncorrelated, not necessarily IID."),
        ("Strict stationarity","Joint distribution shift-invariant for every k and shift h."),
        ("Weak stationarity", "Constant mean, finite constant variance, γ(h) depends only on h."),
        ("Ergodicity",        "Time avgs → ensemble avgs. Sufficient: γ(h)→0 as h→∞."),
        ("Linear process",    "X_t = μ + Σ ψ_j Z_{t-j}, Σψ_j² < ∞. Causal if ψ_j=0 for j<0."),
        ("Wold decomposition","Every cov-stationary process = MA(∞) regular part + deterministic part."),
        ("Causality",         "AR roots φ(z)=0 satisfy |z|>1. AR(1): |φ_1|<1."),
        ("Invertibility",     "MA roots θ(z)=0 satisfy |z|>1. MA(1): |θ_1|<1."),
        ("ARMA(p,q)",         "φ(B)X_t = θ(B)Z_t. ARIMA adds (1-B)^d. SARIMA adds seasonal polys."),
    ], columns=["Term", "Definition"])
    st.dataframe(cheat, use_container_width=True, hide_index=True)

    st.markdown("### Pipeline in one stroke")
    st.markdown(
        "**(1)** Plot & inspect → **(2)** Stabilize (log/Box–Cox + diff) → **(3)** Stationarity tests "
        "(ADF, KPSS, PP) → **(4)** Identify (p,d,q) from ACF/PACF → **(5)** Estimate (Yule–Walker / "
        "CSS / MLE) → **(6)** Diagnose: residual ACF, Ljung–Box, JB, ARCH-LM → **(7)** Select by "
        "AIC/BIC/HQIC (lower better) → **(8)** Forecast + rolling-origin CV (MAE, RMSE, MAPE, MASE)."
    )

    st.markdown("### ACF / PACF identification")
    df = pd.DataFrame([
        ("AR(p)",        "tails off",          "**cuts at p**"),
        ("MA(q)",        "**cuts at q**",      "tails off"),
        ("ARMA(p,q)",    "tails off",          "tails off"),
        ("I(1) / RW",    "slow decay near 1",  "spike at 1"),
        ("White noise",  "≈ 0 all lags",       "≈ 0 all lags"),
    ], columns=["Process", "ACF", "PACF"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Bands: ±1.96/√n.")

    st.markdown("### Tests at a glance")
    df = pd.DataFrame([
        ("ADF",            "Unit root",                    "Reject ⇒ stationary"),
        ("KPSS",           "Stationary",                   "Reject ⇒ non-stationary"),
        ("PP",             "Unit root",                    "HAC-robust stationarity"),
        ("Ljung–Box",      "Residuals are WN",             "Reject ⇒ residuals not white"),
        ("Jarque–Bera",    "Residuals normal",             "Reject ⇒ skew/kurtosis problem"),
        ("ARCH-LM",        "No ARCH effect",               "Reject ⇒ fit GARCH"),
        ("Engle–Granger / Johansen", "No cointegration",   "Reject ⇒ cointegrated"),
    ], columns=["Test", "H₀", "Decision"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Information criteria")
    st.markdown(
        "- **AIC** $= -2\\ln L + 2k$ — lightest penalty, may over-fit, efficient for prediction.\n"
        "- **BIC** $= -2\\ln L + k\\ln n$ — heaviest for $n>7$, consistent for true order.\n"
        "- **HQIC** $= -2\\ln L + 2k\\ln\\ln n$ — middle ground."
    )

    st.markdown("### Frequency-domain one-liners")
    st.markdown(
        "- **Wiener–Khinchin**: $f(\\omega)$ is the Fourier transform of $\\gamma(h)$.\n"
        "- **Variance decomposition**: $\\gamma(0) = \\int_{-\\pi}^{\\pi} f(\\omega)\\,d\\omega$.\n"
        "- **Periodogram**: sample analogue of $f(\\omega)$; unbiased but inconsistent — must smooth.\n"
        "- **Coherence**: $|f_{XY}|^2 / (f_X f_Y) \\in [0, 1]$. Frequency-resolved squared correlation."
    )

    st.markdown("### Causality & dynamics one-liners")
    st.markdown(
        "- **Granger**: past X improves forecast of Y beyond past Y. Linear, predictive. F-test on lag coefficients.\n"
        "- **Takens**: delay embedding with $E\\geq 2d+1$ reconstructs the attractor.\n"
        "- **CCM**: $X\\to Y$ iff cross-mapping skill ρ converges as library $L$ grows. Nonlinear-friendly.\n"
        "- **Simplex**: forecast by $E+1$-NN weighted average. Optimal $E$ peaks the skill curve.\n"
        "- **S-map θ-test**: skill rising with θ ⇒ nonlinear dynamics.\n"
        "- **VAR stability**: companion-matrix eigenvalues inside unit circle.\n"
        "- **Cointegration**: I(1) series with $Y - \\beta X$ being I(0). Use VECM, not differenced VAR.\n"
        "- **Kalman**: predict ($\\alpha_{t\\mid t-1}$, $P_{t\\mid t-1}$) → update via gain $K_t$."
    )

    st.markdown("### Quick recall checklist — close the page and write these down")
    st.markdown(
        "1. The two stationarity definitions and the implication relationships.\n"
        "2. The definition of ergodicity and a sufficient condition.\n"
        "3. Causality root condition + one-line proof (geometric series).\n"
        "4. Invertibility root condition and why we need it.\n"
        "5. Wold's theorem and why it justifies ARMA.\n"
        "6. The 8-step Box–Jenkins pipeline.\n"
        "7. ADF and KPSS — opposite nulls and decision rule.\n"
        "8. ACF/PACF identification table for AR, MA, ARMA, I(1), WN.\n"
        "9. Ljung–Box statistic and df = m − p − q.\n"
        "10. AIC / BIC / HQIC — formulas and which over-fits.\n"
        "11. Wiener–Khinchin and the variance decomposition.\n"
        "12. Granger vs CCM — when each works and when each fails.\n"
        "13. Takens embedding statement: $E \\geq 2d + 1$.\n"
        "14. Simplex (skill ⇒ determinism), S-map θ-test (skill rising ⇒ nonlinearity).\n"
        "15. VAR stability via companion-matrix eigenvalues.\n"
        "16. Cointegration definition and why VECM beats differenced VAR."
    )

    callout("plain", "Final note",
            "You have done eight notebooks of work to get here. Walk in calm, read each question twice, "
            "and trust the preparation. <strong>Good luck, JBong.</strong>")


# ============================================================================
# Page: Recall Drill (interactive flashcards)
# ============================================================================
DRILL_QUESTIONS = [
    ("State the three conditions of weak stationarity.",
     "(1) Constant mean E[X_t]=μ. (2) Finite constant variance Var(X_t)=γ(0). "
     "(3) Cov(X_t, X_{t+h}) = γ(h) depends only on the lag h, not on t."),
    ("State the root condition for an ARMA process to be causal.",
     "Every root z of φ(z) = 0 must satisfy |z| > 1 (lie strictly outside the unit circle)."),
    ("State the root condition for an ARMA process to be invertible.",
     "Every root z of θ(z) = 0 must satisfy |z| > 1 (lie strictly outside the unit circle)."),
    ("Quick rule for AR(1) causality.",
     "|φ_1| < 1."),
    ("Quick rule for MA(1) invertibility.",
     "|θ_1| < 1."),
    ("State Wold's decomposition theorem (informally).",
     "Every covariance-stationary, purely non-deterministic process can be written as a causal MA(∞) "
     "regular part plus a deterministic part — X_t = Σ ψ_j Z_{t-j} + V_t with Σ ψ_j² < ∞."),
    ("Define ergodicity in one sentence; give a sufficient condition.",
     "Time averages converge to ensemble averages: ar X_n → μ. "
     "Sufficient condition: γ(h) → 0 as h → ∞."),
    ("Write the Ljung–Box statistic and its asymptotic distribution under H₀.",
     "Q(m) = n(n+2) Σ_{k=1}^m ρ̂_k²/(n−k) ~ χ²_{m−p−q} under H₀: residuals are WN."),
    ("In ACF/PACF identification, what does 'ACF cuts off at q, PACF tails off' indicate?",
     "An MA(q) model."),
    ("In ACF/PACF identification, what does 'PACF cuts off at p, ACF tails off' indicate?",
     "An AR(p) model."),
    ("List the 8 steps of the Box–Jenkins pipeline.",
     "(1) Plot & inspect → (2) Stabilize variance and mean → (3) Test for stationarity → "
     "(4) Identify orders (p,d,q) → (5) Estimate parameters → (6) Diagnose residuals → "
     "(7) Select among candidates by AIC/BIC/HQIC → (8) Forecast and validate."),
    ("Null hypotheses of the ADF and KPSS tests — and their decision rules.",
     "ADF H₀: unit root (non-stationary). Reject ⇒ stationary. "
     "KPSS H₀: stationary. Reject ⇒ non-stationary. "
     "Combined rule: ADF rejects + KPSS does NOT reject ⇒ stationary."),
    ("State the Wiener–Khinchin theorem.",
     "For a stationary process with summable autocovariances, the spectral density is the Fourier "
     "transform of the autocovariance: f(ω) = (1/2π) Σ_h γ(h) e^{−iωh}."),
    ("Why is the raw periodogram inconsistent?",
     "It is asymptotically unbiased, but its variance does NOT shrink as n grows. "
     "To get a consistent estimator, you must smooth (Bartlett, Parzen, Daniell, Welch)."),
    ("State Takens' embedding theorem.",
     "For a smooth dynamical system on a compact attractor of box-counting dimension d, "
     "almost every delay embedding (X_t, X_{t−τ}, …, X_{t−(E−1)τ}) with E ≥ 2d+1 "
     "embeds the original attractor diffeomorphically."),
    ("How does the S-map θ-parameter detect nonlinearity?",
     "Plot forecast skill ρ vs θ. If ρ rises with θ, the system has state-dependent (nonlinear) "
     "dynamics. Flat or decreasing means a single global linear model is sufficient."),
    ("Why use a VECM instead of a differenced VAR for cointegrated I(1) series?",
     "Differencing throws away the long-run equilibrium relationship encoded in the cointegrating "
     "vector. The VECM keeps it: ΔX_t = α β' X_{t−1} + ΣΓ_i ΔX_{t−i} + Z_t."),
    ("State the VAR stability condition.",
     "All eigenvalues of the companion matrix have modulus less than 1, equivalently all roots of "
     "det(I − Φ_1 z − ⋯ − Φ_p z^p) = 0 lie outside the unit circle."),
    ("AIC vs BIC — which over-fits and why?",
     "AIC = −2 ln L + 2k uses a constant penalty (2 per parameter); BIC = −2 ln L + k ln n uses "
     "a log-n penalty. For n > e² ≈ 7, BIC penalizes more heavily. AIC tends to over-fit; "
     "BIC is consistent for the true model order if it lies in the candidate set."),
    ("State the Kalman filter's two steps in words.",
     "Predict: propagate the prior state distribution forward through the transition equation. "
     "Update: combine the prediction with the new observation, weighted by the Kalman gain K_t, "
     "to form the posterior."),
]


def page_recall():
    st.markdown('<div class="eyebrow">🎯 Self-test</div>', unsafe_allow_html=True)
    st.markdown("# Recall Drill")
    st.markdown(
        f'<p style="font-style:italic;color:{COL_MUTED};">'
        "Read the prompt, write your answer (in your head or on paper), then reveal the model answer. "
        "Aim for fluency, not perfection.</p>",
        unsafe_allow_html=True,
    )

    # Persistent state
    if "drill_idx" not in st.session_state:
        st.session_state.drill_idx = 0
        st.session_state.drill_order = list(range(len(DRILL_QUESTIONS)))

    total = len(DRILL_QUESTIONS)
    idx = st.session_state.drill_idx % total
    qi = st.session_state.drill_order[idx]
    q, a = DRILL_QUESTIONS[qi]

    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
        f'letter-spacing:0.18em;text-transform:uppercase;color:{COL_ACCENT};margin-top:18px;">'
        f"Question {idx + 1} of {total}</div>",
        unsafe_allow_html=True,
    )
    callout("def", "Prompt", q)

    if st.button("Reveal answer", key=f"reveal_{idx}"):
        callout("key", "Model answer", a)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("← Previous", key="drill_prev"):
            st.session_state.drill_idx = (st.session_state.drill_idx - 1) % total
            st.rerun()
    with c2:
        if st.button("🎲 Shuffle", key="drill_shuffle"):
            rng = np.random.default_rng()
            order = list(range(total))
            rng.shuffle(order)
            st.session_state.drill_order = order
            st.session_state.drill_idx = 0
            st.rerun()
    with c3:
        if st.button("Next →", key="drill_next"):
            st.session_state.drill_idx = (st.session_state.drill_idx + 1) % total
            st.rerun()

    st.progress((idx + 1) / total)
    st.caption(f"Use these between sessions. Aim to clear all {total} questions twice tonight.")


# ============================================================================
# Router
# ============================================================================
PAGES = {
    "🏠 Home":                    page_home,
    "A · Stationarity ★":         page_stationarity,
    "B · Causality & Invertibility ★": page_causality,
    "C · Box–Jenkins Pipeline":   page_box_jenkins,
    "D · ACF/PACF Identification ★": page_acf_pacf,
    "E · Forecasting ★":          page_forecasting,
    "F · VAR & Cointegration":    page_var,
    "G · ARCH/GARCH":             page_garch,
    "H · Spectral Analysis ★":    page_spectral,
    "I · Granger vs CCM":         page_granger_ccm,
    "J · Simplex & S-map":        page_simplex,
    "K · State Space & Kalman":   page_state_space,
    "★ Cheat Sheet":              page_cheat,
    "🎯 Recall Drill":            page_recall,
}

with st.sidebar:
    st.markdown("## ATSA Reviewer")
    st.caption("Part 1 · Closed-Book · 25%")
    st.markdown("---")
    page_name = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.caption("📅 Exam: 02 May 2026, 12:30")
    st.caption("🎯 Pass mark: ≥ 80%")
    st.caption("⭐ = interactive simulation")

PAGES[page_name]()
