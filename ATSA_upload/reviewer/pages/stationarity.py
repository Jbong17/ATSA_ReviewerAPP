"""Page A — Stationarity: definitions + comparison + 3 conditions + differencing + ADF/KPSS."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ..components import callout, section_header
from ..data import gen_ar1, gen_random_walk, gen_trend_stationary
from ..plots import plot_acf_pacf_pair, plot_rolling_stats, plot_series
from ..tests import run_tests
from ..theme import COL_ACCENT, COL_GREEN, COL_INK


def render() -> None:
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

        win = max(20, int(n3 * 0.1))
        st.pyplot(plot_rolling_stats(x, window=win, title=choice))

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
