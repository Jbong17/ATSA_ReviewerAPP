"""Cheat-Sheet page — compressed must-knows for the last 30 minutes."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..components import callout
from ..theme import COL_MUTED


def render() -> None:
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
