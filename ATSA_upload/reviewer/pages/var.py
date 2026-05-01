"""Page F — VAR, Granger preview, integration, cointegration, VECM, spurious regression."""
from __future__ import annotations

import streamlit as st

from ..components import callout, section_header


def render() -> None:
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
