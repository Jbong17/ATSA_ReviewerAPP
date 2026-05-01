"""Page C — Box–Jenkins pipeline: 8-step protocol with decision gates."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..components import callout, section_header
from ..theme import COL_ACCENT, COL_CARD


def render() -> None:
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
