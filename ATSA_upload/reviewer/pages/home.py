"""Home page — exam metadata, install instructions, section index."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..components import callout
from ..theme import COL_MUTED


def render() -> None:
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
        st.markdown("**Candidate:** Jerald B. Bongalos")
        st.markdown("**Exam date:** 02 May 2026, 12:30–16:30")
        st.markdown("**Duration:** 4 hours total")
    with col2:
        st.markdown("**Pass mark:** ≥ 80% accumulated")
        st.markdown("**Part 1 weight:** 25% · *closed everything*")
        st.markdown("**Part 2 weight:** 75% · *open everything*")

    callout(
        "key", "How to use this app",
        "Open one section per study block. Sections marked ★ have <strong>live data simulations</strong> "
        "you can play with — change a slider, watch the ACF and the test statistics change in real time. "
        "End every session with the <em>Cheat Sheet</em> for retention."
    )

    st.markdown("### Sections")
    rows = [
        ("A", "Stationarity ★",            "Three conditions, ergodicity, Wold theorem", "T1"),
        ("B", "Causality & Invertibility ★", "Unit-circle root condition, worked checks", "T1"),
        ("C", "Box–Jenkins Pipeline",      "The eight-step protocol with decisions",      "T1"),
        ("D", "ACF/PACF Identification ★", "Read the plots, name the model",              "T1"),
        ("E", "Forecasting ★",             "MMSE, prediction intervals, validation",      "T2"),
        ("F", "VAR & Cointegration",       "Multivariate stability, VECM",                "T2"),
        ("G", "ARCH/GARCH",                "Volatility clustering",                        "T3"),
        ("H", "Spectral Analysis ★",       "Wiener–Khinchin, periodogram",                 "T2"),
        ("I", "Granger vs CCM",            "Linear-stochastic vs nonlinear-deterministic", "T2"),
        ("J", "Simplex & S-map",           "Empirical dynamic modeling",                   "T2"),
        ("K", "State Space & Kalman",      "Predict & update",                             "T3"),
        ("★", "Cheat Sheet",               "All must-knows, mobile-readable",              "—"),
        ("🎯", "Recall Drill",             "Self-test flashcards",                         "—"),
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
