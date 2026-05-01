"""Page J — Simplex projection and S-map (Empirical Dynamic Modeling)."""
from __future__ import annotations

import streamlit as st

from ..components import callout, section_header


def render() -> None:
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
