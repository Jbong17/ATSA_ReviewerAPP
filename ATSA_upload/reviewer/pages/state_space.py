"""Page K — State-space form and the Kalman filter."""
from __future__ import annotations

import streamlit as st

from ..components import callout, section_header


def render() -> None:
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
