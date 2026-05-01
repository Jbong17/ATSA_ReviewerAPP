"""Page E — Forecasting: MMSE, prediction intervals, validation strategy."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..components import callout, section_header
from ..data import gen_ar1, gen_random_walk
from ..plots import plot_forecast


def render() -> None:
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
            "is flat at the last value; interval width grows like <em>√h</em> — at <em>h=100</em>, it's a 10× "
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
