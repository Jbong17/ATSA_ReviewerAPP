"""Page D — ACF/PACF identification lab + reference table + walk-throughs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..components import callout, section_header
from ..data import gen_arma, gen_random_walk
from ..plots import plot_acf_pacf_pair, plot_series


def render() -> None:
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
                "Sample plots draw <em>± 1.96/√n</em> bands assuming WN. Spikes outside the bands "
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
