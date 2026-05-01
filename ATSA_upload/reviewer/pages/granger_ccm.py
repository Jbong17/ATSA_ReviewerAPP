"""Page I — Granger causality vs CCM (Convergent Cross-Mapping)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..components import callout, section_header


def render() -> None:
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
