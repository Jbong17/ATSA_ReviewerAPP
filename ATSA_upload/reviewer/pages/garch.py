"""Page G — ARCH/GARCH conditional volatility models."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ..components import callout, section_header
from ..theme import COL_ACCENT, COL_INK


def render() -> None:
    section_header(
        "G", "ARCH / GARCH — Volatility Models",
        "Models the conditional variance, not the conditional mean. Likely a one-line definition question.",
        "T3",
    )

    st.markdown("#### Stylized facts")
    st.markdown(
        "- **Volatility clustering** — large changes follow large changes, small follow small.\n"
        "- **Heavy tails** — big moves more frequent than a Normal would predict.\n"
        "- **Leverage effects** — asymmetric volatility response to good vs bad news."
    )
    st.markdown("An ARMA models the conditional *mean*; ARCH/GARCH models the conditional *variance*.")

    st.markdown("#### ARCH(q) — Engle (1982)")
    st.markdown("With $Z_t = \\sigma_t\\,\\varepsilon_t$ and $\\varepsilon_t \\sim$ IID(0, 1):")
    st.latex(r"\sigma_t^2 = \alpha_0 + \alpha_1 Z_{t-1}^2 + \cdots + \alpha_q Z_{t-q}^2.")

    st.markdown("#### GARCH(p, q) — Bollerslev (1986)")
    st.latex(r"\sigma_t^2 = \alpha_0 + \sum_{i=1}^q \alpha_i Z_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2.")
    callout("key", "GARCH(1, 1) — the workhorse",
            "Extremely parsimonious yet surprisingly robust across financial applications. "
            "Parameters <em>α<sub>0</sub>, α<sub>1</sub>, β<sub>1</sub></em> — that's it.")

    st.markdown("#### Diagnostic")
    callout("plain", "ARCH-LM test",
            "Regress squared residuals on their own lags. Test joint significance: "
            "<em>nR<sup>2</sup> ∼ χ<sup>2</sup><sub>q</sub></em> under H₀ of no ARCH effect. If significant, fit ARCH/GARCH on top of the mean model.")

    st.markdown("#### Visual demo — volatility clustering")
    if st.button("Generate a GARCH-like series", key="G_gen"):
        rng = np.random.default_rng(42)
        n = 600
        sigma2 = np.zeros(n); z = np.zeros(n)
        sigma2[0] = 1.0
        a0, a1, b1 = 0.05, 0.1, 0.85
        for t in range(1, n):
            sigma2[t] = a0 + a1 * z[t-1]**2 + b1 * sigma2[t-1]
            z[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
        fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        axes[0].plot(z, color=COL_INK, linewidth=0.6)
        axes[0].set_title("simulated GARCH(1,1) series — volatility clustering visible", loc="left")
        axes[1].plot(np.sqrt(sigma2), color=COL_ACCENT, linewidth=1.0)
        axes[1].set_title("conditional std σ_t — the latent volatility process", loc="left")
        axes[1].set_xlabel("time t")
        fig.tight_layout()
        st.pyplot(fig)
