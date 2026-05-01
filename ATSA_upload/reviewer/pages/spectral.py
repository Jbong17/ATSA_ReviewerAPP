"""Page H — Spectral analysis: Wiener–Khinchin, periodogram, hidden-frequency demo."""
from __future__ import annotations

import numpy as np
import streamlit as st

from ..components import callout, section_header
from ..plots import plot_periodogram


def render() -> None:
    section_header(
        "H", "Spectral Analysis",
        "The frequency-domain view. Closed-book questions hit definitions and Wiener–Khinchin.",
        "T2",
    )

    st.markdown("#### Two views of the same information")
    st.markdown(
        "- **Time domain** — autocovariance $\\gamma(h)$, ACF, ARMA models.\n"
        "- **Frequency domain** — spectral density $f(\\omega)$, periodogram, Fourier representation.\n"
        "The Fourier transform is the bridge."
    )

    callout("thm", "Wiener–Khinchin",
            "<em>f(ω) = (1 / 2π) · Σ<sub>h</sub> γ(h) e<sup>-iωh</sup></em> &nbsp; "
            "and inversely &nbsp; <em>γ(h) = ∫<sub>-π</sub><sup>π</sup> f(ω) e<sup>iωh</sup> dω</em>.")
    st.markdown("**Variance decomposition** — the total variance equals the integral of the spectrum:")
    st.latex(r"\gamma(0) = \int_{-\pi}^{\pi} f(\omega)\, d\omega.")

    st.markdown("#### Reading a spectrum")
    st.markdown(
        "- Peak at low $\\omega$ → slow trend / long memory.\n"
        "- Peak at $\\omega_0$ → period $2\\pi/\\omega_0$. Monthly data with annual seasonality has a peak at $\\omega = 2\\pi/12$.\n"
        "- Flat spectrum → white noise.\n"
        "- Peak at high $\\omega$ → rapid oscillation, anti-persistence."
    )

    callout("def", "Periodogram",
            "<em>I(ω<sub>j</sub>) = (1/n) · |Σ<sub>t</sub> X<sub>t</sub> e<sup>-iω<sub>j</sub>t</sup>|<sup>2</sup></em> &nbsp; "
            "for Fourier frequencies <em>ω<sub>j</sub> = 2πj / n</em>.")
    callout("warn", "Critical caveat",
            "The raw periodogram is asymptotically <strong>unbiased</strong> but "
            "<strong>inconsistent</strong> — its variance does NOT shrink with <em>n</em>. To get a "
            "consistent estimator, you must <em>smooth</em> (Bartlett, Parzen, Daniell, Welch).")

    st.markdown("#### Live demo — find the hidden frequencies")
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.slider("Frequency 1 (cycles per step)", 0.02, 0.45, 0.10, 0.005, key="H_f1")
        a1 = st.slider("Amplitude 1", 0.0, 5.0, 2.0, 0.1, key="H_a1")
    with c2:
        f2 = st.slider("Frequency 2", 0.02, 0.45, 0.25, 0.005, key="H_f2")
        a2 = st.slider("Amplitude 2", 0.0, 5.0, 1.0, 0.1, key="H_a2")
    noise_sd = st.slider("Noise σ", 0.0, 3.0, 1.0, 0.1, key="H_noise")
    n_h = st.slider("Sample size", 200, 2000, 500, 100, key="H_n")

    rng = np.random.default_rng(7)
    t = np.arange(n_h)
    x_h = (a1 * np.sin(2 * np.pi * f1 * t)
           + a2 * np.sin(2 * np.pi * f2 * t + 0.7)
           + noise_sd * rng.standard_normal(n_h))
    st.pyplot(plot_periodogram(x_h, fs=1.0))

    callout("key", "What to notice",
            "Two clean spikes appear in the periodogram exactly at the frequencies you set. "
            "As you increase the noise σ, the spectrum baseline rises and the peaks become harder to see — "
            "but smoothing (orange line) keeps them detectable.")

    st.markdown("#### Cross-spectrum tools (two series)")
    st.markdown(
        "For jointly stationary $\\{X_t, Y_t\\}$ the cross-covariance $\\gamma_{XY}(h) = "
        "\\mathrm{Cov}(X_t, Y_{t+h})$ has Fourier transform $f_{XY}(\\omega)$ (complex). Useful pieces:"
    )
    st.markdown(
        "- **Coherence**: $C_{XY}(\\omega) = |f_{XY}|^2/(f_X f_Y) \\in [0,1]$ — frequency-resolved squared correlation.\n"
        "- **Phase**: $\\arg f_{XY}(\\omega)$ — lead/lag at frequency $\\omega$.\n"
        "- **Gain**: $|f_{XY}|/f_X$ — frequency-domain regression coefficient."
    )
