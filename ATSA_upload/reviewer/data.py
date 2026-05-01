"""Synthetic data generators used across the reviewer pages."""
from __future__ import annotations

import numpy as np
import streamlit as st
from statsmodels.tsa.arima_process import ArmaProcess


@st.cache_data
def gen_ar1(phi: float, n: int = 400, seed: int = 42) -> np.ndarray:
    """Generate an AR(1) realization. φ controls persistence."""
    rng = np.random.default_rng(seed)
    if abs(phi) >= 1:
        # Direct recursion — also handles random walk and explosive cases.
        x = np.zeros(n)
        z = rng.standard_normal(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + z[t]
        return x
    ar = ArmaProcess(np.r_[1, -phi], np.r_[1])
    return ar.generate_sample(nsample=n, distrvs=rng.standard_normal)


@st.cache_data
def gen_arma(ar_coefs: tuple, ma_coefs: tuple, n: int = 400, seed: int = 42) -> np.ndarray:
    """Generate ARMA(p, q). Coefficients in 'natural' form:

        X_t = ar_coefs[0]·X_{t-1} + … + Z_t + ma_coefs[0]·Z_{t-1} + …
    """
    rng = np.random.default_rng(seed)
    ar = np.r_[1, -np.array(ar_coefs)]      # statsmodels uses signs flipped
    ma = np.r_[1, np.array(ma_coefs)]
    proc = ArmaProcess(ar, ma)
    return proc.generate_sample(nsample=n, distrvs=rng.standard_normal)


@st.cache_data
def gen_random_walk(n: int = 400, seed: int = 42, drift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(drift + rng.standard_normal(n))


@st.cache_data
def gen_trend_stationary(n: int = 400, seed: int = 42, slope: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return slope * np.arange(n) + rng.standard_normal(n)
