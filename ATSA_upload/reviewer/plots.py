"""Matplotlib plot helpers used across pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from .theme import (
    COL_ACCENT, COL_GREEN, COL_INK, COL_MUTED, COL_RULE, COL_SOFT,
)


def plot_series(x: np.ndarray, title: str = "", color=None, mean_line=True, ax=None):
    """Plot a single series with optional mean overlay."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 2.6))
    else:
        fig = ax.figure
    ax.plot(x, color=color or COL_INK, linewidth=0.9)
    if mean_line:
        ax.axhline(np.mean(x), color=COL_ACCENT, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(title, loc="left", color=COL_INK)
    ax.set_xlabel("time t")
    fig.tight_layout()
    return fig


def plot_rolling_stats(x: np.ndarray, window: int, title: str = "") -> plt.Figure:
    """Plot series with rolling mean and ±1 rolling-std envelope."""
    s = pd.Series(x)
    rmean = s.rolling(window).mean()
    rstd  = s.rolling(window).std()

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(x, color=COL_INK, linewidth=0.8, alpha=0.85, label="series")
    ax.plot(rmean, color=COL_ACCENT, linewidth=1.3, label=f"rolling mean (w={window})")
    ax.fill_between(s.index, rmean - rstd, rmean + rstd,
                    color=COL_SOFT, alpha=0.18, label="±1 rolling std")
    ax.set_title(title, loc="left")
    ax.set_xlabel("time t")
    ax.legend(loc="upper left", framealpha=0)
    fig.tight_layout()
    return fig


def plot_acf_pacf_pair(x: np.ndarray, lags: int = 30) -> plt.Figure:
    """Side-by-side ACF and PACF with shared y-limits."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    plot_acf(x, lags=lags, ax=axes[0], alpha=0.05, color=COL_INK,
             vlines_kwargs={"colors": COL_INK})
    plot_pacf(x, lags=lags, ax=axes[1], alpha=0.05, method="ywm", color=COL_INK,
              vlines_kwargs={"colors": COL_INK})
    axes[0].set_title("ACF",  loc="left")
    axes[1].set_title("PACF", loc="left")
    for a in axes:
        a.set_ylim(-1.05, 1.05)
        a.set_xlabel("lag")
    fig.tight_layout()
    return fig


def plot_unit_circle(roots: list[complex], title: str = "Roots in the complex plane") -> plt.Figure:
    """Visualize roots relative to the unit circle."""
    fig, ax = plt.subplots(figsize=(5, 5))
    forbidden = Circle((0, 0), 1.0, color=COL_SOFT, alpha=0.15, zorder=0)
    ax.add_patch(forbidden)
    theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(np.cos(theta), np.sin(theta), color=COL_ACCENT, linewidth=1.6)
    ax.axhline(0, color=COL_INK, linewidth=0.7)
    ax.axvline(0, color=COL_INK, linewidth=0.7)
    for r in roots:
        good = abs(r) > 1
        ax.plot(r.real, r.imag, "o",
                color=COL_GREEN if good else COL_ACCENT,
                markersize=10, markeredgecolor=COL_INK, markeredgewidth=0.8)
        ax.annotate(f"|z|={abs(r):.2f}",
                    (r.real, r.imag), xytext=(8, 8),
                    textcoords="offset points", fontsize=9,
                    color=COL_GREEN if good else COL_ACCENT, fontweight="bold")
    lim = max(2.0, max([abs(r) for r in roots] + [1.5]) * 1.15)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Re(z)"); ax.set_ylabel("Im(z)")
    ax.set_title(title, loc="left")
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_GREEN,
               markeredgecolor=COL_INK, markersize=10, label="|z| > 1  ✓ causal/invertible"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_ACCENT,
               markeredgecolor=COL_INK, markersize=10, label="|z| ≤ 1  ✗ violates"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=8.5,
              framealpha=0.85, edgecolor=COL_RULE)
    fig.tight_layout()
    return fig


def plot_periodogram(x: np.ndarray, fs: float = 1.0) -> plt.Figure:
    """Plot the raw periodogram and a smoothed (Daniell-style) estimate."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fx = np.fft.rfft(x - np.mean(x))
    pgram = (np.abs(fx) ** 2) / n

    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
    axes[0].plot(np.arange(n), x, color=COL_INK, linewidth=0.7)
    axes[0].set_title("series", loc="left"); axes[0].set_xlabel("")

    axes[1].plot(freqs, pgram, color=COL_INK, linewidth=0.6, alpha=0.55, label="raw")
    if len(pgram) > 9:
        smooth = pd.Series(pgram).rolling(7, center=True, min_periods=1).mean()
        axes[1].plot(freqs, smooth, color=COL_ACCENT, linewidth=1.4,
                     label="smoothed (Daniell)")
    axes[1].set_title("periodogram I(ω)", loc="left")
    axes[1].set_xlabel("frequency (cycles per time step)")
    axes[1].legend(framealpha=0)
    fig.tight_layout()
    return fig


def plot_forecast(x: np.ndarray, model: str, horizon: int = 30, sigma: float = 1.0) -> plt.Figure:
    """Plot the series with point forecasts and 95% prediction intervals.

    Implements the analytic AR(1) and random-walk forecasts directly.
    """
    n = len(x)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(np.arange(n), x, color=COL_INK, linewidth=0.9, label="observed")

    fut_idx = np.arange(n, n + horizon)

    if model == "AR(1)":
        phi = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        mean_ = np.mean(x)
        forecasts = mean_ + (x[-1] - mean_) * phi ** np.arange(1, horizon + 1)
        psi = phi ** np.arange(horizon)
        var_h = sigma ** 2 * np.cumsum(psi ** 2)
        se = np.sqrt(var_h)
        ax.set_title(f"AR(1) forecast (estimated φ ≈ {phi:.2f}) — mean reverts", loc="left")
    else:  # Random walk
        forecasts = np.full(horizon, x[-1])
        var_h = sigma ** 2 * np.arange(1, horizon + 1)  # grows linearly
        se = np.sqrt(var_h)
        ax.set_title("Random-walk forecast — flat point, growing intervals", loc="left")

    ax.plot(fut_idx, forecasts, color=COL_ACCENT, linewidth=1.6, label="forecast")
    ax.fill_between(fut_idx, forecasts - 1.96 * se, forecasts + 1.96 * se,
                    color=COL_SOFT, alpha=0.25, label="95% PI")
    ax.axvline(n - 0.5, color=COL_MUTED, linestyle=":", linewidth=1)
    ax.set_xlabel("time t")
    ax.legend(framealpha=0, loc="upper left")
    fig.tight_layout()
    return fig
