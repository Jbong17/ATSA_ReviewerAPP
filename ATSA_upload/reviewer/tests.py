"""Statistical tests used by the stationarity page."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def run_tests(x: np.ndarray) -> pd.DataFrame:
    """Run ADF and KPSS, return a small results table."""
    rows: list[tuple] = []
    try:
        adf = adfuller(x, autolag="AIC")
        rows.append((
            "ADF", "Unit root (non-stationary)",
            f"{adf[0]:.3f}", f"{adf[1]:.3f}",
            "✓ stationary" if adf[1] < 0.05 else "✗ cannot reject unit root",
        ))
    except Exception as e:  # noqa: BLE001 — surface in the table rather than crash
        rows.append(("ADF", "—", "err", "err", str(e)))
    try:
        kp = kpss(x, regression="c", nlags="auto")
        rows.append((
            "KPSS", "Stationary (level)",
            f"{kp[0]:.3f}", f"{kp[1]:.3f}",
            "✗ rejects stationarity" if kp[1] < 0.05 else "✓ cannot reject stationarity",
        ))
    except Exception as e:  # noqa: BLE001
        rows.append(("KPSS", "—", "err", "err", str(e)))
    return pd.DataFrame(rows, columns=["Test", "Null H₀", "Statistic", "p-value", "Verdict"])
