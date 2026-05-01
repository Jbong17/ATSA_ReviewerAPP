"""Page registry — maps display label → render() callable.

Each page module exposes a single ``render()`` function that draws the page.
The router in ``app.py`` looks up the chosen label here and invokes it.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable

from . import (
    acf_pacf, box_jenkins, causality, cheat, forecasting, garch, granger_ccm,
    home, recall, simplex, spectral, state_space, stationarity, var,
)

PAGES: "OrderedDict[str, Callable[[], None]]" = OrderedDict([
    ("🏠 Home",                          home.render),
    ("A · Stationarity ★",               stationarity.render),
    ("B · Causality & Invertibility ★",  causality.render),
    ("C · Box–Jenkins Pipeline",         box_jenkins.render),
    ("D · ACF/PACF Identification ★",    acf_pacf.render),
    ("E · Forecasting ★",                forecasting.render),
    ("F · VAR & Cointegration",          var.render),
    ("G · ARCH/GARCH",                   garch.render),
    ("H · Spectral Analysis ★",          spectral.render),
    ("I · Granger vs CCM",               granger_ccm.render),
    ("J · Simplex & S-map",              simplex.render),
    ("K · State Space & Kalman",         state_space.render),
    ("★ Cheat Sheet",                    cheat.render),
    ("🎯 Recall Drill",                  recall.render),
])
