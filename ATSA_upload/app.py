"""
ATSA Comprehensive Exam Part 1 — Interactive Reviewer (Streamlit / PWA-ready)

Thin entry point. All real work lives in the ``reviewer`` package:

    reviewer/
      theme.py            colors, matplotlib defaults, CSS injection
      components.py       callout(), section_header()
      data.py             gen_ar1, gen_arma, gen_random_walk, gen_trend_stationary
      plots.py            plot_series, plot_acf_pacf_pair, plot_unit_circle, ...
      tests.py            run_tests (ADF + KPSS)
      content/drill.py    20 recall-drill flashcards
      pages/              one module per page; each exposes render()
      pages/__init__.py   PAGES registry (label → render callable)

Author     : Prepared for J. B. Bongalos
Exam date  : 02 May 2026, 12:30–16:30
Run locally:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

# st.set_page_config MUST be the first Streamlit call, before any reviewer imports
# that themselves call into Streamlit.
st.set_page_config(
    page_title="ATSA Reviewer",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        "About": (
            "ATSA Comp Exam · Part 1 (Closed-Book, 25%) · BSDS Edition\n\n"
            "Interactive companion with live data simulations."
        )
    },
)

from reviewer.pages import PAGES  # noqa: E402 — must come after set_page_config
from reviewer.theme import apply_theme  # noqa: E402

apply_theme()


# ----------------------------------------------------------------------------
# Sidebar nav + dispatch
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ATSA Reviewer")
    st.caption("Part 1 · Closed-Book · 25%")
    st.markdown("---")
    page_name = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.caption("📅 Exam: 02 May 2026, 12:30")
    st.caption("🎯 Pass mark: ≥ 80%")
    st.caption("⭐ = interactive simulation")

PAGES[page_name]()
