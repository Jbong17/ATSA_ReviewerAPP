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
    # collapsed because nav now lives at the top of the page — sidebar would
    # just take horizontal space on desktop without helping mobile (Streamlit
    # auto-collapses sidebars below ~768px viewport, requiring an extra tap).
    initial_sidebar_state="collapsed",
    menu_items={
        "About": (
            "ATSA Comp Exam · Part 1 (Closed-Book, 25%) · BSDS Edition\n\n"
            "Interactive companion with live data simulations."
        )
    },
)

from reviewer.pages import PAGES  # noqa: E402 — must come after set_page_config
from reviewer.theme import COL_ACCENT, COL_MUTED, COL_RULE, apply_theme  # noqa: E402

apply_theme()


# ----------------------------------------------------------------------------
# Top-of-page nav — always visible on every viewport, no sidebar required
# ----------------------------------------------------------------------------
st.markdown(
    f"""
<div style="margin-bottom:6px;">
  <div style="font-family:'Fraunces',Georgia,serif;font-weight:600;font-size:22px;
              color:{COL_ACCENT};line-height:1.1;">ATSA Reviewer</div>
  <div style="font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:10.5px;
              letter-spacing:0.18em;text-transform:uppercase;color:{COL_MUTED};">
    Part 1 · Closed-Book · 25%  ·  Exam 02 May 2026, 12:30
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Persist last-viewed page across reruns so reveal/shuffle/slider widgets don't
# yank the user back to Home on every interaction.
if "current_page" not in st.session_state:
    st.session_state.current_page = next(iter(PAGES))

page_name = st.selectbox(
    "Section",
    list(PAGES.keys()),
    index=list(PAGES.keys()).index(st.session_state.current_page),
    label_visibility="collapsed",
    key="nav_select",
)
st.session_state.current_page = page_name
st.markdown(
    f"<hr style='margin:8px 0 18px;border:none;border-top:1px solid {COL_RULE};'>",
    unsafe_allow_html=True,
)

PAGES[page_name]()
