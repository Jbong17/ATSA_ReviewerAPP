"""
Theme module — colors, matplotlib defaults, CSS injection.

Rendering-bug fix notes (vs the original single-file app):
  * Replaced the over-broad `[class*="css"]` selector with targeted selectors
    rooted at `.stApp` and `section[data-testid="stSidebar"]`. The old selector
    matched every emotion-generated Streamlit class and fought with internal
    styling, especially in 1.32+.
  * Removed gratuitous `!important` from font-family rules; reserved only for
    the few places where Streamlit's specificity demands it.
  * Switched the Google-Fonts dependency from `@import` (render-blocking, fails
    silently) to `<link rel="preconnect">` + a `<link rel="stylesheet">` tag
    with `font-display: swap`, so text is visible immediately in the system
    serif fallback while the web font loads.
  * Explicit `color` declarations on `body`, `.stApp`, sidebar labels, and
    radio options to guarantee legibility even if the web font never arrives.
  * Strong serif fallback chain: Fraunces / Crimson Pro → Georgia → Times New
    Roman → serif.
"""
from __future__ import annotations

import matplotlib as mpl
import streamlit as st

# ----------------------------------------------------------------------------
# Color tokens — match the printed reviewer's editorial aesthetic
# ----------------------------------------------------------------------------
COL_BG     = "#f7f3ec"   # warm cream
COL_CARD   = "#fffaf0"   # lighter card
COL_INK    = "#1c1814"   # near-black ink
COL_MUTED  = "#6b6259"
COL_RULE   = "#d9cfc0"
COL_ACCENT = "#8b1d1d"   # oxblood
COL_SOFT   = "#b8553a"   # terracotta
COL_GREEN  = "#7a9466"
COL_BLUE   = "#6b8caf"
COL_YELLOW = "#d4a373"


# ----------------------------------------------------------------------------
# Matplotlib defaults — applied to every figure
# ----------------------------------------------------------------------------
def apply_matplotlib_theme() -> None:
    mpl.rcParams.update({
        "axes.facecolor":   COL_BG,
        "figure.facecolor": COL_BG,
        "savefig.facecolor": COL_BG,
        "axes.edgecolor":   COL_INK,
        "axes.labelcolor":  COL_INK,
        "axes.titlecolor":  COL_INK,
        "text.color":       COL_INK,
        "xtick.color":      COL_INK,
        "ytick.color":      COL_INK,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":      True,
        "grid.color":     COL_RULE,
        "grid.alpha":     0.6,
        "grid.linewidth": 0.5,
        "font.family":    "serif",
        "font.size":      9.5,
        "axes.titlesize":   11,
        "axes.titleweight": "bold",
        "axes.labelsize":   10,
        "legend.fontsize":  9,
        "legend.frameon":   False,
    })


# ----------------------------------------------------------------------------
# PWA / mobile meta tags
# ----------------------------------------------------------------------------
def inject_pwa_meta() -> None:
    st.markdown(
        f"""
<meta name="theme-color" content="{COL_ACCENT}">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="ATSA Reviewer">
<meta name="mobile-web-app-capable" content="yes">
<meta name="application-name" content="ATSA Reviewer">
""",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Custom CSS — typography + colored callouts + rendering fixes
# ----------------------------------------------------------------------------
_CSS = f"""
<!-- Preconnect speeds up font fetch; stylesheet uses font-display:swap so text is visible immediately. -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Crimson+Pro:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">

<style>
/* ----- Base text colors — explicit so missing fonts can't hide content ---- */
.stApp {{
    background: {COL_BG};
    color: {COL_INK};
}}
.stApp, .stApp p, .stApp li, .stApp span, .stApp div, .stApp label,
.stMarkdown, .stMarkdown p, .stMarkdown li {{
    font-family: 'Crimson Pro', Georgia, 'Times New Roman', serif;
    color: {COL_INK};
}}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    font-family: 'Fraunces', Georgia, 'Times New Roman', serif;
    letter-spacing: -0.005em;
    color: {COL_INK};
}}
.stApp h1 {{ font-weight: 600; }}
.stApp h2, .stApp h3 {{ font-weight: 600; }}

/* ----- Sidebar -------------------------------------------------------- */
section[data-testid="stSidebar"] {{
    background: {COL_CARD};
    border-right: 1px solid {COL_RULE};
}}
section[data-testid="stSidebar"] * {{
    color: {COL_INK};
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {COL_ACCENT};
    font-family: 'Fraunces', Georgia, serif;
}}
/* The radio nav: ensure each option label is visible. This is the rule that
   was effectively missing in the original app — the broad `!important` font
   rule was overriding color in some Streamlit versions. */
section[data-testid="stSidebar"] [data-baseweb="radio"] label,
section[data-testid="stSidebar"] [data-baseweb="radio"] label *,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stRadio label * {{
    color: {COL_INK} !important;
    font-family: 'Crimson Pro', Georgia, serif !important;
    font-size: 15px;
    line-height: 1.5;
}}

/* ----- Eyebrow / section number --------------------------------------- */
.eyebrow {{
    font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace;
    font-size: 10.5px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {COL_ACCENT};
    margin-bottom: 4px;
}}

/* ----- Tier pill ------------------------------------------------------ */
.pill {{
    display: inline-block;
    font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace;
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 8px;
    background: {COL_ACCENT};
    color: {COL_BG};
    border-radius: 2px;
    margin-left: 6px;
    vertical-align: middle;
}}

/* ----- Hide Streamlit chrome for cleaner mobile view ------------------ */
#MainMenu {{ visibility: hidden; }}
footer    {{ visibility: hidden; }}
header    {{ visibility: hidden; }}

/* ----- Mobile spacing ------------------------------------------------- */
.block-container {{ padding-top: 1.5rem; padding-bottom: 4rem; }}
@media (max-width: 600px) {{
    .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    .stApp h1 {{ font-size: 1.7rem !important; }}
    .stApp h2 {{ font-size: 1.35rem !important; }}
}}

/* ----- Math ----------------------------------------------------------- */
.katex {{ font-size: 1.02em; }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Convenience: do everything theme-related in one call
# ----------------------------------------------------------------------------
def apply_theme() -> None:
    """Apply matplotlib + PWA meta + custom CSS in one go."""
    apply_matplotlib_theme()
    inject_pwa_meta()
    inject_css()
