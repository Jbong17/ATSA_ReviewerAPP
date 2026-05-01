"""Reusable UI components — colored callouts and consistent section headers."""
from __future__ import annotations

import streamlit as st

from .theme import COL_MUTED, COL_RULE

# (background, border, label-text) tuples
CALLOUT_STYLES = {
    "def":   ("#fdf6dd", "#d4a373", "#8a6a1a"),
    "thm":   ("#e8eef5", "#6b8caf", "#2c4a6e"),
    "plain": ("#f0e6f0", "#8a5a8a", "#6a3f6a"),
    "key":   ("#e6efe1", "#7a9466", "#4a6638"),
    "warn":  ("#fbe8e0", "#c0613f", "#8a3a1c"),
}


def callout(kind: str, label: str, body_md: str) -> None:
    """Render a colored callout box. ``body_md`` is rendered as inline HTML/markdown.

    kind ∈ {def, thm, plain, key, warn}
    """
    bg, border, txt = CALLOUT_STYLES.get(kind, CALLOUT_STYLES["plain"])
    st.markdown(
        f"""
<div style="background:{bg};border-left:3px solid {border};padding:14px 18px;
            margin:14px 0 18px;border-radius:2px;">
  <div style="font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;font-size:10.5px;
              letter-spacing:0.18em;text-transform:uppercase;color:{txt};
              margin-bottom:6px;font-weight:500;">{label}</div>
  <div style="font-family:'Crimson Pro',Georgia,serif;font-size:16px;line-height:1.55;">{body_md}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def section_header(num: str, title: str, lede: str, tier: str = "T1") -> None:
    """Render a consistent section header with eyebrow, title, tier pill, lede."""
    st.markdown(
        f"""
<div class="eyebrow">Section {num}</div>
<h1 style="margin-top:0;margin-bottom:6px;">{title} <span class="pill">{tier}</span></h1>
<p style="font-style:italic;color:{COL_MUTED};margin:0 0 22px;
          border-bottom:1px solid {COL_RULE};padding-bottom:14px;">{lede}</p>
""",
        unsafe_allow_html=True,
    )
