"""Recall-Drill page — 20 prompt-and-reveal flashcards with shuffle/prev/next."""
from __future__ import annotations

import random

import streamlit as st

from ..components import callout
from ..content.drill import DRILL_QUESTIONS
from ..theme import COL_MUTED


def render() -> None:
    st.markdown('<div class="eyebrow">🎯 Self-test</div>', unsafe_allow_html=True)
    st.markdown("# Recall Drill")
    st.markdown(
        f'<p style="font-style:italic;color:{COL_MUTED};">'
        "Read the prompt, write your answer (in your head or on paper), then reveal the model answer. "
        "Aim for fluency, not perfection.</p>",
        unsafe_allow_html=True,
    )

    # Persistent state
    if "drill_idx" not in st.session_state:
        st.session_state.drill_idx = 0
        st.session_state.drill_order = list(range(len(DRILL_QUESTIONS)))

    total = len(DRILL_QUESTIONS)
    idx = st.session_state.drill_idx % total
    qi = st.session_state.drill_order[idx]
    q, a = DRILL_QUESTIONS[qi]

    st.markdown(f"**Question {idx + 1} of {total}**")
    callout("def", "Prompt", q)

    with st.expander("Reveal answer"):
        st.markdown(a)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("← Previous", key="drill_prev"):
            st.session_state.drill_idx = (st.session_state.drill_idx - 1) % total
            st.rerun()
    with c2:
        if st.button("🔀 Shuffle", key="drill_shuffle"):
            order = list(range(total))
            random.shuffle(order)
            st.session_state.drill_order = order
            st.session_state.drill_idx = 0
            st.rerun()
    with c3:
        if st.button("Next →", key="drill_next"):
            st.session_state.drill_idx = (st.session_state.drill_idx + 1) % total
            st.rerun()

    st.progress((idx + 1) / total)
    st.caption(f"Use these between sessions. Aim to clear all {total} questions twice tonight.")
