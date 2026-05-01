"""Page B — Causality & Invertibility: definitions + live unit-circle + worked examples."""
from __future__ import annotations

import streamlit as st

from ..components import callout, section_header
from ..data import gen_ar1
from ..plots import plot_series, plot_unit_circle
from ..theme import COL_ACCENT, COL_GREEN


def render() -> None:
    section_header(
        "B", "Causality & Invertibility",
        "Two conditions on the roots of the AR and MA polynomials. Probably the single most "
        "computable closed-book question.",
        "T1",
    )

    tab_def, tab_unit, tab_examples = st.tabs([
        "📚 Definitions", "🎯 Live unit-circle demo", "✏️ Worked examples",
    ])

    with tab_def:
        st.markdown("#### The ARMA(p, q) model")
        st.latex(r"\phi(B)\, X_t = \theta(B)\, Z_t,\quad \{Z_t\}\sim\text{WN}(0,\sigma^2),")
        st.latex(r"\phi(B) = 1-\phi_1 B-\cdots-\phi_p B^p,\quad \theta(B) = 1+\theta_1 B+\cdots+\theta_q B^q.")

        callout("def", "Causality",
                "The process can be written as <em>X<sub>t</sub> = Σ<sub>j=0</sub>^∞ ψ<sub>j</sub> Z<sub>t-j</sub></em> with "
                "<em>Σ |ψ<sub>j</sub>| < ∞</em> — present and past innovations only.")
        callout("plain", "In plain English — causality",
                "Today's value depends only on shocks that have already happened. It cannot depend on "
                "future shocks. We don't have access to the future when we forecast.")

        callout("def", "Invertibility",
                "<em>Z<sub>t</sub> = Σ<sub>j=0</sub><sup>∞</sup> π<sub>j</sub> X<sub>t-j</sub></em> with <em>Σ |π<sub>j</sub>| &lt; ∞</em> — innovations "
                "recoverable from past observations.")
        callout("plain", "In plain English — invertibility",
                "Looking at the data, you can work backwards and figure out what the random shocks were. "
                "Without it, the model is unusable for forecasting.")

        st.markdown("#### The root condition")
        callout("thm", "Causality",
                "An ARMA process is causal iff every root of <em>φ(z)=0</em> has <em>|z| > 1</em> — outside the unit circle.")
        callout("thm", "Invertibility",
                "Same picture for <em>θ(z)</em>: every root of <em>θ(z)=0</em> must have <em>|z|>1</em>.")

        st.markdown("#### Why outside the unit circle?")
        st.markdown(
            "For an AR(1), $\\phi(B) = 1 - \\phi B$. To rewrite as a one-sided sum we invert via "
            "the geometric series:"
        )
        st.latex(r"\frac{1}{1-\phi B} = \sum_{j=0}^{\infty}\phi^j B^j,")
        st.markdown(
            "which converges only when $|\\phi|<1$, i.e. when the root $z=1/\\phi$ has $|z|>1$. "
            "If the root were inside, the series would blow up — no causal MA(∞) representation exists."
        )

    with tab_unit:
        st.markdown("#### Move the AR(1) coefficient and watch the root")
        st.markdown(
            "For AR(1) the polynomial is $\\phi(z) = 1 - \\phi_1 z$, so the single root is $z = 1/\\phi_1$. "
            "Drag the slider through and through 1 to see the boundary."
        )

        phi1 = st.slider("φ₁ (AR(1) coefficient)", -1.5, 1.5, 0.7, 0.05, key="B_phi1")
        root = complex(1.0 / phi1) if abs(phi1) > 1e-3 else complex(1e6)

        c1, c2 = st.columns([1.1, 1.0])
        with c1:
            st.pyplot(plot_unit_circle([root], title=f"AR(1) root — φ₁ = {phi1}"))
        with c2:
            st.markdown(f"**Root location:** $z = 1/\\phi_1 = {root.real:.3f}$")
            st.markdown(f"**$|z|$:** {abs(root):.3f}")
            if abs(root) > 1:
                st.markdown(f"<span style='color:{COL_GREEN};font-weight:600'>✓ Causal</span>",
                            unsafe_allow_html=True)
            elif abs(root) == 1:
                st.markdown(f"<span style='color:{COL_ACCENT};font-weight:600'>= unit root (boundary, non-stationary)</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:{COL_ACCENT};font-weight:600'>✗ Non-causal (explosive)</span>",
                            unsafe_allow_html=True)
            st.markdown("---")
            x = gen_ar1(phi1, n=400)
            st.pyplot(plot_series(x, title="Simulated path", mean_line=False))

        callout("key", "What to notice",
                "When |φ₁| < 1 the root sits outside the unit circle (green) and the series wobbles around a "
                "constant level. When |φ₁| crosses 1 the root moves inside the circle (red) and the series "
                "explodes. At φ₁ = 1 you get the random walk — boundary case, non-stationary.")

        st.markdown("#### MA(1) version")
        theta1 = st.slider("θ₁ (MA(1) coefficient)", -1.5, 1.5, 0.5, 0.05, key="B_theta1")
        root_ma = complex(-1.0 / theta1) if abs(theta1) > 1e-3 else complex(-1e6)
        st.pyplot(plot_unit_circle([root_ma], title=f"MA(1) root — θ₁ = {theta1}"))
        if abs(root_ma) > 1:
            st.success(f"|z| = {abs(root_ma):.3f}  ✓  invertible")
        else:
            st.warning(f"|z| = {abs(root_ma):.3f}  ✗  not invertible — reflect the root to fix")

    with tab_examples:
        st.markdown("#### Quick rules")
        callout("plain", "AR(1) and MA(1)",
                "AR(1) is causal iff <em>|φ<sub>1</sub>| < 1</em>. MA(1) is invertible iff <em>|θ<sub>1</sub>|<1</em>.")
        callout("plain", "AR(2) causal triangle",
                "For <em>φ(z)=1-φ<sub>1</sub> z-φ<sub>2</sub> z<sup>2</sup></em>, the causal region is "
                "<em>φ<sub>1</sub>+φ<sub>2</sub>&lt;1, &nbsp;φ<sub>2</sub>-φ<sub>1</sub>&lt;1, &nbsp;|φ<sub>2</sub>|&lt;1</em>.")

        st.markdown("##### Example B.1 — AR(1) causality")
        st.markdown("$X_t = 0.7\\,X_{t-1} + Z_t$. Root: $z = 1/0.7 \\approx 1.43 > 1$ ⇒ causal ✓")
        st.markdown("Counter-example: $X_t = 1.5\\,X_{t-1} + Z_t$. Root $z = 0.67 < 1$ ⇒ non-causal ✗")

        st.markdown("##### Example B.2 — MA(1) invertibility")
        st.markdown("$X_t = Z_t + 0.5\\,Z_{t-1}$. Root: $z = -2$, $|z|=2>1$ ⇒ invertible ✓")

        st.markdown("##### Example B.3 — Reflecting a non-invertible MA")
        st.markdown(
            "$X_t = Z_t + 2\\,Z_{t-1}$ has root $z=-0.5$ (inside) ⇒ not invertible. "
            "Reflect: replace $\\theta_1 = 2$ with $\\theta'_1 = 1/2$ and adjust $\\sigma'^2 = 4\\sigma^2$. "
            "Same ACF, but now invertible. Identification always picks the invertible representation."
        )

        callout("warn", "Why we want both at once",
                "Without invertibility, multiple MA models share the same ACF — identification is ambiguous. "
                "Without causality, the model needs future shocks. Together: <em>uniquely identifiable and "
                "physically usable.</em>")
