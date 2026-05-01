# ATSA Reviewer — Streamlit App (modular)

Interactive companion for the ATSA comprehensive exam (Part 1, closed-book, 25%).
Mobile-friendly, with live simulations for stationarity, ACF/PACF identification,
differencing, the unit circle, forecasting, and spectral analysis.

## What changed in this version

The original `app.py` was a single 1,681-line monolith with a CSS rule
(`[class*="css"] { ... !important }`) that fought with Streamlit's internal
styling. On slow networks (where Google Fonts didn't load) the rule starved
sidebar nav labels and headings of their color, so they appeared invisible
or smeared.

Two changes:

1. **Render fix.** The blanket selector and aggressive `!important` are gone.
   Color is now declared explicitly on `.stApp`, sidebar labels, and radio
   options so text stays legible even if the web font never arrives.
   Web fonts load with `font-display: swap` so the page is readable
   immediately in a system serif fallback.
2. **Modular layout.** The app is now a small entry file plus a `reviewer/`
   package with one module per concern.

```
ATSA_ReviewerAPP/
├── app.py                     # ~50 lines — page_config, theme, sidebar, router
├── requirements.txt
└── reviewer/
    ├── theme.py               # colors, matplotlib defaults, CSS, PWA meta
    ├── components.py          # callout(), section_header()
    ├── data.py                # synthetic data generators
    ├── plots.py               # all matplotlib helpers
    ├── tests.py               # ADF + KPSS runner
    ├── content/
    │   └── drill.py           # 20 flashcards
    └── pages/
        ├── __init__.py        # PAGES registry
        ├── home.py
        ├── stationarity.py
        ├── causality.py
        ├── box_jenkins.py
        ├── acf_pacf.py
        ├── forecasting.py
        ├── var.py
        ├── garch.py
        ├── spectral.py
        ├── granger_ccm.py
        ├── simplex.py
        ├── state_space.py
        ├── cheat.py
        └── recall.py
```

Each page module exposes a single `render()` function. To add a page:

1. Drop a new module under `reviewer/pages/`.
2. Import it in `reviewer/pages/__init__.py` and add a `PAGES[label] = mod.render`
   entry. The sidebar nav and router pick it up automatically.

## Run locally (laptop)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`).

### Phone use without internet

Streamlit needs a Python server. To use this from your phone offline:

1. Run `streamlit run app.py --server.address 0.0.0.0` on your laptop.
2. Find the laptop's local IP (e.g. `192.168.1.42`).
3. On your phone, connect to the **same Wi-Fi**, open
   `http://192.168.1.42:8501`. No internet round-trip required.

## Deploy to Streamlit Community Cloud

1. Push to GitHub (root must contain `app.py`, `requirements.txt`, and the
   `reviewer/` package).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, click
   *New app*, point at the repo, set *Main file path* to `app.py`, deploy.
3. On phone: add the cloud URL to home screen (Chrome → menu → *Add to Home
   screen*; Safari → Share → *Add to Home Screen*).

## What's in the app

| Page | What it covers | Interactive? |
| --- | --- | --- |
| Home | Exam metadata, install instructions | — |
| A · Stationarity | Definitions, stationary vs not, three conditions, differencing demo, ADF/KPSS | ★ |
| B · Causality & Invertibility | Live unit-circle plot driven by AR/MA coefficient sliders | ★ |
| C · Box–Jenkins Pipeline | The 8-step protocol with all the test references | — |
| D · ACF/PACF Identification | Pick model + parameters, see live ACF/PACF | ★ |
| E · Forecasting | AR(1) vs random-walk forecast with prediction intervals | ★ |
| F · VAR & Cointegration | Multivariate stability, VECM, spurious regression | — |
| G · ARCH/GARCH | Volatility clustering with simulated GARCH series | (button) |
| H · Spectral Analysis | Find hidden frequencies in the periodogram | ★ |
| I · Granger vs CCM | Side-by-side decision logic | — |
| J · Simplex & S-map | Embedding, Simplex, the S-map θ-test | — |
| K · State Space & Kalman | Predict and update equations | — |
| ★ Cheat Sheet | Compressed must-knows in mobile-readable tables | — |
| 🎯 Recall Drill | 20 prompt-and-reveal flashcards with shuffle | ★ |

## Author note

Prepared for J. B. Bongalos for the ATSA Comprehensive Exam Part 1
(02 May 2026, 12:30–16:30, closed everything, ≥80% pass mark).
