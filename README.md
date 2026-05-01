# ATSA Reviewer — Streamlit App

Interactive companion for the ATSA comprehensive exam (Part 1, closed-book, 25%).
Mobile-friendly, with live simulations for stationarity, ACF/PACF identification,
differencing, the unit circle, forecasting, and spectral analysis.

## Run locally (laptop)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in
any browser.

## Deploy to Streamlit Community Cloud (recommended for phone access)

The fastest way to get this on your phone as a near-PWA:

1. **Push to GitHub.** Create a new public repo (e.g. `atsa-reviewer`) and
   commit two files at the root:
   - `app.py`
   - `requirements.txt`

2. **Deploy.** Go to [share.streamlit.io](https://share.streamlit.io), sign in
   with GitHub, click *New app*, point it at your repo, set *Main file path*
   to `app.py`, click *Deploy*. First deploy takes ~2 minutes.

3. **Open on phone.** When the cloud URL is live, open it in:
   - **Android (Chrome):** menu (⋮) → *Add to Home screen* → confirm.
   - **iOS (Safari):** Share button (□↑) → *Add to Home Screen* → confirm.

   The app will now launch from the home screen icon and open in a near-fullscreen
   webview — close enough to a native app for studying purposes.

> **Note on "true" PWA**: Streamlit Community Cloud doesn't expose the manifest
> or service worker in a way that makes it a fully installable PWA on every
> platform. The "Add to Home Screen" path above works on every modern mobile
> browser and is the recommended approach. If you want a true PWA wrapper, see
> the `streamlit-pwa-template` project on GitHub which embeds the app in an
> iframe on a static site you control.

## Tips for mobile use

- The sidebar collapses into a hamburger menu on narrow screens — tap the `>`
  arrow at the top-left to navigate between sections.
- Plots reflow automatically. Pinch-to-zoom works on the live simulations.
- Sliders are tap-and-drag. For finer control, tap the slider once then use the
  on-screen `←` `→` keys.
- The Recall Drill page has *Previous / Shuffle / Next* buttons designed to be
  tappable with one thumb.

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

Same content as the BSDS-edition HTML/PDF reviewer, but reformatted for mobile
consumption with live data simulations replacing the static SVGs.
