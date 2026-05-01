"""20 prompt-and-reveal flashcards for the recall drill page."""

DRILL_QUESTIONS: list[tuple[str, str]] = [
    ("State the three conditions of weak stationarity.",
     "(1) Constant mean E[X_t]=μ. (2) Finite constant variance Var(X_t)=γ(0). "
     "(3) Cov(X_t, X_{t+h}) = γ(h) depends only on the lag h, not on t."),
    ("State the root condition for an ARMA process to be causal.",
     "Every root z of φ(z) = 0 must satisfy |z| > 1 (lie strictly outside the unit circle)."),
    ("State the root condition for an ARMA process to be invertible.",
     "Every root z of θ(z) = 0 must satisfy |z| > 1 (lie strictly outside the unit circle)."),
    ("Quick rule for AR(1) causality.",
     "|φ_1| < 1."),
    ("Quick rule for MA(1) invertibility.",
     "|θ_1| < 1."),
    ("State Wold's decomposition theorem (informally).",
     "Every covariance-stationary, purely non-deterministic process can be written as a causal MA(∞) "
     "regular part plus a deterministic part — X_t = Σ ψ_j Z_{t-j} + V_t with Σ ψ_j² < ∞."),
    ("Define ergodicity in one sentence; give a sufficient condition.",
     "Time averages converge to ensemble averages: bar X_n → μ. "
     "Sufficient condition: γ(h) → 0 as h → ∞."),
    ("Write the Ljung–Box statistic and its asymptotic distribution under H₀.",
     "Q(m) = n(n+2) Σ_{k=1}^m ρ̂_k²/(n−k) ~ χ²_{m−p−q} under H₀: residuals are WN."),
    ("In ACF/PACF identification, what does 'ACF cuts off at q, PACF tails off' indicate?",
     "An MA(q) model."),
    ("In ACF/PACF identification, what does 'PACF cuts off at p, ACF tails off' indicate?",
     "An AR(p) model."),
    ("List the 8 steps of the Box–Jenkins pipeline.",
     "(1) Plot & inspect → (2) Stabilize variance and mean → (3) Test for stationarity → "
     "(4) Identify orders (p,d,q) → (5) Estimate parameters → (6) Diagnose residuals → "
     "(7) Select among candidates by AIC/BIC/HQIC → (8) Forecast and validate."),
    ("Null hypotheses of the ADF and KPSS tests — and their decision rules.",
     "ADF H₀: unit root (non-stationary). Reject ⇒ stationary. "
     "KPSS H₀: stationary. Reject ⇒ non-stationary. "
     "Combined rule: ADF rejects + KPSS does NOT reject ⇒ stationary."),
    ("State the Wiener–Khinchin theorem.",
     "For a stationary process with summable autocovariances, the spectral density is the Fourier "
     "transform of the autocovariance: f(ω) = (1/2π) Σ_h γ(h) e^{−iωh}."),
    ("Why is the raw periodogram inconsistent?",
     "It is asymptotically unbiased, but its variance does NOT shrink as n grows. "
     "To get a consistent estimator, you must smooth (Bartlett, Parzen, Daniell, Welch)."),
    ("State Takens' embedding theorem.",
     "For a smooth dynamical system on a compact attractor of box-counting dimension d, "
     "almost every delay embedding (X_t, X_{t−τ}, …, X_{t−(E−1)τ}) with E ≥ 2d+1 "
     "embeds the original attractor diffeomorphically."),
    ("How does the S-map θ-parameter detect nonlinearity?",
     "Plot forecast skill ρ vs θ. If ρ rises with θ, the system has state-dependent (nonlinear) "
     "dynamics. Flat or decreasing means a single global linear model is sufficient."),
    ("Why use a VECM instead of a differenced VAR for cointegrated I(1) series?",
     "Differencing throws away the long-run equilibrium relationship encoded in the cointegrating "
     "vector. The VECM keeps it: ΔX_t = α β' X_{t−1} + ΣΓ_i ΔX_{t−i} + Z_t."),
    ("State the VAR stability condition.",
     "All eigenvalues of the companion matrix have modulus less than 1, equivalently all roots of "
     "det(I − Φ_1 z − ⋯ − Φ_p z^p) = 0 lie outside the unit circle."),
    ("AIC vs BIC — which over-fits and why?",
     "AIC = −2 ln L + 2k uses a constant penalty (2 per parameter); BIC = −2 ln L + k ln n uses "
     "a log-n penalty. For n > e² ≈ 7, BIC penalizes more heavily. AIC tends to over-fit; "
     "BIC is consistent for the true model order if it lies in the candidate set."),
    ("State the Kalman filter's two steps in words.",
     "Predict: propagate the prior state distribution forward through the transition equation. "
     "Update: combine the prediction with the new observation, weighted by the Kalman gain K_t, "
     "to form the posterior."),
]
