"""
simulate_rct.py
---------------
Simulates a binary treatment assignment and binary outcome (7-day return)
on the user feature matrix X, with a known heterogeneous CATE τ(X) as ground truth.

Two assignment modes:
    'rct'          -- Bernoulli(0.5), pure randomization (sanity-check baseline)
    'observational' -- P(T=1|X) is a learned propensity (high-activity users are
                      more likely to receive the notification), introducing
                      intentional selection bias to stress-test estimators.

The CATE is kept available as ground truth (tau_true column) to enable
PEHE evaluation.

Usage:
    from src.simulate_rct import simulate_treatment_outcome

    df = simulate_treatment_outcome(X, mode='observational', seed=42)
    # df has columns: all X features + treatment, outcome, tau_true, propensity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid


# ---------------------------------------------------------------------------
# Ground-truth CATE function
# ---------------------------------------------------------------------------

def _true_cate(X: pd.DataFrame) -> np.ndarray:
    """
    Heterogeneous CATE τ(X) -- constructed so that:
        - Low-recency (recently active) users have smaller positive τ
          (they'd return anyway → less room for uplift)
        - High-recency (dormant) users have larger positive τ
          (most likely to be nudged back)
        - Genre-diverse, high-activity users have modest positive τ
        - Very inactive users (activity_level == 0) have near-zero τ
          (notification probably ineffective)

    Returns a 1-D numpy array of τ values, same length as X.
    """
    n = len(X)
    X_np = X.values if isinstance(X, pd.DataFrame) else X

    # Re-extract named columns for clarity (position-independent)
    cols = list(X.columns) if isinstance(X, pd.DataFrame) else list(range(X_np.shape[1]))
    col_idx = {c: i for i, c in enumerate(cols)}

    recency = X_np[:, col_idx["recency"]].astype(float)
    activity = X_np[:, col_idx["activity_level"]].astype(float)
    genre_entropy = X_np[:, col_idx["genre_entropy"]].astype(float)

    # Normalise to [0, 1] ranges (approximate, robust to outliers)
    recency_norm = np.clip(recency / (recency.max() + 1e-9), 0, 1)
    activity_norm = np.clip(activity / (activity.quantile(0.95) if isinstance(activity, pd.Series)
                                        else np.percentile(activity, 95) + 1e-9), 0, 1)
    entropy_norm = np.clip(genre_entropy / (genre_entropy.max() + 1e-9), 0, 1)

    # τ ∝ recency (dormant users benefit most), tempered by activity
    # Active-but-dormant (high past activity, long recency) → highest τ
    tau = (
        0.30 * recency_norm          # main driver: dormant → high τ
        + 0.10 * activity_norm       # some activity helps (engaged base)
        + 0.05 * entropy_norm        # diverse viewers → slightly more receptive
        - 0.05 * activity_norm * (1 - recency_norm)  # recently-active → already retained
    )

    # Scale so max CATE ~ 0.25 (a realistic lift on a binary outcome)
    tau = tau / (tau.max() + 1e-9) * 0.25

    return tau.astype(float)


# ---------------------------------------------------------------------------
# Propensity model (observational mode)
# ---------------------------------------------------------------------------

def _propensity(X: pd.DataFrame) -> np.ndarray:
    """
    Logistic propensity P(T=1|X).

    High-activity, recently-active users are more likely to receive the
    notification (e.g., platform algorithm targets engaged users).
    Clips to [0.05, 0.95] to avoid extreme weights.
    """
    cols = list(X.columns)
    col_idx = {c: i for i, c in enumerate(cols)}
    X_np = X.values.astype(float)

    recency = X_np[:, col_idx["recency"]]
    activity = X_np[:, col_idx["activity_level"]]

    recency_norm = recency / (recency.max() + 1e-9)
    activity_norm = activity / (np.percentile(activity, 95) + 1e-9)

    # Log-odds: active users (low recency, high activity) are more likely treated
    log_odds = (
        -0.5                             # intercept → ~38% base rate
        + 1.5 * activity_norm            # more active → more likely treated
        - 1.0 * recency_norm             # more dormant → less likely treated
    )

    prop = np.clip(expit(log_odds), 0.05, 0.95)
    return prop


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def simulate_treatment_outcome(
    X: pd.DataFrame,
    mode: str = "observational",
    seed: int = 42,
    baseline_return_rate: float = 0.30,
) -> pd.DataFrame:
    """
    Simulate treatment assignment T and binary outcome Y (return within 7 days).

    Parameters
    ----------
    X                    : User feature matrix from feature_engineering.py.
    mode                 : 'rct' or 'observational'.
    seed                 : Random seed for reproducibility.
    baseline_return_rate : P(Y=1 | T=0, X) base rate (intercept).

    Returns
    -------
    df : X with additional columns:
        - treatment   : int (0/1)
        - outcome     : int (0/1)  -- Y (returned within 7 days)
        - tau_true    : float      -- Ground-truth CATE
        - propensity  : float      -- True P(T=1|X)
    """
    rng = np.random.default_rng(seed)
    X = X.copy()

    tau = _true_cate(X)

    # ---- Treatment assignment ----
    if mode == "rct":
        prop = np.full(len(X), 0.5)
        T = rng.binomial(1, prop)
    elif mode == "observational":
        prop = _propensity(X)
        T = rng.binomial(1, prop)
    else:
        raise ValueError(f"mode must be 'rct' or 'observational', got '{mode}'")

    # ---- Outcome simulation ----
    # Baseline P(Y=1|T=0, X) via logistic regression on features
    X_np = X.values.astype(float)
    # Normalise each column to ~[0,1] for linear combination
    col_ranges = X_np.max(axis=0) - X_np.min(axis=0)
    col_ranges[col_ranges == 0] = 1.0
    X_norm = (X_np - X_np.min(axis=0)) / col_ranges

    # Baseline log-odds (user propensity to return without treatment)
    from scipy.special import logit
    base_logodds = logit(baseline_return_rate)

    # Add feature contributions (small, realistic)
    feature_weights = np.zeros(X_np.shape[1])
    col_names = list(X.columns)
    for name, w in [
        ("activity_level", 0.8),
        ("avg_rating", 0.3),
        ("genre_entropy", 0.2),
        ("recency", -0.6),    # more recency (dormant) → less likely to return naturally
    ]:
        if name in col_names:
            feature_weights[col_names.index(name)] = w

    mu0_logodds = base_logodds + X_norm @ feature_weights  # shape (n,)
    mu0 = expit(mu0_logodds)                               # P(Y=1 | T=0)

    # Treatment effect: add τ(X) to probability, clip to [0,1]
    mu1 = np.clip(mu0 + tau, 0.0, 1.0)                    # P(Y=1 | T=1)

    # Realised outcome
    p_y = np.where(T == 1, mu1, mu0)
    Y = rng.binomial(1, p_y)

    # ---- Value Score (Lifetime Value Proxy) ----
    # Value is only meaningful if the user returns (Y=1)
    # Built from 3 components: 
    # 1. Frequency (activity_level provides a strong baseline)
    # 2. Diversity (genre_entropy adds to the ceiling of engagement)
    # 3. Decay (recent activity implies sustained engagement, high recency implies quick decay)
    
    col_idx = {c: i for i, c in enumerate(col_names)}
    activity = X_np[:, col_idx["activity_level"]]
    entropy = X_np[:, col_idx["genre_entropy"]] if "genre_entropy" in col_idx else np.ones(len(X))
    recency = X_np[:, col_idx["recency"]]
    
    # Scale to typical ranges
    freq_component = np.log1p(activity) * 20
    diversity_component = entropy * 15
    decay_penalty = np.sqrt(recency) * 5
    
    # Non-linear interaction: highly active users who are diverse get a multiplier
    interaction = (activity > np.median(activity)) * (entropy > np.median(entropy)) * 25
    
    # Base value score
    raw_value = freq_component + diversity_component - decay_penalty + interaction
    
    # Add noise to simulate real-world variance (unexplained factors)
    value_noise = rng.normal(0, 10, size=len(X))
    final_value = np.clip(raw_value + value_noise, 0, 200)
    
    # Value is zero if they churned (did not return)
    value_score = final_value * Y

    # ---- Package results ----
    df = X.copy()
    df["treatment"] = T.astype(int)
    df["outcome"] = Y.astype(int)
    df["value_score"] = value_score
    df["tau_true"] = tau
    df["propensity"] = prop

    ate_true = tau.mean()
    print(
        f"[simulate_rct] mode={mode} | n={len(df):,} | "
        f"P(T=1)={T.mean():.3f} | P(Y=1)={Y.mean():.3f} | "
        f"True ATE={ate_true:.4f} | Mean Value (Retained)={value_score[Y==1].mean():.1f}"
    )
    return df
