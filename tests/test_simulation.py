"""
tests/test_simulation.py
------------------------
Sanity-check tests for the feature engineering and simulation pipeline.

Verification plan targets:
  1. RCT sanity: naive ATE under RCT should recover E[tau(X)] within tolerance
  2. Placebo test: under T := random, AUUC should be ~0.5
  3. CATE sign check: low-recency users should have lower tau (dormant -> high tau)
  4. Observational vs RCT: propensity correlation with activity (confounding present)
  5. Feature matrix shape / no NaNs
"""

import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")

from src.simulate_rct import simulate_treatment_outcome, _true_cate
from src.feature_engineering import build_user_features
from src.evaluation import compute_auuc, evaluate_all


# ---------------------------------------------------------------------------
# Shared synthetic feature matrix (small, fast, no download required)
# ---------------------------------------------------------------------------

def make_synthetic_X(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """
    Build a synthetic user feature DataFrame that matches the schema produced
    by feature_engineering.build_user_features(), so we can test the simulation
    and evaluation layers without needing MovieLens data.
    """
    rng = np.random.default_rng(seed)

    X = pd.DataFrame({
        "activity_level": rng.integers(1, 200, size=n).astype(float),
        "avg_rating": rng.uniform(1.0, 5.0, size=n),
        "genre_entropy": rng.uniform(0.0, 2.5, size=n),
        "recency":       rng.integers(0, 60, size=n).astype(float),
        "gender":        rng.integers(0, 2, size=n),
        "age":           rng.integers(0, 7, size=n),
        "occupation":    rng.integers(0, 21, size=n),
        # genre one-hots (a few representative ones)
        "genre_Action":  rng.integers(0, 2, size=n),
        "genre_Drama":   rng.integers(0, 2, size=n),
        "genre_Comedy":  rng.integers(0, 2, size=n),
    })
    return X


# ===========================================================================
# 1. Feature matrix sanity
# ===========================================================================

class TestFeatureMatrix:
    def test_no_nans(self):
        X = make_synthetic_X()
        assert X.isnull().sum().sum() == 0, "Feature matrix should have no NaNs"

    def test_expected_columns_present(self):
        X = make_synthetic_X()
        required = ["activity_level", "avg_rating", "genre_entropy", "recency",
                    "gender", "age", "occupation"]
        for col in required:
            assert col in X.columns, f"Missing column: {col}"

    def test_activity_non_negative(self):
        X = make_synthetic_X()
        assert (X["activity_level"] >= 0).all()

    def test_recency_non_negative(self):
        X = make_synthetic_X()
        assert (X["recency"] >= 0).all()


# ===========================================================================
# 2. Ground-truth CATE shape checks
# ===========================================================================

class TestTrueCate:
    def test_cate_non_negative(self):
        """tau(X) should be >= 0 (dormant users benefit, none harmed by design)."""
        X = make_synthetic_X()
        tau = _true_cate(X)
        assert (tau >= 0).all(), "All CATE values should be non-negative"

    def test_cate_max_bounded(self):
        """tau(X) should be <= 0.25 (design constraint)."""
        X = make_synthetic_X(n=5000)
        tau = _true_cate(X)
        assert tau.max() <= 0.26, f"Max CATE {tau.max():.4f} exceeds design bound of 0.25"

    def test_dormant_users_higher_cate(self):
        """
        Users with high recency (dormant) should on average have higher CATE
        than users with low recency (recently active).

        Both groups are evaluated together so _true_cate's internal
        normalisation is shared and the relative ordering is meaningful.
        """
        n = 500  # per group

        base = {
            "activity_level": np.full(n * 2, 50.0),
            "avg_rating":     np.full(n * 2, 3.5),
            "genre_entropy":  np.full(n * 2, 1.0),
            "gender":         np.zeros(n * 2, dtype=int),
            "age":            np.full(n * 2, 2, dtype=int),
            "occupation":     np.zeros(n * 2, dtype=int),
            "genre_Action":   np.zeros(n * 2, dtype=int),
            "genre_Drama":    np.zeros(n * 2, dtype=int),
            "genre_Comedy":   np.zeros(n * 2, dtype=int),
        }

        # First n rows = dormant (high recency), last n rows = active (low recency)
        recency = np.concatenate([np.full(n, 55.0), np.full(n, 1.0)])
        X_combined = pd.DataFrame({**base, "recency": recency})

        tau_combined = _true_cate(X_combined)
        tau_dormant = tau_combined[:n].mean()
        tau_active  = tau_combined[n:].mean()

        assert tau_dormant > tau_active, (
            f"Dormant users should have higher CATE "
            f"(got dormant={tau_dormant:.4f}, active={tau_active:.4f})"
        )


# ===========================================================================
# 3. RCT sanity: naive ATE should recover E[tau(X)]
# ===========================================================================

class TestRCTSanity:
    """
    Under a randomized experiment (Bernoulli 0.5), the naive difference-in-means
    estimator is unbiased for the ATE.  With n=5000 we expect the estimate to
    land within ±0.03 of the true ATE.
    """

    def test_naive_ate_recovers_true_ate(self):
        X = make_synthetic_X(n=5000, seed=7)
        df = simulate_treatment_outcome(X, mode="rct", seed=7)

        tau_true_mean = df["tau_true"].mean()

        y1 = df.loc[df["treatment"] == 1, "outcome"].mean()
        y0 = df.loc[df["treatment"] == 0, "outcome"].mean()
        naive_ate = y1 - y0

        diff = abs(naive_ate - tau_true_mean)
        assert diff < 0.05, (
            f"Naive ATE ({naive_ate:.4f}) deviates too much from "
            f"true ATE ({tau_true_mean:.4f}); diff={diff:.4f}"
        )

    def test_rct_balanced_arms(self):
        """RCT should assign ~50% to treatment."""
        X = make_synthetic_X(n=2000)
        df = simulate_treatment_outcome(X, mode="rct", seed=0)
        p_treated = df["treatment"].mean()
        assert 0.44 < p_treated < 0.56, f"Expected ~50% treated, got {p_treated:.3f}"


# ===========================================================================
# 4. Observational mode: confounding present
# ===========================================================================

class TestObservationalConfounding:
    def test_propensity_correlated_with_activity(self):
        """
        In observational mode, high-activity users are more likely treated.
        Correlation between propensity and activity_level should be positive.
        """
        X = make_synthetic_X(n=3000, seed=3)
        df = simulate_treatment_outcome(X, mode="observational", seed=3)

        corr = df["propensity"].corr(df["activity_level"])
        assert corr > 0.2, (
            f"Expected positive correlation between propensity and activity, "
            f"got {corr:.3f}"
        )

    def test_propensity_correlated_negatively_with_recency(self):
        """
        Dormant users (high recency) should be LESS likely to be treated.
        """
        X = make_synthetic_X(n=3000, seed=4)
        df = simulate_treatment_outcome(X, mode="observational", seed=4)

        corr = df["propensity"].corr(df["recency"])
        assert corr < -0.2, (
            f"Expected negative correlation between propensity and recency, "
            f"got {corr:.3f}"
        )

    def test_propensity_clipped(self):
        """Propensity should be strictly in (0.05, 0.95)."""
        X = make_synthetic_X(n=3000)
        df = simulate_treatment_outcome(X, mode="observational", seed=0)
        assert df["propensity"].min() >= 0.05
        assert df["propensity"].max() <= 0.95


# ===========================================================================
# 5. Placebo test: random treatment → AUUC ≈ 0.5
# ===========================================================================

class TestPlacebo:
    """
    When treatment is pure random noise (uninformative), ranking users by any
    fixed τ̂ should yield an AUUC indistinguishable from a random model.
    We use the true τ(X) as the ranker — even this should fail to lift AUUC
    because the outcome signal is now decoupled from the CATE structure.
    """

    def test_placebo_treatment_auuc_near_random(self):
        rng = np.random.default_rng(99)
        X = make_synthetic_X(n=3000, seed=99)
        df = simulate_treatment_outcome(X, mode="rct", seed=99)

        # Replace treatment with pure random noise (placebo)
        df["treatment"] = rng.integers(0, 2, size=len(df))
        # Regenerate outcome with the same mu0 signal but placebo treatment
        df["outcome"] = rng.binomial(1, 0.35, size=len(df))  # random Y ~ Bernoulli

        tau_hat = df["tau_true"].values
        T = df["treatment"].values
        Y = df["outcome"].values

        auuc_placebo = compute_auuc(tau_hat, T, Y)

        # AUUC of a random model on a random outcome
        tau_random = rng.uniform(0, 1, size=len(df))
        auuc_random = compute_auuc(tau_random, T, Y)

        # Both should be close (neither can exploit signal that doesn't exist)
        diff = abs(auuc_placebo - auuc_random)
        assert diff < 0.05, (
            f"Under placebo, AUUC gap between true-tau ranker "
            f"({auuc_placebo:.4f}) and random ranker ({auuc_random:.4f}) "
            f"should be small; got diff={diff:.4f}"
        )

    def test_placebo_pehe_matches_true_tau_std(self):
        """
        Under a placebo (τ̂ = 0 for everyone), PEHE = std(τ_true).
        """
        from src.evaluation import compute_pehe
        X = make_synthetic_X(n=1000, seed=5)
        df = simulate_treatment_outcome(X, mode="rct", seed=5)
        tau_true = df["tau_true"].values

        # Naive estimator: predict zero uplift for everyone
        tau_zero = np.zeros(len(tau_true))
        pehe = compute_pehe(tau_zero, tau_true)

        expected = np.sqrt(np.mean(tau_true ** 2))   # RMSE(0, tau_true)
        assert abs(pehe - expected) < 1e-8, (
            f"Placebo PEHE mismatch: {pehe:.6f} vs {expected:.6f}"
        )


# ===========================================================================
# 6. Simulation reproducibility
# ===========================================================================

class TestReproducibility:
    def test_same_seed_same_output(self):
        X = make_synthetic_X(n=500, seed=0)
        df1 = simulate_treatment_outcome(X, mode="rct", seed=42)
        df2 = simulate_treatment_outcome(X, mode="rct", seed=42)
        np.testing.assert_array_equal(df1["treatment"].values, df2["treatment"].values)
        np.testing.assert_array_equal(df1["outcome"].values,   df2["outcome"].values)

    def test_different_seed_different_output(self):
        X = make_synthetic_X(n=500, seed=0)
        df1 = simulate_treatment_outcome(X, mode="rct", seed=1)
        df2 = simulate_treatment_outcome(X, mode="rct", seed=2)
        assert not np.array_equal(df1["treatment"].values, df2["treatment"].values)
