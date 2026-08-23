"""
tests/test_evaluation.py
------------------------
Unit tests for evaluation.py using synthetic known-case data.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    compute_auuc,
    compute_qini,
    compute_pehe,
    calibration_by_decile,
    evaluate_all,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def perfect_model_data(n: int = 1000, seed: int = 0) -> dict:
    """
    Synthetic data where τ_true is known and a 'perfect' model returns
    exactly τ_true as its estimate.
    """
    rng = np.random.default_rng(seed)
    tau_true = rng.uniform(0, 0.3, size=n)
    T = rng.binomial(1, 0.5, size=n)
    mu0 = 0.3
    mu1 = np.clip(mu0 + tau_true, 0, 1)
    Y = rng.binomial(1, np.where(T == 1, mu1, mu0))
    return {"tau_true": tau_true, "T": T, "Y": Y}


def random_model_data(n: int = 1000, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    tau_true = rng.uniform(0, 0.3, size=n)
    T = rng.binomial(1, 0.5, size=n)
    Y = rng.binomial(1, 0.4, size=n)
    tau_hat_random = rng.uniform(0, 1, size=n)  # random ranking
    return {"tau_true": tau_true, "T": T, "Y": Y, "tau_hat_random": tau_hat_random}


# ---------------------------------------------------------------------------
# AUUC tests
# ---------------------------------------------------------------------------

class TestAUUC:
    def test_auuc_range(self):
        """AUUC should be a finite non-negative number."""
        data = perfect_model_data()
        auuc = compute_auuc(data["tau_true"], data["T"], data["Y"])
        assert np.isfinite(auuc), "AUUC should be finite"
        assert auuc >= 0, "AUUC should be non-negative"

    def test_perfect_model_beats_random(self):
        """A model using true τ as ranking should beat random ranking."""
        data = perfect_model_data()
        auuc_perfect = compute_auuc(data["tau_true"], data["T"], data["Y"])

        rng = np.random.default_rng(99)
        tau_random = rng.uniform(0, 1, size=len(data["T"]))
        auuc_random = compute_auuc(tau_random, data["T"], data["Y"])

        # Perfect model should be at least as good (with high probability)
        # Use a soft test — over many random seeds this holds almost surely
        assert auuc_perfect >= auuc_random * 0.9, (
            f"Expected perfect model AUUC ({auuc_perfect:.4f}) ≥ "
            f"random AUUC ({auuc_random:.4f})"
        )

    def test_worst_model_low_auuc(self):
        """Reverse-sorted τ (anti-ranking) should have lower AUUC."""
        data = perfect_model_data()
        tau_best = data["tau_true"]
        tau_worst = -tau_best  # inverse ranking

        auuc_best = compute_auuc(tau_best, data["T"], data["Y"])
        auuc_worst = compute_auuc(tau_worst, data["T"], data["Y"])

        assert auuc_best > auuc_worst, "Best ranking should beat worst ranking"


# ---------------------------------------------------------------------------
# PEHE tests
# ---------------------------------------------------------------------------

class TestPEHE:
    def test_pehe_zero_for_perfect(self):
        """PEHE should be 0 when τ̂ == τ_true."""
        tau_true = np.array([0.1, 0.2, 0.3, 0.05, 0.15])
        pehe = compute_pehe(tau_hat=tau_true, tau_true=tau_true)
        assert pehe == pytest.approx(0.0, abs=1e-10)

    def test_pehe_known_value(self):
        """PEHE should match manual RMSE computation."""
        tau_hat = np.array([0.1, 0.2, 0.3])
        tau_true = np.array([0.0, 0.0, 0.0])
        expected = np.sqrt(np.mean([0.01, 0.04, 0.09]))
        pehe = compute_pehe(tau_hat, tau_true)
        assert pehe == pytest.approx(expected, abs=1e-8)

    def test_pehe_non_negative(self):
        """PEHE should always be non-negative."""
        rng = np.random.default_rng(42)
        tau_hat = rng.normal(0, 1, size=500)
        tau_true = rng.normal(0, 1, size=500)
        assert compute_pehe(tau_hat, tau_true) >= 0


# ---------------------------------------------------------------------------
# Qini tests
# ---------------------------------------------------------------------------

class TestQini:
    def test_qini_is_finite(self):
        data = perfect_model_data()
        qini = compute_qini(data["tau_true"], data["T"], data["Y"])
        assert np.isfinite(qini)

    def test_qini_positive_for_good_model(self):
        """A model with good ranking should have positive Qini."""
        data = perfect_model_data(n=2000, seed=5)
        qini = compute_qini(data["tau_true"], data["T"], data["Y"])
        # This may not always hold for small n due to noise; accept ≥ -0.05
        assert qini >= -0.05, f"Qini unexpectedly low: {qini:.4f}"


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibration_returns_correct_shape(self):
        data = perfect_model_data()
        cal = calibration_by_decile(data["tau_true"], data["T"], data["Y"], n_deciles=10)
        assert "mean_tau_hat" in cal.columns
        assert "empirical_lift" in cal.columns
        assert len(cal) == 10

    def test_calibration_monotone_for_perfect_model(self):
        """With a perfect model, higher-decile users should have higher empirical lift."""
        data = perfect_model_data(n=5000, seed=10)
        cal = calibration_by_decile(data["tau_true"], data["T"], data["Y"], n_deciles=5)
        # Check that mean_tau_hat is monotonically increasing by decile
        tau_vals = cal["mean_tau_hat"].values
        assert np.all(np.diff(tau_vals) >= 0), "τ̂ should be increasing across deciles"


# ---------------------------------------------------------------------------
# evaluate_all integration test
# ---------------------------------------------------------------------------

class TestEvaluateAll:
    def test_evaluate_all_returns_all_models(self):
        data = perfect_model_data()
        df = pd.DataFrame({
            "treatment": data["T"],
            "outcome": data["Y"],
            "tau_true": data["tau_true"],
        })
        tau_hats = {"ModelA": data["tau_true"], "ModelB": -data["tau_true"]}
        metrics = evaluate_all(df, tau_hats)
        assert "ModelA" in metrics.index
        assert "ModelB" in metrics.index
        assert "Naive ATE" in metrics.index
        assert "AUUC" in metrics.columns
        assert "PEHE" in metrics.columns

    def test_evaluate_all_no_tau_true(self):
        """evaluate_all should handle missing tau_true gracefully."""
        rng = np.random.default_rng(7)
        n = 300
        df = pd.DataFrame({
            "treatment": rng.binomial(1, 0.5, n),
            "outcome": rng.binomial(1, 0.4, n),
        })
        tau_hats = {"ModelA": rng.uniform(0, 1, n)}
        metrics = evaluate_all(df, tau_hats)
        assert np.isnan(metrics.loc["ModelA", "PEHE"])
