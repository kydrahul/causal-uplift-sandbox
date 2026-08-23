"""
doubleml_estimator.py
---------------------
Fits a CausalForestDML (Double ML) estimator directly via EconML and runs
a manual refutation suite equivalent to dowhy's, but without the
dowhy/networkx version dependency.

The four refutation tests implemented are identical in purpose to the
dowhy API originals:
  1. placebo_treatment  -- replace T with random permutation → CATE → 0
  2. random_common_cause -- add noise confounder → estimate stays stable
  3. data_subset         -- bootstrap 80% subsets → estimate stays stable
  4. unobserved_confounder -- flip a fraction of T labels → sensitivity check

Usage:
    from src.doubleml_estimator import run_doubleml, run_refutations

    dml_result = run_doubleml(df, feature_cols)
    refutation_results = run_refutations(df, feature_cols, dml_result)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Double ML estimation (pure EconML, no dowhy dependency)
# ---------------------------------------------------------------------------

def run_doubleml(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_estimators: int = 500,
    seed: int = 42,
) -> dict:
    """
    Fit CausalForestDML directly via EconML.

    Parameters
    ----------
    df           : DataFrame with 'treatment' and 'outcome' columns + feature_cols.
    feature_cols : List of covariate column names.
    n_estimators : Number of trees in CausalForestDML.
    seed         : Random seed.

    Returns
    -------
    dict with keys:
        tau_hat   : np.ndarray -- per-user CATE estimates
        ate       : float -- average treatment effect
        ate_ci    : tuple (lower, upper) -- 95% CI on ATE
        dml_model : fitted EconML CausalForestDML object
    """
    from econml.dml import CausalForestDML
    from lightgbm import LGBMRegressor, LGBMClassifier

    X = df[feature_cols].values.astype(float)
    T = df["treatment"].values.astype(float)
    Y = df["outcome"].values.astype(float)

    print("[doubleml] Fitting CausalForestDML ...")
    dml = CausalForestDML(
        model_y=LGBMRegressor(n_estimators=200, verbose=-1, random_state=seed),
        model_t=LGBMClassifier(n_estimators=200, verbose=-1, random_state=seed),
        n_estimators=n_estimators,
        min_samples_leaf=20,
        random_state=seed,
        discrete_treatment=True,
    )
    dml.fit(Y, T, X=X)

    # Per-user CATE
    tau_hat = dml.effect(X).flatten()
    ate = float(tau_hat.mean())

    # 95% CI on ATE via EconML inference
    try:
        dml.tune(Y, T, X=X)
        inference = dml.effect_interval(X, alpha=0.05)
        ate_ci = (float(inference[0].mean()), float(inference[1].mean()))
    except Exception:
        # Fallback: bootstrap CI on ATE
        rng = np.random.default_rng(seed)
        boot_ates = []
        n = len(X)
        for _ in range(100):
            idx = rng.integers(0, n, size=n)
            boot_ates.append(dml.effect(X[idx]).mean())
        ate_ci = (float(np.percentile(boot_ates, 2.5)),
                  float(np.percentile(boot_ates, 97.5)))

    print(
        f"[doubleml] Done | ATE = {ate:.4f} "
        f"| 95% CI = [{ate_ci[0]:.4f}, {ate_ci[1]:.4f}]"
    )

    return {
        "tau_hat": tau_hat,
        "ate": ate,
        "ate_ci": ate_ci,
        "dml_model": dml,
    }


# ---------------------------------------------------------------------------
# Refutation suite (manual implementation, dowhy-equivalent)
# ---------------------------------------------------------------------------

def _fit_dml(X, T, Y, n_estimators: int = 100, seed: int = 0):
    """Helper: fit a fresh CausalForestDML and return (model, ate)."""
    from econml.dml import CausalForestDML
    from lightgbm import LGBMRegressor, LGBMClassifier

    dml = CausalForestDML(
        model_y=LGBMRegressor(n_estimators=100, verbose=-1, random_state=seed),
        model_t=LGBMClassifier(n_estimators=100, verbose=-1, random_state=seed),
        n_estimators=n_estimators,
        min_samples_leaf=10,
        random_state=seed,
        discrete_treatment=True,
    )
    dml.fit(Y, T, X=X)
    ate = float(dml.effect(X).mean())
    return dml, ate


def run_refutations(
    df: pd.DataFrame,
    feature_cols: list[str],
    dml_result: dict,
    n_simulations: int = 30,
    seed: int = 42,
) -> dict:
    """
    Run four refutation tests on the DoubleML estimate.

    Tests and their interpretations:
    ─────────────────────────────────────────────────────────────────────
    1. placebo_treatment : Replace T with random permutation.
       Expected: new ATE ≈ 0 (model responds to real treatment signal).
       Passed if |new_ATE| < 0.5 × |original_ATE|.

    2. random_common_cause : Add a pure-noise covariate.
       Expected: ATE stays close to original (robust to spurious confounders).
       Passed if |new_ATE - original_ATE| < 0.1 × |original_ATE| + 0.005.

    3. data_subset : Refit on 80% random subsets (bootstrap stability).
       Expected: mean(subset ATEs) ≈ original ATE.
       Passed if mean |subset_ATE - original_ATE| < 0.05.

    4. unobserved_confounder : Flip 10% of treatment labels (sensitivity).
       Expected: ATE doesn't change dramatically.
       Passed if |new_ATE - original_ATE| < 0.1.
    ─────────────────────────────────────────────────────────────────────

    Returns
    -------
    dict keyed by refuter name; each value has:
        new_effect  : float
        p_value     : float (where applicable, else nan)
        passed      : bool
        summary     : str
    """
    rng = np.random.default_rng(seed)
    original_ate = dml_result["ate"]

    X = df[feature_cols].values.astype(float)
    T = df["treatment"].values.astype(float)
    Y = df["outcome"].values.astype(float)
    n = len(X)

    results = {}

    # ── 1. Placebo treatment ────────────────────────────────────────────
    print("\n[doubleml refutation] 1/4 -- Placebo treatment refuter ...")
    placebo_ates = []
    for i in range(n_simulations):
        T_placebo = rng.permutation(T)
        _, ate_p = _fit_dml(X, T_placebo, Y, seed=seed + i)
        placebo_ates.append(ate_p)

    placebo_mean = float(np.mean(placebo_ates))
    # p-value: fraction of placebo ATEs that are as large as original
    pval_placebo = float(np.mean(np.abs(placebo_ates) >= abs(original_ate)))
    passed_placebo = abs(placebo_mean) < 0.5 * abs(original_ate)

    results["placebo_treatment"] = {
        "new_effect": placebo_mean,
        "p_value": pval_placebo,
        "passed": passed_placebo,
        "summary": (
            f"Placebo ATE = {placebo_mean:.4f} (original = {original_ate:.4f}) | "
            f"p = {pval_placebo:.3f} | passed = {passed_placebo}"
        ),
    }
    print(f"   {results['placebo_treatment']['summary']}")

    # ── 2. Random common cause ──────────────────────────────────────────
    print("[doubleml refutation] 2/4 -- Random common cause refuter ...")
    rcc_ates = []
    for i in range(n_simulations):
        noise = rng.normal(0, 1, size=(n, 1))
        X_noisy = np.concatenate([X, noise], axis=1)
        _, ate_r = _fit_dml(X_noisy, T, Y, seed=seed + i)
        rcc_ates.append(ate_r)

    rcc_mean = float(np.mean(rcc_ates))
    rcc_shift = abs(rcc_mean - original_ate)
    pval_rcc = float(np.mean(np.abs(np.array(rcc_ates) - original_ate)
                              >= rcc_shift))
    passed_rcc = rcc_shift < 0.1 * abs(original_ate) + 0.005

    results["random_common_cause"] = {
        "new_effect": rcc_mean,
        "p_value": pval_rcc,
        "passed": passed_rcc,
        "summary": (
            f"RCC ATE = {rcc_mean:.4f} | shift = {rcc_shift:.4f} | "
            f"p = {pval_rcc:.3f} | passed = {passed_rcc}"
        ),
    }
    print(f"   {results['random_common_cause']['summary']}")

    # ── 3. Data subset refuter ──────────────────────────────────────────
    print("[doubleml refutation] 3/4 -- Data subset refuter ...")
    subset_ates = []
    for i in range(n_simulations):
        idx = rng.choice(n, size=int(0.8 * n), replace=False)
        _, ate_s = _fit_dml(X[idx], T[idx], Y[idx], seed=seed + i)
        subset_ates.append(ate_s)

    subset_mean = float(np.mean(subset_ates))
    subset_shift = float(np.mean(np.abs(np.array(subset_ates) - original_ate)))
    pval_subset = float(np.mean(np.abs(np.array(subset_ates) - original_ate)
                                >= abs(subset_mean - original_ate)))
    passed_subset = subset_shift < 0.05

    results["data_subset"] = {
        "new_effect": subset_mean,
        "p_value": pval_subset,
        "passed": passed_subset,
        "summary": (
            f"Subset ATE = {subset_mean:.4f} | mean shift = {subset_shift:.4f} | "
            f"p = {pval_subset:.3f} | passed = {passed_subset}"
        ),
    }
    print(f"   {results['data_subset']['summary']}")

    # ── 4. Unobserved confounder sensitivity ────────────────────────────
    print("[doubleml refutation] 4/4 -- Unobserved confounder sensitivity ...")
    flip_rate = 0.10
    flip_ates = []
    for i in range(n_simulations):
        T_flipped = T.copy()
        flip_idx = rng.choice(n, size=int(flip_rate * n), replace=False)
        T_flipped[flip_idx] = 1 - T_flipped[flip_idx]
        _, ate_f = _fit_dml(X, T_flipped, Y, seed=seed + i)
        flip_ates.append(ate_f)

    flip_mean = float(np.mean(flip_ates))
    flip_shift = abs(flip_mean - original_ate)
    passed_flip = flip_shift < 0.10

    results["unobserved_confounder"] = {
        "new_effect": flip_mean,
        "p_value": float("nan"),  # sensitivity analysis, not hypothesis test
        "passed": passed_flip,
        "summary": (
            f"Flip-10% ATE = {flip_mean:.4f} | shift = {flip_shift:.4f} | "
            f"passed = {passed_flip}"
        ),
    }
    print(f"   {results['unobserved_confounder']['summary']}")

    return results
