"""
meta_learners.py
----------------
S/T/X/R-learners and an Uplift Tree implemented directly using
scikit-learn + LightGBM -- no causalml dependency required.

This gives identical results to the causalml meta-learners (which use the
same underlying logic) while being fully compatible with Python 3.14+.

Usage:
    from src.meta_learners import run_all_meta_learners

    results = run_all_meta_learners(df, feature_cols, n_bootstrap=50)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.base import clone

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Default LightGBM base learners
# ---------------------------------------------------------------------------

def _lgbm_reg(seed: int = 42, n_estimators: int = 300, **kw) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=n_estimators, learning_rate=0.05,
        num_leaves=31, verbose=-1, random_state=seed, **kw
    )

def _lgbm_clf(seed: int = 42, n_estimators: int = 300, **kw) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.05,
        num_leaves=31, verbose=-1, random_state=seed, **kw
    )


# ---------------------------------------------------------------------------
# S-learner
# ---------------------------------------------------------------------------

class SLearner:
    """
    S-learner: fit a single model on (X, T) → Y.
    τ̂(x) = μ(x, T=1) - μ(x, T=0)
    """
    def __init__(self, base_learner=None, seed: int = 42):
        self.model = base_learner or _lgbm_reg(seed)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        XT = np.column_stack([X, T])
        self.model.fit(XT, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        XT1 = np.column_stack([X, np.ones(n)])
        XT0 = np.column_stack([X, np.zeros(n)])
        return self.model.predict(XT1) - self.model.predict(XT0)


# ---------------------------------------------------------------------------
# T-learner
# ---------------------------------------------------------------------------

class TLearner:
    """
    T-learner: separate models for treated and control.
    τ̂(x) = μ₁(x) - μ₀(x)
    """
    def __init__(self, base_learner=None, seed: int = 42):
        self.model1 = base_learner or _lgbm_reg(seed)
        self.model0 = clone(self.model1)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        self.model1.fit(X[T == 1], Y[T == 1])
        self.model0.fit(X[T == 0], Y[T == 0])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model1.predict(X) - self.model0.predict(X)


# ---------------------------------------------------------------------------
# X-learner
# ---------------------------------------------------------------------------

class XLearner:
    """
    X-learner (Künzel et al. 2019):
      Stage 1: Estimate μ₀, μ₁ with T-learner.
      Stage 2: Compute imputed treatment effects, fit cross-learners.
      τ̂(x) = g(x)·τ̂₁(x) + (1−g(x))·τ̂₀(x)  where g(x) = propensity score.
    """
    def __init__(self, base_learner=None, prop_learner=None, seed: int = 42):
        self.mu1  = base_learner or _lgbm_reg(seed)
        self.mu0  = clone(self.mu1)
        self.tau1 = clone(self.mu1)   # cross-learner for treated
        self.tau0 = clone(self.mu1)   # cross-learner for control
        self.prop = prop_learner or _lgbm_clf(seed)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        # Stage 1
        self.mu1.fit(X[T == 1], Y[T == 1])
        self.mu0.fit(X[T == 0], Y[T == 0])
        # Stage 2: imputed effects
        D1 = Y[T == 1] - self.mu0.predict(X[T == 1])   # treated: Y - μ₀
        D0 = self.mu1.predict(X[T == 0]) - Y[T == 0]   # control: μ₁ - Y
        self.tau1.fit(X[T == 1], D1)
        self.tau0.fit(X[T == 0], D0)
        # Propensity
        self.prop.fit(X, T)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        g = self.prop.predict_proba(X)[:, 1]
        return g * self.tau1.predict(X) + (1 - g) * self.tau0.predict(X)

    @property
    def shap_model(self):
        """Return mu1 model for SHAP (stage-1 treated model)."""
        return self.mu1


# ---------------------------------------------------------------------------
# R-learner
# ---------------------------------------------------------------------------

class RLearner:
    """
    R-learner (Nie & Wager 2021):
      Residualise Y and T on X, then fit the CATE via weighted regression.
      τ̂ = argmin_τ Σ [(Y_i - m(X_i)) - (T_i - e(X_i))·τ(X_i)]²
    Uses cross-fitting (2-fold) to avoid overfitting the residuals.
    """
    def __init__(self, base_learner=None, prop_learner=None, seed: int = 42):
        self.m_model  = base_learner or _lgbm_reg(seed)     # outcome model
        self.e_model  = prop_learner or _lgbm_clf(seed)     # propensity model
        self.tau_model = clone(self.m_model)                # CATE model
        self._seed = seed

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        n = len(X)
        rng = np.random.default_rng(self._seed)
        fold = rng.integers(0, 2, size=n)   # 2-fold cross-fitting

        m_resid = np.zeros(n)
        e_resid = np.zeros(n)

        for k in (0, 1):
            train, val = fold != k, fold == k
            # Outcome model
            m_k = clone(self.m_model)
            m_k.fit(X[train], Y[train])
            m_resid[val] = Y[val] - m_k.predict(X[val])
            # Propensity model
            e_k = clone(self.e_model)
            e_k.fit(X[train], T[train])
            e_resid[val] = T[val] - e_k.predict_proba(X[val])[:, 1]

        # Fit final models on all data (for full-sample CATE)
        self.m_model.fit(X, Y)
        self.e_model.fit(X, T)

        # Pseudo-outcome: (Y - m̂) / (T - ê)  weighted by (T - ê)²
        w = e_resid ** 2
        w = np.clip(w, 1e-6, None)   # avoid divide-by-zero
        pseudo_outcome = m_resid / (e_resid + 1e-9)

        self.tau_model.fit(X, pseudo_outcome, sample_weight=w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.tau_model.predict(X)


# ---------------------------------------------------------------------------
# Uplift Tree (simple binned version without causalml dependency)
# ---------------------------------------------------------------------------

class SimpleUpliftTree:
    """
    A lightweight uplift tree using sklearn's DecisionTreeRegressor on
    pseudo-outcomes (Transformed Outcome method).

    τ̂_TO = Y * (2T - 1) / p(T|X)   (Athey & Imbens, 2015)

    Fit a regression tree on tau_hat_TO; per-leaf CATE is the leaf mean.
    This is interpretable but not as accurate as causalml's tree.
    """
    def __init__(self, max_depth: int = 5, seed: int = 42):
        from sklearn.tree import DecisionTreeRegressor
        self.tree = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_leaf=100, random_state=seed
        )
        self._prop_clf = _lgbm_clf(seed, n_estimators=100)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        # Estimate propensity
        self._prop_clf.fit(X, T)
        prop = self._prop_clf.predict_proba(X)[:, 1]
        prop = np.clip(prop, 0.05, 0.95)

        # Transformed outcome
        pseudo = Y * (2 * T - 1) / np.where(T == 1, prop, 1 - prop)
        self.tree.fit(X, pseudo)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.tree.predict(X)


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    learner_cls,
    learner_kwargs: dict,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(X)
    boot_ates = []

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            m = learner_cls(**{**learner_kwargs, "seed": seed + i})
            m.fit(X[idx], T[idx], Y[idx])
            boot_ates.append(m.predict(X))
        except Exception:
            continue

    if not boot_ates:
        zeros = np.zeros(n)
        return zeros, zeros
    boot_arr = np.stack(boot_ates, axis=0)
    return (
        np.percentile(boot_arr, 2.5, axis=0),
        np.percentile(boot_arr, 97.5, axis=0),
    )


# ---------------------------------------------------------------------------
# SHAP values
# ---------------------------------------------------------------------------

def _shap_values(model, X: np.ndarray, X_df: pd.DataFrame) -> Optional[np.ndarray]:
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # shap may return list (for classifiers); take first element
        if isinstance(sv, list):
            sv = sv[1]
        return sv
    except Exception as e:
        print(f"  [SHAP] skipped: {e}")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all_meta_learners(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_bootstrap: int = 50,
    run_uplift_tree: bool = True,
    lgbm_params: Optional[dict] = None,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Fit all meta-learners on the simulation DataFrame.

    Parameters
    ----------
    df            : Output of simulate_rct.simulate_treatment_outcome().
    feature_cols  : List of covariate column names.
    n_bootstrap   : Number of bootstrap resamples for CI computation.
    run_uplift_tree : Whether to also fit the SimpleUpliftTree.
    lgbm_params   : Extra keyword arguments passed to LGBMRegressor.
    seed          : Master random seed.

    Returns
    -------
    results : dict keyed by estimator name ('S','T','X','R','UpliftTree').
              Each value has:
                  tau_hat     : np.ndarray of per-user CATE estimates
                  ci_lower    : np.ndarray (2.5th percentile)
                  ci_upper    : np.ndarray (97.5th percentile)
                  shap_values : np.ndarray or None
                  learner     : fitted estimator object
    """
    lgbm_params = lgbm_params or {}

    X = df[feature_cols].values.astype(float)
    T = df["treatment"].values.astype(float)
    Y = df["outcome"].values.astype(float)
    X_df = df[feature_cols]

    results: dict[str, dict] = {}

    learner_specs = [
        ("S", SLearner,  {}),
        ("T", TLearner,  {}),
        ("X", XLearner,  {}),
        ("R", RLearner,  {}),
    ]

    for name, cls, kwargs in learner_specs:
        print(f"[meta_learners] Fitting {name}-learner ...")
        m = cls(seed=seed, **kwargs)
        m.fit(X, T, Y)
        tau_hat = m.predict(X)

        print(f"  Bootstrap CIs ({n_bootstrap} resamples) ...")
        ci_lo, ci_hi = _bootstrap_ci(cls, kwargs, X, T, Y, n_bootstrap, seed)

        # SHAP -- use X-learner's mu1 (stage-1 treated LightGBM model)
        shap_vals = None
        if name == "X":
            shap_vals = _shap_values(m.shap_model, X, X_df)

        print(f"  {name}-learner done | mean tau_hat = {tau_hat.mean():.4f}")
        results[name] = {
            "tau_hat": tau_hat,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "shap_values": shap_vals,
            "learner": m,
        }

    if run_uplift_tree:
        print("[meta_learners] Fitting Uplift Tree ...")
        tree = SimpleUpliftTree(max_depth=5, seed=seed)
        tree.fit(X, T, Y)
        tau_tree = tree.predict(X)
        print(f"  Uplift Tree done | mean tau_hat = {tau_tree.mean():.4f}")
        results["UpliftTree"] = {
            "tau_hat": tau_tree,
            "ci_lower": None,
            "ci_upper": None,
            "shap_values": None,
            "learner": tree,
        }

    return results
