"""
evaluation.py
-------------
Uplift model evaluation metrics:
    - AUUC  (Area Under Uplift Curve) -- primary metric
    - Qini coefficient (normalized AUUC variant)
    - PEHE (Precision in Estimation of HTE) -- requires ground-truth τ
    - Calibration: mean predicted CATE vs. empirical lift by decile

Usage:
    from src.evaluation import evaluate_all, compute_auuc

    metrics = evaluate_all(df, tau_hats={'S': ..., 'T': ..., 'X': ...})
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# NumPy 2.0 renamed trapz → trapezoid
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


# ---------------------------------------------------------------------------
# Uplift curve helpers
# ---------------------------------------------------------------------------

def _uplift_curve_data(
    tau_hat: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort users by descending τ̂, then compute cumulative uplift at each
    fractile of the population.

    Returns
    -------
    fractions : 1-D array of population fractions [0, 1]
    uplifts   : 1-D array of cumulative uplift at each fraction
    """
    order = np.argsort(-tau_hat)
    T_sorted = treatment[order]
    Y_sorted = outcome[order]

    n = len(tau_hat)
    fractions = np.linspace(0, 1, n_bins + 1)
    uplifts = np.zeros(n_bins + 1)

    for i, frac in enumerate(fractions):
        k = max(1, int(round(frac * n)))
        T_top = T_sorted[:k]
        Y_top = Y_sorted[:k]

        n_treated = T_top.sum()
        n_control = k - n_treated

        if n_treated == 0 or n_control == 0:
            uplifts[i] = 0.0
        else:
            y1 = Y_top[T_top == 1].mean()
            y0 = Y_top[T_top == 0].mean()
            uplifts[i] = (y1 - y0) * frac  # scale by fraction for area calculation

    return fractions, uplifts


def compute_auuc(
    tau_hat: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = 100,
) -> float:
    """
    Compute Area Under the Uplift Curve (AUUC).

    A random model scores 0.5 (the diagonal). Models above 0.5 are useful.
    """
    fractions, uplifts = _uplift_curve_data(tau_hat, treatment, outcome, n_bins)
    auuc = float(_trapz(uplifts, fractions))
    return auuc


def compute_random_auuc(
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = 100,
) -> float:
    """AUUC for a random (uninformed) model -- serves as the baseline."""
    tau_random = np.random.default_rng(0).uniform(0, 1, size=len(treatment))
    return compute_auuc(tau_random, treatment, outcome, n_bins)


# ---------------------------------------------------------------------------
# Qini coefficient
# ---------------------------------------------------------------------------

def compute_qini(
    tau_hat: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = 100,
) -> float:
    """
    Qini coefficient = AUUC_model - AUUC_random.
    A positive Qini means the model outperforms random ranking.
    """
    auuc_model = compute_auuc(tau_hat, treatment, outcome, n_bins)
    auuc_random = compute_random_auuc(treatment, outcome, n_bins)
    return float(auuc_model - auuc_random)


# ---------------------------------------------------------------------------
# PEHE (requires ground-truth τ)
# ---------------------------------------------------------------------------

def compute_pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    """
    sqrt(PEHE) = RMSE between estimated and true individual CATE.
    Lower is better.
    """
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_by_decile(
    tau_hat: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Split users into deciles by τ̂ and compare mean predicted CATE
    to empirical lift (mean Y|T=1 - mean Y|T=0) within each decile.

    Returns a DataFrame with columns:
        decile, mean_tau_hat, empirical_lift, n_treated, n_control
    """
    df = pd.DataFrame({
        "tau_hat": tau_hat,
        "treatment": treatment,
        "outcome": outcome,
    })
    df["decile"] = pd.qcut(df["tau_hat"], q=n_deciles, labels=False, duplicates="drop")

    rows = []
    for decile, group in df.groupby("decile"):
        treated = group[group["treatment"] == 1]["outcome"]
        control = group[group["treatment"] == 0]["outcome"]

        empirical_lift = (
            treated.mean() - control.mean()
            if len(treated) > 0 and len(control) > 0
            else np.nan
        )
        rows.append({
            "decile": int(decile),
            "mean_tau_hat": group["tau_hat"].mean(),
            "empirical_lift": empirical_lift,
            "n_treated": len(treated),
            "n_control": len(control),
        })

    return pd.DataFrame(rows).sort_values("decile")


# ---------------------------------------------------------------------------
# Uplift curve data for plotting
# ---------------------------------------------------------------------------

def get_uplift_curve(
    tau_hat: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = 100,
) -> pd.DataFrame:
    """Return a DataFrame with 'fraction' and 'uplift' columns for plotting."""
    fractions, uplifts = _uplift_curve_data(tau_hat, treatment, outcome, n_bins)
    return pd.DataFrame({"fraction": fractions, "uplift": uplifts})


# ---------------------------------------------------------------------------
# Evaluate all models
# ---------------------------------------------------------------------------

def evaluate_all(
    df: pd.DataFrame,
    tau_hats: dict[str, np.ndarray],
    n_bins: int = 100,
) -> pd.DataFrame:
    """
    Compute AUUC, Qini, and PEHE for every model in tau_hats.

    Parameters
    ----------
    df       : DataFrame containing 'treatment', 'outcome', 'tau_true' columns.
    tau_hats : dict mapping model name → per-user CATE array.

    Returns
    -------
    metrics : DataFrame with rows = models, columns = [AUUC, Qini, PEHE].
    """
    T = df["treatment"].values
    Y = df["outcome"].values
    tau_true = df["tau_true"].values if "tau_true" in df.columns else None

    rows = []
    for name, tau_hat in tau_hats.items():
        auuc = compute_auuc(tau_hat, T, Y, n_bins)
        qini = compute_qini(tau_hat, T, Y, n_bins)
        pehe = compute_pehe(tau_hat, tau_true) if tau_true is not None else np.nan
        rows.append({"Model": name, "AUUC": auuc, "Qini": qini, "PEHE": pehe})

    # Naive ATE baseline (everyone gets mean ATE as their τ̂)
    ate_naive = (df.loc[df["treatment"] == 1, "outcome"].mean()
                 - df.loc[df["treatment"] == 0, "outcome"].mean())
    tau_naive = np.full(len(df), ate_naive)
    auuc_naive = compute_auuc(tau_naive, T, Y, n_bins)
    qini_naive = compute_qini(tau_naive, T, Y, n_bins)
    pehe_naive = compute_pehe(tau_naive, tau_true) if tau_true is not None else np.nan
    rows.append({"Model": "Naive ATE", "AUUC": auuc_naive, "Qini": qini_naive, "PEHE": pehe_naive})

    metrics = pd.DataFrame(rows).set_index("Model")
    return metrics
