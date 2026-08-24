"""
viz.py
------
All visualization functions for the uplift modeling project.

Plots produced:
    1. plot_uplift_curves()     -- all models' uplift curves overlaid
    2. plot_qini_curves()       -- Qini curves
    3. plot_cate_distribution() -- violin plot of CATE by user segment
    4. plot_shap_beeswarm()     -- SHAP beeswarm for X-learner
    5. plot_calibration()       -- predicted vs empirical lift by decile
    6. plot_refutation_table()  -- refutation results as a styled table
    7. plot_metrics_bar()       -- AUUC/Qini bar chart across models

All functions save to results/figures/ unless ax is provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = sns.color_palette("tab10")
MODEL_COLORS = {
    "S": PALETTE[0],
    "T": PALETTE[1],
    "X": PALETTE[2],
    "R": PALETTE[3],
    "UpliftTree": PALETTE[4],
    "DoubleML": PALETTE[5],
    "Naive ATE": PALETTE[7],
}

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
})


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    print(f"[viz] Saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Uplift curves
# ---------------------------------------------------------------------------

def plot_uplift_curves(
    uplift_dfs: dict[str, pd.DataFrame],
    save: bool = True,
) -> plt.Figure:
    """
    Parameters
    ----------
    uplift_dfs : dict mapping model name → DataFrame with columns
                 ['fraction', 'uplift'].
                 Use evaluation.get_uplift_curve() to generate these.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, df in uplift_dfs.items():
        color = MODEL_COLORS.get(name, None)
        ax.plot(df["fraction"], df["uplift"], label=name, color=color, linewidth=2)

    # Random baseline (diagonal)
    ax.plot([0, 1], [0, uplift_dfs[next(iter(uplift_dfs))]["uplift"].max()],
            "--", color="gray", linewidth=1.5, label="Random baseline")

    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Cumulative uplift")
    ax.set_title("Uplift Curves -- All Models")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    if save:
        _save(fig, "uplift_curves.png")
    return fig


# ---------------------------------------------------------------------------
# 2. Qini curves
# ---------------------------------------------------------------------------

def plot_qini_curves(
    uplift_dfs: dict[str, pd.DataFrame],
    random_uplift: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> plt.Figure:
    """Qini = uplift curve - random curve area."""
    fig, ax = plt.subplots(figsize=(9, 6))

    if random_uplift is not None:
        rand_arr = random_uplift["uplift"].values
    else:
        rand_arr = None

    for name, df in uplift_dfs.items():
        uplift_arr = df["uplift"].values
        fractions = df["fraction"].values
        if rand_arr is not None and len(rand_arr) == len(uplift_arr):
            qini_arr = uplift_arr - rand_arr
        else:
            qini_arr = uplift_arr
        color = MODEL_COLORS.get(name, None)
        ax.plot(fractions, qini_arr, label=name, color=color, linewidth=2)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, label="Random baseline")
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Qini (uplift above random)")
    ax.set_title("Qini Curves -- All Models")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    if save:
        _save(fig, "qini_curves.png")
    return fig


# ---------------------------------------------------------------------------
# 3. CATE distribution by segment
# ---------------------------------------------------------------------------

def plot_cate_distribution(
    df: pd.DataFrame,
    tau_hats: dict[str, np.ndarray],
    segment_col: str = "recency_bin",
    save: bool = True,
) -> plt.Figure:
    """
    Violin plot of CATE estimates by user recency segment.
    Adds recency_bin automatically if not present.
    """
    plot_df = df.copy()

    # Create recency bins if missing
    if segment_col not in plot_df.columns and "recency" in plot_df.columns:
        plot_df["recency_bin"] = pd.qcut(
            plot_df["recency"], q=4,
            duplicates="drop",
        )
        # Convert intervals to string to make seaborn happy
        plot_df["recency_bin"] = plot_df["recency_bin"].astype(str)
        segment_col = "recency_bin"

    n_models = len(tau_hats)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, tau_hat) in zip(axes, tau_hats.items()):
        tmp = plot_df[[segment_col]].copy()
        tmp["tau_hat"] = tau_hat

        sns.violinplot(
            data=tmp,
            x=segment_col,
            y="tau_hat",
            ax=ax,
            palette="muted",
            inner="quartile",
        )
        ax.set_title(f"{name}-learner")
        ax.set_xlabel("Recency quartile")
        ax.tick_params(axis="x", rotation=30)
        ax.axhline(0, color="red", linewidth=1, linestyle="--")

    axes[0].set_ylabel("Estimated CATE (τ̂)")
    fig.suptitle("CATE Distribution by User Recency Segment", y=1.02, fontsize=14)
    plt.tight_layout()

    if save:
        _save(fig, "cate_distribution.png")
    return fig


# ---------------------------------------------------------------------------
# 4. SHAP beeswarm
# ---------------------------------------------------------------------------

def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    model_name: str = "X-learner",
    save: bool = True,
) -> plt.Figure:
    """SHAP beeswarm plot (top 10 features)."""
    try:
        import shap
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            X,
            plot_type="dot",
            max_display=10,
            show=False,
        )
        ax = plt.gca()
        ax.set_title(f"SHAP Feature Importance -- {model_name}")
        if save:
            _save(plt.gcf(), f"shap_beeswarm_{model_name.replace(' ', '_')}.png")
        return plt.gcf()
    except Exception as e:
        print(f"[viz] SHAP beeswarm skipped: {e}")
        return plt.figure()


# ---------------------------------------------------------------------------
# 5. Calibration plot
# ---------------------------------------------------------------------------

def plot_calibration(
    calibration_dfs: dict[str, pd.DataFrame],
    save: bool = True,
) -> plt.Figure:
    """
    Parameters
    ----------
    calibration_dfs : dict mapping model name → output of
                      evaluation.calibration_by_decile().
    """
    n_models = len(calibration_dfs)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, cal_df) in zip(axes, calibration_dfs.items()):
        ax.scatter(
            cal_df["decile"] + 1,
            cal_df["empirical_lift"],
            s=80, label="Empirical lift", color="steelblue", zorder=3,
        )
        ax.plot(
            cal_df["decile"] + 1,
            cal_df["mean_tau_hat"],
            "-o", label="Mean τ̂", color="tomato", linewidth=2,
        )
        ax.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax.set_xlabel("Decile (1=lowest τ̂)")
        ax.set_title(f"{name} Calibration")
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Lift / CATE")
    fig.suptitle("Calibration: Predicted CATE vs. Empirical Lift by Decile", fontsize=13)
    plt.tight_layout()

    if save:
        _save(fig, "calibration_plots.png")
    return fig


# ---------------------------------------------------------------------------
# 6. Refutation results table
# ---------------------------------------------------------------------------

def plot_refutation_table(
    refutation_results: dict,
    save: bool = True,
) -> plt.Figure:
    """Render refutation results as a styled matplotlib table."""
    rows = []
    for test_name, res in refutation_results.items():
        if "error" in res:
            rows.append({
                "Test": test_name,
                "New Effect": "--",
                "p-value": "--",
                "Passed": "ERROR",
            })
        else:
            rows.append({
                "Test": test_name,
                "New Effect": f"{res.get('estimated_effect', np.nan):.4f}",
                "p-value": f"{res.get('p_value', np.nan):.3f}",
                "Passed": "✓" if res.get("passed", False) else "✗",
            })

    df_table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(rows)))
    ax.axis("off")

    tbl = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)

    # Colour header
    for j in range(len(df_table.columns)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Colour pass/fail
    for i, row in enumerate(rows):
        cell = tbl[i + 1, df_table.columns.get_loc("Passed")]
        if row["Passed"] == "✓":
            cell.set_facecolor("#d5f5e3")
        elif row["Passed"] == "✗":
            cell.set_facecolor("#fadbd8")

    ax.set_title("Refutation Test Results", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()

    if save:
        _save(fig, "refutation_table.png")
    return fig


# ---------------------------------------------------------------------------
# 7. Metrics bar chart
# ---------------------------------------------------------------------------

def plot_metrics_bar(
    metrics_df: pd.DataFrame,
    metric: str = "AUUC",
    save: bool = True,
) -> plt.Figure:
    """Bar chart of a single metric across all models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = metrics_df.index.tolist()
    values = metrics_df[metric].values
    colors = [MODEL_COLORS.get(m, PALETTE[8]) for m in models]

    bars = ax.bar(models, values, color=colors, edgecolor="white", linewidth=1.5)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Comparison -- All Estimators")
    ax.tick_params(axis="x", rotation=30)

    if save:
        _save(fig, f"metrics_bar_{metric.lower()}.png")
    return fig

# ---------------------------------------------------------------------------
# Value vs Uplift Scatter Plot
# ---------------------------------------------------------------------------

def plot_value_uplift_quadrant(df_segments: pd.DataFrame, uplift_threshold: float, value_threshold: float, save: bool = True):
    """
    Plots a 2x2 scatter grid of CATE (Uplift) vs Predicted Value.
    Highlights the "Mismatch" quadrant.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for segments
    color_map = {
        "Star Users (High Uplift, High Value)": "#2ecc71",      # Green
        "Mismatch (High Uplift, Low Value)": "#e74c3c",         # Red
        "Sure Things (Low Uplift, High Value)": "#3498db",      # Blue
        "Lost Causes (Low Uplift, Low Value)": "#95a5a6"        # Gray
    }
    
    for segment, color in color_map.items():
        subset = df_segments[df_segments["segment"] == segment]
        ax.scatter(subset["uplift"], subset["predicted_value"], c=color, label=segment, alpha=0.6, edgecolors='none')

    # Draw quadrant lines
    ax.axvline(x=uplift_threshold, color='white', linestyle='--', alpha=0.5)
    ax.axhline(y=value_threshold, color='white', linestyle='--', alpha=0.5)
    
    # Annotate Mismatch Quadrant
    ax.text(
        df_segments["uplift"].max(), 
        value_threshold / 2, 
        "⚠️ High Uplift, Low Value\n(Don't Treat)", 
        color="#e74c3c", fontsize=12, fontweight='bold', ha='right', va='center'
    )
    
    ax.set_title("Value-Aware Uplift Analysis", fontsize=16, fontweight='bold', color='white')
    ax.set_xlabel("Predicted Causal Uplift (CATE)", fontsize=12, color='white')
    ax.set_ylabel("Predicted Potential Lifetime Value", fontsize=12, color='white')
    ax.legend(loc='upper left', frameon=False, labelcolor='white')
    
    plt.tight_layout()
    if save:
        _save(fig, "value_quadrant.png")
    return fig
