"""
run_pipeline.py
---------------
End-to-end pipeline runner -- executes all 5 phases in sequence and
saves all intermediate artifacts and figures to results/.

Run from project root:
    python run_pipeline.py [--mode observational|rct] [--n-bootstrap 50]

Phases:
    1. Data loading + feature engineering
    2. Treatment & outcome simulation
    3. Meta-learners (S/T/X/R + Uplift Tree)
    4. DoubleML + refutations
    5. Evaluation & visualization
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows (avoid cp1252 errors from any dependency prints)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script execution

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
DATA    = ROOT / "data"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(ROOT))


# ============================================================================
# Phase 1 -- Data & Features
# ============================================================================

def phase1_data(args):
    print("\n" + "="*60)
    print("PHASE 1 -- Data loading & feature engineering")
    print("="*60)
    t0 = time.time()

    from src.data_loader import load_merged
    from src.feature_engineering import build_user_features

    merged = load_merged()
    X = build_user_features(merged, split_days=30)

    print(f"  Feature matrix: {X.shape[0]:,} users x {X.shape[1]} features")
    print(f"  Elapsed: {time.time()-t0:.1f}s")
    return X


# ============================================================================
# Phase 2 -- Simulation
# ============================================================================

def phase2_simulate(X, args):
    print("\n" + "="*60)
    print(f"PHASE 2 -- Simulation (mode={args.mode})")
    print("="*60)
    t0 = time.time()

    from src.simulate_rct import simulate_treatment_outcome

    df = simulate_treatment_outcome(X, mode=args.mode, seed=42)

    # Also run RCT sanity check
    df_rct = simulate_treatment_outcome(X, mode="rct", seed=42)
    naive_ate_rct = (df_rct.loc[df_rct["treatment"]==1,"outcome"].mean()
                     - df_rct.loc[df_rct["treatment"]==0,"outcome"].mean())
    true_ate = df_rct["tau_true"].mean()
    print(f"\n  [Sanity] RCT naive ATE = {naive_ate_rct:.4f} | True ATE = {true_ate:.4f} "
          f"| Diff = {abs(naive_ate_rct - true_ate):.4f}")

    df.to_parquet(DATA / "simulation_observational.parquet")
    df_rct.to_parquet(DATA / "simulation_rct.parquet")

    # CATE distribution plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df["tau_true"], bins=50, color="steelblue", edgecolor="white")
    ax.axvline(true_ate, color="red", linestyle="--", label=f"True ATE = {true_ate:.4f}")
    ax.set_title("Ground-truth CATE distribution")
    ax.set_xlabel("tau(X)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "sim_cate_dist.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Elapsed: {time.time()-t0:.1f}s")
    return df


# ============================================================================
# Phase 3 -- Meta-learners
# ============================================================================

def phase3_meta_learners(df, args):
    print("\n" + "="*60)
    print("PHASE 3 -- Meta-learners (S/T/X/R + Uplift Tree)")
    print("="*60)
    t0 = time.time()

    from src.meta_learners import run_all_meta_learners
    from src.evaluation import evaluate_all, get_uplift_curve
    from src.viz import plot_uplift_curves, plot_cate_distribution

    feature_cols = [c for c in df.columns
                    if c not in ["treatment", "outcome", "tau_true", "propensity", "value_score"]]

    results = run_all_meta_learners(
        df,
        feature_cols=feature_cols,
        n_bootstrap=args.n_bootstrap,
        run_uplift_tree=True,
    )

    tau_hats = {name: res["tau_hat"] for name, res in results.items()}

    # Metrics
    metrics = evaluate_all(df, tau_hats)
    print("\n  === Meta-learner metrics ===")
    print(metrics.to_string())
    metrics.to_csv(RESULTS / "meta_learner_metrics.csv")

    # Uplift curves
    uplift_dfs = {
        name: get_uplift_curve(tau, df["treatment"].values, df["outcome"].values)
        for name, tau in tau_hats.items()
    }
    fig = plot_uplift_curves(uplift_dfs, save=True)
    import matplotlib.pyplot as plt
    plt.close(fig)

    # CATE violins (S/T/X/R only)
    subset = {k: v for k, v in tau_hats.items() if k in ["S","T","X","R"]}
    fig = plot_cate_distribution(df, tau_hats=subset, save=True)
    plt.close(fig)

    # SHAP for X-learner
    if results["X"]["shap_values"] is not None:
        from src.viz import plot_shap_beeswarm
        fig = plot_shap_beeswarm(
            results["X"]["shap_values"], df[feature_cols],
            model_name="X-learner", save=True
        )
        plt.close(fig)

    # Save model results
    with open(RESULTS / "meta_learner_results.pkl", "wb") as f:
        pickle.dump(results, f)
        
    # Save specific models for API serving
    import joblib
    MODELS_DIR = Path(__file__).resolve().parent / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(results["S"]["learner"], MODELS_DIR / "s_learner.joblib")

    print(f"  Elapsed: {time.time()-t0:.1f}s")
    return results, metrics, uplift_dfs


# ============================================================================
# Phase 4 -- DoubleML + refutations
# ============================================================================

def phase4_doubleml(df, args):
    print("\n" + "="*60)
    print("PHASE 4 -- DoubleML (CausalForestDML) + Refutations")
    print("="*60)
    t0 = time.time()

    from src.doubleml_estimator import run_doubleml, run_refutations
    from src.viz import plot_refutation_table
    import matplotlib.pyplot as plt

    feature_cols = [c for c in df.columns
                    if c not in ["treatment", "outcome", "tau_true", "propensity", "value_score"]]

    dml_result = run_doubleml(df, feature_cols, n_estimators=200, seed=42)
    tau_dml = dml_result["tau_hat"]

    print(f"\n  ATE = {dml_result['ate']:.4f} | "
          f"95% CI = [{dml_result['ate_ci'][0]:.4f}, {dml_result['ate_ci'][1]:.4f}]")

    # Refutations (standalone -- no dowhy dependency)
    refutation_results = run_refutations(
        df, feature_cols, dml_result,
        n_simulations=args.n_refute_sims,
        seed=42,
    )

    # Refutation table plot
    fig = plot_refutation_table(refutation_results, save=True)
    plt.close(fig)

    # Save tau_hat and model
    np.save(RESULTS / "dml_tau_hat.npy", tau_dml)
    
    import joblib
    MODELS_DIR = Path(__file__).resolve().parent / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(dml_result["dml_model"], MODELS_DIR / "doubleml_model.joblib")

    print(f"  Elapsed: {time.time()-t0:.1f}s")
    return tau_dml, refutation_results


# ============================================================================
# Phase 4b -- Value Modeling & Segment Mismatch Analysis
# ============================================================================

def phase4b_value_model(df, tau_dml, args):
    print("\n" + "="*60)
    print("PHASE 4b -- Value Modeling & Mismatch Analysis")
    print("="*60)
    t0 = time.time()
    
    from src.value_model import train_value_model
    from src.segment_analysis import run_segment_analysis
    from src.viz import plot_value_uplift_quadrant
    import matplotlib.pyplot as plt
    import joblib

    feature_cols = [c for c in df.columns
                    if c not in ["treatment", "outcome", "tau_true", "propensity", "value_score"]]

    value_model, refutation_results = train_value_model(df, feature_cols, seed=42)
    
    # Save the model
    MODELS_DIR = Path(__file__).resolve().parent / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(value_model, MODELS_DIR / "value_model.joblib")
    
    # Run Segment Analysis
    if tau_dml is not None:
        seg_results = run_segment_analysis(df, feature_cols, tau_dml, value_model)
        
        # Plot
        df_segments = pd.DataFrame({
            "uplift": tau_dml,
            "predicted_value": seg_results["predicted_value"],
            "segment": seg_results["segments"]
        })
        fig = plot_value_uplift_quadrant(
            df_segments, 
            uplift_threshold=seg_results["uplift_threshold"], 
            value_threshold=seg_results["value_threshold"],
            save=False
        )
        fig.savefig(FIGURES / "value_quadrant.png", bbox_inches="tight")
        
        # Also save to frontend public dir for the dashboard
        FRONTEND_PUBLIC = Path(__file__).resolve().parent / "frontend" / "public"
        FRONTEND_PUBLIC.mkdir(exist_ok=True)
        fig.savefig(FRONTEND_PUBLIC / "value_quadrant.png", bbox_inches="tight", transparent=True)
        
        plt.close(fig)
    
    print(f"  Elapsed: {time.time()-t0:.1f}s")
    return value_model


# ============================================================================
# Phase 5 -- Full evaluation & visualization
# ============================================================================

def phase5_eval_viz(df, meta_results, tau_dml, meta_uplift_dfs, args):
    print("\n" + "="*60)
    print("PHASE 5 -- Full evaluation & visualization")
    print("="*60)
    t0 = time.time()

    from src.evaluation import evaluate_all, get_uplift_curve, calibration_by_decile
    from src.viz import (
        plot_uplift_curves, plot_qini_curves, plot_calibration,
        plot_metrics_bar
    )
    import matplotlib.pyplot as plt

    T = df["treatment"].values
    Y = df["outcome"].values

    tau_hats = {name: res["tau_hat"] for name, res in meta_results.items()}
    if tau_dml is not None:
        tau_hats["DoubleML"] = tau_dml

    metrics = evaluate_all(df, tau_hats)
    print("\n  === All model metrics ===")
    print(metrics.to_string())
    metrics.to_csv(RESULTS / "all_model_metrics.csv")

    # Uplift curves (all models)
    uplift_dfs = {
        name: get_uplift_curve(tau, T, Y) for name, tau in tau_hats.items()
    }
    fig = plot_uplift_curves(uplift_dfs, save=True); plt.close(fig)

    # Qini curves
    tau_rand = np.random.default_rng(0).uniform(0, 1, size=len(T))
    random_uplift = get_uplift_curve(tau_rand, T, Y)
    fig = plot_qini_curves(uplift_dfs, random_uplift=random_uplift, save=True); plt.close(fig)

    # Calibration
    cal_models = ["S", "T", "X", "R", "DoubleML"]
    cal_dfs = {
        name: calibration_by_decile(tau, T, Y, n_deciles=10)
        for name, tau in tau_hats.items() if name in cal_models
    }
    fig = plot_calibration(cal_dfs, save=True); plt.close(fig)

    # Bar charts
    fig = plot_metrics_bar(metrics, metric="AUUC", save=True); plt.close(fig)
    fig = plot_metrics_bar(metrics, metric="Qini", save=True); plt.close(fig)
    pehe_df = metrics.dropna(subset=["PEHE"])
    if not pehe_df.empty:
        fig = plot_metrics_bar(pehe_df, metric="PEHE", save=True); plt.close(fig)

    print(f"  Elapsed: {time.time()-t0:.1f}s")
    return metrics


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Run full uplift modeling pipeline")
    p.add_argument("--mode", default="observational", choices=["observational","rct"],
                   help="Simulation mode (default: observational)")
    p.add_argument("--n-bootstrap", type=int, default=50,
                   help="Bootstrap resamples for meta-learner CIs (default: 50)")
    p.add_argument("--n-refute-sims", type=int, default=30,
                   help="Simulations for dowhy refutations (default: 30)")
    p.add_argument("--skip-dml", action="store_true",
                   help="Skip Phase 4 (DoubleML) to save time")
    p.add_argument("--resume-dml", action="store_true",
                   help="Skip Phases 1-3, load cached data, and run Phase 4 and 5 only")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    total_t0 = time.time()

    if args.resume_dml:
        print("\n" + "="*60)
        print("RESUMING FROM PHASE 4 (Loading cached data)")
        print("="*60)
        df = pd.read_parquet(DATA / "simulation_observational.parquet")
        with open(RESULTS / "meta_learner_results.pkl", "rb") as f:
            meta_results = pickle.load(f)
        
        # We need meta_uplift_dfs for Phase 5 which isn't cached cleanly, so recompute them
        from src.evaluation import get_uplift_curve
        meta_uplift_dfs = {
            name: get_uplift_curve(res["tau_hat"], df["treatment"].values, df["outcome"].values)
            for name, res in meta_results.items()
        }
    else:
        X          = phase1_data(args)
        df         = phase2_simulate(X, args)
        meta_results, meta_metrics, meta_uplift_dfs = phase3_meta_learners(df, args)

    tau_dml = None
    if not args.skip_dml:
        tau_dml, refutation_results = phase4_doubleml(df, args)
        
    # Phase 4b: Value Modeling
    value_model = phase4b_value_model(df, tau_dml, args)

    final_metrics = phase5_eval_viz(df, meta_results, tau_dml, meta_uplift_dfs, args)

    print("\n" + "="*60)
    print(f"PIPELINE COMPLETE  -- total time: {time.time()-total_t0:.1f}s")
    print(f"Results saved to:  {RESULTS}")
    print(f"Figures saved to:  {FIGURES}")
    print("="*60)
    print("\nFinal metrics table:")
    print(final_metrics.to_string())
