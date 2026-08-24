"""
value_model.py
--------------
Trains a supervised regression model (LightGBM) to predict a user's 
potential lifetime value, CONDITIONAL on them being retained (outcome == 1).

We predict E[Value | Retained]. We do NOT train on churned users (who have
0 value) because that would devolve the model into predicting churn again.
We want to know: "If we save this user, how valuable will they actually be?"

Usage:
    from src.value_model import train_value_model
    model, refutation_results = train_value_model(df, feature_cols)
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_value_model(df: pd.DataFrame, feature_cols: list[str], seed: int = 42):
    """
    Train LGBMRegressor on retained users to predict `value_score`.
    Runs basic refutations to ensure the model isn't just fitting noise.
    """
    print("\n[value_model] Training Value Prediction Model (E[Value | Retained]) ...")
    
    # Train only on retained users
    df_retained = df[df["outcome"] == 1].copy()
    
    X = df_retained[feature_cols].values
    y = df_retained["value_score"].values
    
    # Train Regressor
    model = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"   In-sample R2: {r2:.4f} | RMSE: {rmse:.2f}")
    
    # ---- Regression Refutation Suite ----
    print("[value_model] Running refutations ...")
    refutation_results = {}
    rng = np.random.default_rng(seed)
    n = len(X)
    
    # 1. Placebo Target (Shuffle y) -> Expect R2 ~ 0
    y_shuffled = rng.permutation(y)
    model_placebo = LGBMRegressor(n_estimators=100, random_state=seed, verbose=-1)
    model_placebo.fit(X, y_shuffled)
    r2_placebo = r2_score(y_shuffled, model_placebo.predict(X))
    
    passed_placebo = abs(r2_placebo) < 0.1
    refutation_results["placebo"] = {
        "passed": passed_placebo,
        "summary": f"Placebo R2 = {r2_placebo:.4f} (expected ~0) | passed = {passed_placebo}"
    }
    print(f"   {refutation_results['placebo']['summary']}")
    
    # 2. Random Common Cause (Add Noise Feature) -> Expect R2 to remain stable
    noise = rng.normal(0, 1, size=(n, 1))
    X_noisy = np.concatenate([X, noise], axis=1)
    model_rcc = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1)
    model_rcc.fit(X_noisy, y)
    r2_rcc = r2_score(y, model_rcc.predict(X_noisy))
    
    # Passed if the noise doesn't artificially inflate or degrade the score
    shift = abs(r2 - r2_rcc)
    passed_rcc = shift < 0.05
    refutation_results["rcc"] = {
        "passed": passed_rcc,
        "summary": f"RCC R2 = {r2_rcc:.4f} (shift = {shift:.4f}) | passed = {passed_rcc}"
    }
    print(f"   {refutation_results['rcc']['summary']}")
    
    return model, refutation_results
