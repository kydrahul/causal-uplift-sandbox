"""
feature_engineering.py
-----------------------
Transforms raw MovieLens interaction data into a user-level feature matrix X
suitable for causal inference (confounders + potential effect moderators).

Usage:
    from src.data_loader import load_merged
    from src.feature_engineering import build_user_features

    merged = load_merged()
    X = build_user_features(merged, split_days=30)
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_user_features(
    merged: pd.DataFrame,
    split_days: int = 30,
) -> pd.DataFrame:
    """
    Build a per-user feature DataFrame.

    Parameters
    ----------
    merged      : Output of data_loader.load_merged().
    split_days  : Number of days after the first rating used as the
                  "pre-treatment" observation window.

    Returns
    -------
    X : DataFrame indexed by user_id with the following columns:

        Behavioral (from rating history in first `split_days`):
            activity_level  -- rating count in first split_days days
            genre_entropy   -- Shannon entropy over genre proportions
            avg_rating      -- mean rating score given
            recency         -- days between last rating and split point
                              (lower = more recent = more active)

        Demographic (from users.dat):
            age             -- binned age (numeric code from ML categories)
            gender          -- 0/1 (F/M)
            occupation      -- integer code (0-20)

        Top-genre one-hot:
            genre_<name>    -- 1 if most-rated genre is <name>, else 0
    """
    merged = merged.copy()
    merged["datetime"] = pd.to_datetime(merged["datetime"])

    # --- Determine split timestamp per user ---
    first_rating = merged.groupby("user_id")["datetime"].min().rename("first_rating")
    merged = merged.join(first_rating, on="user_id")
    merged["days_since_first"] = (merged["datetime"] - merged["first_rating"]).dt.days

    pre = merged[merged["days_since_first"] <= split_days].copy()
    split_time = (
        merged.groupby("user_id")["first_rating"].first()
        + pd.Timedelta(days=split_days)
    )

    # -------------------------------------------------------------------
    # Behavioral features (computed from pre-window data)
    # -------------------------------------------------------------------
    activity = pre.groupby("user_id").size().rename("activity_level")
    avg_rating = pre.groupby("user_id")["rating"].mean().rename("avg_rating")

    # Genre entropy
    def _genre_entropy(group: pd.DataFrame) -> float:
        genres_flat = []
        for g_str in group["genres"]:
            genres_flat.extend(g_str.split("|"))
        if not genres_flat:
            return 0.0
        counts = pd.Series(genres_flat).value_counts(normalize=True)
        return float(scipy_entropy(counts))

    genre_entropy = pre.groupby("user_id").apply(_genre_entropy).rename("genre_entropy")

    # Recency: days from last pre-window rating to split point
    last_rating_pre = pre.groupby("user_id")["datetime"].max().rename("last_rating_pre")
    recency = (split_time - last_rating_pre).dt.days.rename("recency")
    recency = recency.clip(lower=0)  # can't be negative

    # Top preferred genre (per user, using all pre-window data)
    def _top_genre(group: pd.DataFrame) -> str:
        genres_flat = []
        for g_str in group["genres"]:
            genres_flat.extend(g_str.split("|"))
        if not genres_flat:
            return "Unknown"
        return pd.Series(genres_flat).value_counts().index[0]

    top_genre = pre.groupby("user_id").apply(_top_genre).rename("top_genre")

    # -------------------------------------------------------------------
    # Demographic features (static, from users.dat columns)
    # -------------------------------------------------------------------
    user_meta = merged[["user_id", "gender", "age", "occupation"]].drop_duplicates(
        subset="user_id"
    ).set_index("user_id")

    # Encode gender: F→0, M→1
    user_meta["gender"] = (user_meta["gender"] == "M").astype(int)

    # age is already a numeric category in ML-1M (1,18,25,35,45,50,56)
    # Map to ordinal rank for simplicity
    age_order = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
    user_meta["age"] = user_meta["age"].map(age_order).fillna(0).astype(int)

    # -------------------------------------------------------------------
    # Assemble base features
    # -------------------------------------------------------------------
    X = pd.concat(
        [activity, avg_rating, genre_entropy, recency, top_genre, user_meta],
        axis=1,
    )

    # Fill users with no pre-window activity
    X["activity_level"] = X["activity_level"].fillna(0).astype(float)
    X["avg_rating"] = X["avg_rating"].fillna(X["avg_rating"].median())
    X["genre_entropy"] = X["genre_entropy"].fillna(0.0)
    X["recency"] = X["recency"].fillna(split_days).astype(float)
    X["top_genre"] = X["top_genre"].fillna("Unknown")

    # One-hot encode top genre
    top_genre_dummies = pd.get_dummies(X["top_genre"], prefix="genre")
    X = pd.concat([X.drop(columns=["top_genre"]), top_genre_dummies], axis=1)

    # Cast occupation to int
    X["occupation"] = X["occupation"].fillna(0).astype(int)

    print(f"[feature_engineering] Feature matrix: {X.shape[0]:,} users × {X.shape[1]} features")
    return X


def get_feature_names(X: pd.DataFrame) -> list[str]:
    """Return ordered list of feature column names."""
    return list(X.columns)
