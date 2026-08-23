"""
data_loader.py
--------------
Downloads and parses the MovieLens 1M dataset into a single merged DataFrame.

Usage:
    from src.data_loader import load_movielens_1m
    ratings, users, movies = load_movielens_1m()
"""

import io
import os
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ml-1m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_ml1m(dest: Path) -> None:
    """Download and extract ml-1m if not already present."""
    ratings_path = dest / "ratings.dat"
    if ratings_path.exists():
        print(f"[data_loader] ml-1m already present at {dest}. Skipping download.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[data_loader] Downloading MovieLens 1M from {ML1M_URL} ...")
    with urllib.request.urlopen(ML1M_URL) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.namelist():
            filename = Path(member).name
            if filename in {"ratings.dat", "users.dat", "movies.dat"}:
                with zf.open(member) as src, open(dest / filename, "wb") as dst:
                    dst.write(src.read())

    print(f"[data_loader] Extracted to {dest}")


def load_movielens_1m(
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens 1M into three DataFrames.

    Returns
    -------
    ratings : DataFrame with columns [user_id, movie_id, rating, timestamp]
    users   : DataFrame with columns [user_id, gender, age, occupation, zip_code]
    movies  : DataFrame with columns [movie_id, title, genres]
    """
    _download_ml1m(data_dir)

    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    users = pd.read_csv(
        data_dir / "users.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )

    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    print(
        f"[data_loader] Loaded: {len(ratings):,} ratings | "
        f"{len(users):,} users | {len(movies):,} movies"
    )
    return ratings, users, movies


def load_merged(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Convenience function that returns ratings merged with user metadata.
    Timestamp is converted to datetime.
    """
    ratings, users, movies = load_movielens_1m(data_dir)

    # Convert Unix timestamp → datetime
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")

    merged = ratings.merge(users, on="user_id", how="left")
    # Keep genre string on movies for feature engineering, but don't explode here
    merged = merged.merge(movies[["movie_id", "genres"]], on="movie_id", how="left")

    return merged
