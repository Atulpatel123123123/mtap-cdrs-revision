"""MovieLens-1M genre-pair matrices (second dataset family; user-overlap CDR)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Primary-genre pairs with enough overlapping users (density-ordered inside builder).
# MovieLens is denser than Amazon — used as a second-family generalization check.
ML_PAIRS = [
    ("ML0", "Action", "Adventure"),  # compact second-family pair used in paper
    ("ML1", "Comedy", "Drama"),
    ("ML2", "Action", "Comedy"),
    ("ML3", "Action", "Drama"),
]


def find_ml1m(root: Optional[str] = None) -> Path:
    candidates = []
    if root:
        candidates.append(Path(root))
    candidates.extend(
        [
            Path("data/ml-1m"),
            Path("/content/ml-1m"),
            Path("/content/drive/MyDrive/ml-1m"),
            Path(__file__).resolve().parents[1] / "data" / "ml-1m",
        ]
    )
    for c in candidates:
        if (c / "ratings.dat").exists() and (c / "movies.dat").exists():
            return c
    raise FileNotFoundError("MovieLens-1M not found (need ratings.dat + movies.dat)")


def load_ml1m(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = find_ml1m(data_dir)
    ratings = pd.read_csv(
        d / "ratings.dat",
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        d / "movies.dat",
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1",
    )
    return ratings, movies


def _genre_ratings(ratings: pd.DataFrame, movies: pd.DataFrame, genre: str) -> pd.DataFrame:
    # Primary genre = first listed genre (common ML CDR practice for disjoint item sets).
    movies = movies.copy()
    movies["PrimaryGenre"] = movies["Genres"].str.split("|").str[0]
    mids = set(movies.loc[movies["PrimaryGenre"] == genre, "MovieID"])
    return ratings[ratings["MovieID"].isin(mids)].copy()


def build_genre_pair(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    genre_a: str,
    genre_b: str,
    min_ratings: int = 20,
    source_genre: Optional[str] = None,
    target_genre: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    ra = _genre_ratings(ratings, movies, genre_a)
    rb = _genre_ratings(ratings, movies, genre_b)
    # Users with enough ratings in BOTH genres
    ca = ra.groupby("UserID").size()
    cb = rb.groupby("UserID").size()
    common = sorted(set(ca[ca >= min_ratings].index) & set(cb[cb >= min_ratings].index))
    ra = ra[ra["UserID"].isin(common)]
    rb = rb[rb["UserID"].isin(common)]

    def dens(df):
        n_u = df["UserID"].nunique()
        n_i = df["MovieID"].nunique()
        n_r = len(df)
        return n_r / max(n_u * n_i, 1)

    dens_a, dens_b = dens(ra), dens(rb)
    if source_genre is None or target_genre is None:
        if dens_a >= dens_b:
            source_genre, target_genre = genre_a, genre_b
            src_df, tgt_df = ra, rb
        else:
            source_genre, target_genre = genre_b, genre_a
            src_df, tgt_df = rb, ra
    else:
        src_df = ra if source_genre == genre_a else rb
        tgt_df = rb if target_genre == genre_b else ra
        if source_genre == genre_b:
            src_df, tgt_df = rb, ra

    S = src_df.pivot_table(index="UserID", columns="MovieID", values="Rating", aggfunc="mean").fillna(0)
    T = tgt_df.pivot_table(index="UserID", columns="MovieID", values="Rating", aggfunc="mean").fillna(0)
    # Align users
    users = S.index.intersection(T.index)
    S, T = S.loc[users], T.loc[users]
    t = T.to_numpy()
    meta = {
        "family": "MovieLens-1M",
        "source": source_genre,
        "target": target_genre,
        "n_users": int(len(users)),
        "n_items_s": int(S.shape[1]),
        "n_items_t": int(T.shape[1]),
        "n_ratings_s": int((S.to_numpy() > 0).sum()),
        "n_ratings_t": int((t > 0).sum()),
        "density_s": float((S.to_numpy() > 0).sum() / max(S.size, 1)),
        "density_t": float((t > 0).sum() / max(t.size, 1)),
        "mean_rating_t": float(t[t > 0].mean()) if (t > 0).any() else float("nan"),
    }
    return S, T, meta
