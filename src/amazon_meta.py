"""Parse SNAP amazon-meta.txt and build user-overlap category-pair matrices."""

from __future__ import annotations

import gzip
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

SNAP_URL = "https://snap.stanford.edu/data/bigdata/amazon/amazon-meta.txt.gz"
# Fixed source/target as in the manuscript dataset table (not re-ranked by density).
PAPER_SPLITS = [
    ("CX1", "Book", "Music"),
    ("CX2", "DVD", "Book"),
    ("CX3", "Video", "Book"),
    ("CX4", "DVD", "Music"),
    ("CX5", "Video", "Music"),
    ("CX6", "DVD", "Video"),
]
CATEGORY_PAIRS = [(s, t) for _, s, t in PAPER_SPLITS]
PAPER_MAP = {f"{s}->{t}": cx for cx, s, t in PAPER_SPLITS}


def ensure_amazon_meta(preferred: Optional[str] = None) -> str:
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(
        [
            r"G:\My Drive\amazon-meta.txt",
            "/content/drive/MyDrive/amazon-meta.txt",
            "/content/drive/MyDrive/amazon-meta.txt.gz",
            "/content/amazon-meta.txt",
            "amazon-meta.txt",
            "amazon-meta.txt.gz",
        ]
    )
    for p in candidates:
        if p and os.path.exists(p) and p.endswith(".txt") and not p.endswith(".gz"):
            return p

    gz_path = None
    for p in candidates:
        if p and os.path.exists(p) and p.endswith(".gz"):
            gz_path = p
            break

    if gz_path is None:
        gz_path = "/content/amazon-meta.txt.gz" if Path("/content").exists() else "amazon-meta.txt.gz"
        print("Downloading SNAP amazon-meta.txt.gz ...")
        print(SNAP_URL)
        urllib.request.urlretrieve(SNAP_URL, gz_path)
        print("Downloaded:", gz_path, "size=", os.path.getsize(gz_path))

    txt_path = "/content/amazon-meta.txt" if Path("/content").exists() else "amazon-meta.txt"
    print("Extracting", gz_path, "->", txt_path)
    with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    drive_txt = "/content/drive/MyDrive/amazon-meta.txt"
    try:
        if Path("/content/drive/MyDrive").exists():
            shutil.copy2(txt_path, drive_txt)
            return drive_txt
    except Exception as exc:  # noqa: BLE001
        print("Could not copy to Drive (ok):", exc)
    return txt_path


def parse_amazon_meta(file_path: str) -> pd.DataFrame:
    ids, asins, titles, groups, salesranks, similars = [], [], [], [], [], []
    customer_ids, ratings, votes, helpfuls = [], [], [], []
    current_id = current_asin = current_title = current_group = None
    current_salesrank = current_similar = None
    in_review_section = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith("Id"):
                in_review_section = False
                current_id = current_asin = current_title = current_group = None
                current_salesrank = current_similar = None

            parts = line.split(":", 1)
            key = parts[0].strip() if parts else ""
            value = parts[1].strip() if len(parts) > 1 else ""

            if key == "Id":
                current_id = value
            elif key == "ASIN":
                current_asin = value
            elif key == "title":
                current_title = value
            elif key == "group":
                current_group = value
            elif key == "salesrank":
                current_salesrank = value
            elif key == "similar":
                current_similar = value
            elif key.startswith("reviews"):
                in_review_section = True
            elif in_review_section and "cutomer:" in line.lower():
                try:
                    tokens = line.strip().split()
                    cust = rating = vote = helpful = None
                    for i, tok in enumerate(tokens):
                        tl = tok.lower()
                        if tl.startswith("cutomer") or tl.startswith("customer"):
                            cust = tokens[i + 1] if i + 1 < len(tokens) else None
                        elif tl.startswith("rating"):
                            rating = tokens[i + 1] if i + 1 < len(tokens) else None
                        elif tl.startswith("votes"):
                            vote = tokens[i + 1] if i + 1 < len(tokens) else None
                        elif tl.startswith("helpful"):
                            helpful = tokens[i + 1] if i + 1 < len(tokens) else None
                    if cust is None or rating is None:
                        continue
                    ids.append(current_id)
                    asins.append(current_asin)
                    titles.append(current_title)
                    groups.append(current_group)
                    salesranks.append(current_salesrank)
                    similars.append(current_similar)
                    customer_ids.append(cust)
                    ratings.append(rating)
                    votes.append(vote)
                    helpfuls.append(helpful)
                except Exception:  # noqa: BLE001
                    continue

    df = pd.DataFrame(
        {
            "Id": ids,
            "ASIN": asins,
            "Title": titles,
            "Group": groups,
            "Salesrank": salesranks,
            "Similar": similars,
            "Customerid": customer_ids,
            "Rating": ratings,
            "Votes": votes,
            "Helpful": helpfuls,
        }
    )
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating", "Customerid", "ASIN", "Group"])
    return df


def build_pair_matrices(
    df: pd.DataFrame,
    cat_a: str,
    cat_b: str,
    rating_threshold: int = 30,
    source_group: Optional[str] = None,
    target_group: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    user_category_counts = df.groupby(["Customerid", "Group"]).size().reset_index(name="num_ratings")
    filtered_users = user_category_counts[user_category_counts["num_ratings"] >= rating_threshold]
    valid_users = filtered_users[filtered_users["Group"].isin([cat_a, cat_b])]
    valid_users = valid_users.groupby("Customerid").filter(lambda x: x["Group"].nunique() > 1)

    filtered_reviews = df[df["Customerid"].isin(valid_users["Customerid"])]
    filtered_reviews = filtered_reviews[filtered_reviews["Group"].isin([cat_a, cat_b])].copy()
    filtered_reviews["Rating"] = pd.to_numeric(filtered_reviews["Rating"], errors="coerce")

    domain_stats = (
        filtered_reviews.groupby("Group")
        .agg(
            num_users=("Customerid", "nunique"),
            num_items=("ASIN", "nunique"),
            num_ratings=("Rating", "count"),
        )
        .reset_index()
    )
    domain_stats["density_level"] = domain_stats["num_ratings"] / (
        domain_stats["num_users"] * domain_stats["num_items"]
    )
    if source_group is None or target_group is None:
        source_group = domain_stats.nlargest(1, "density_level")["Group"].values[0]
        target_group = domain_stats.nsmallest(1, "density_level")["Group"].values[0]

    source_matrix = filtered_reviews[filtered_reviews["Group"] == source_group]
    target_matrix = filtered_reviews[filtered_reviews["Group"] == target_group]

    source_rating_matrix = source_matrix.pivot_table(
        index="Customerid", columns="ASIN", values="Rating", aggfunc="mean"
    ).fillna(0)
    target_rating_matrix = target_matrix.pivot_table(
        index="Customerid", columns="ASIN", values="Rating", aggfunc="mean"
    ).fillna(0)

    common_users = source_rating_matrix.index.intersection(target_rating_matrix.index)
    source_rating_matrix = source_rating_matrix.loc[common_users]
    target_rating_matrix = target_rating_matrix.loc[common_users]

    t_arr = target_rating_matrix.to_numpy()
    meta = {
        "cat_a": cat_a,
        "cat_b": cat_b,
        "source": source_group,
        "target": target_group,
        "n_users": len(common_users),
        "n_items_s": source_rating_matrix.shape[1],
        "n_items_t": target_rating_matrix.shape[1],
        "n_ratings_t": int((t_arr > 0).sum()),
        "density_t": float((t_arr > 0).sum() / t_arr.size),
        "mean_rating_t": float(t_arr[t_arr > 0].mean()) if (t_arr > 0).any() else float("nan"),
    }
    return source_rating_matrix, target_rating_matrix, meta
