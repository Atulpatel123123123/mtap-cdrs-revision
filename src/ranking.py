"""NDCG / Recall helpers for explicit-rating test sets."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = np.asarray(relevances[:k], dtype=float)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float(np.sum((2.0**rel - 1.0) * discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Graded NDCG@k; y_true = true ratings, y_score = predictions for same items."""
    if len(y_true) == 0:
        return np.nan
    order = np.argsort(-y_score)
    ranked = y_true[order]
    ideal = np.sort(y_true)[::-1]
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0:
        return 0.0
    return dcg_at_k(ranked, k) / idcg


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int, threshold: float = 4.0) -> float:
    """Binary Recall@k: fraction of relevant items (rating>=threshold) retrieved in top-k."""
    relevant = np.where(y_true >= threshold)[0]
    if relevant.size == 0:
        return np.nan
    top = np.argsort(-y_score)[:k]
    hit = np.intersect1d(top, relevant).size
    return float(hit / relevant.size)


def user_ranking_metrics(
    R_test: np.ndarray,
    R_pred: np.ndarray,
    ks: Tuple[int, ...] = (5, 10),
    min_test_items: int = 2,
    recall_threshold: float = 4.0,
) -> Dict[str, float]:
    """Average NDCG@k / Recall@k over users with enough test ratings (rank within test items)."""
    n_users = R_test.shape[0]
    buckets: Dict[str, List[float]] = {f"ndcg@{k}": [] for k in ks}
    buckets.update({f"recall@{k}": [] for k in ks})

    for u in range(n_users):
        items = np.where(R_test[u] > 0)[0]
        if items.size < min_test_items:
            continue
        y_true = R_test[u, items]
        y_score = R_pred[u, items]
        for k in ks:
            kk = min(k, items.size)
            buckets[f"ndcg@{k}"].append(ndcg_at_k(y_true, y_score, kk))
            buckets[f"recall@{k}"].append(recall_at_k(y_true, y_score, kk, recall_threshold))

    out = {"n_users_eval": float(len(buckets[f"ndcg@{ks[0]}"]))}
    for key, vals in buckets.items():
        arr = np.asarray([v for v in vals if not np.isnan(v)], dtype=float)
        out[key] = float(arr.mean()) if arr.size else float("nan")
    return out
