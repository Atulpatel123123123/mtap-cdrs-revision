"""Sparse MFB_CDRS (matches notebook math; iterates observed ratings only)."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def _to_numpy(R) -> np.ndarray:
    return R.to_numpy() if hasattr(R, "to_numpy") else np.asarray(R)


def observed_triples(R: np.ndarray) -> np.ndarray:
    u, i = np.where(R > 0)
    return np.column_stack([u, i, R[u, i]]).astype(np.float64)


def compute_global_mean(R: np.ndarray) -> float:
    mask = R > 0
    return float(R[mask].mean()) if np.any(mask) else 0.0


def evaluate(R_true: np.ndarray, R_pred: np.ndarray) -> Tuple[float, float]:
    mask = R_true > 0
    err = R_pred[mask] - R_true[mask]
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def predict(b, w_user, w_item, P, Q) -> np.ndarray:
    return b + w_user[:, None] + w_item[None, :] + P @ Q.T


def train_mf_bias(
    R_source: np.ndarray,
    n_factors: int = 10,
    lambda_reg: float = 0.01,
    learning_rate: float = 0.005,
    epochs: int = 20,
    seed: int = 42,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_users, n_items = R_source.shape
    b = compute_global_mean(R_source)
    w_user = np.zeros(n_users, dtype=np.float64)
    w_item = np.zeros(n_items, dtype=np.float64)
    P = rng.normal(scale=0.1, size=(n_users, n_factors))
    Q = rng.normal(scale=0.1, size=(n_items, n_factors))
    triples = observed_triples(R_source)

    for _ in range(epochs):
        order = rng.permutation(len(triples))
        for j in order:
            u, i, r_ui = triples[j]
            u = int(u)
            i = int(i)
            pred = b + w_user[u] + w_item[i] + np.dot(P[u], Q[i])
            err = r_ui - pred
            w_user[u] += learning_rate * (err - lambda_reg * w_user[u])
            w_item[i] += learning_rate * (err - lambda_reg * w_item[i])
            pu = P[u].copy()
            P[u] += learning_rate * (err * Q[i] - lambda_reg * P[u])
            Q[i] += learning_rate * (err * pu - lambda_reg * Q[i])
    return b, w_user, w_item, P, Q


def transfer_training_bias(
    R_target: np.ndarray,
    b_source,
    w_user_source,
    w_item_source,
    P_source,
    Q_source,
    b_target,
    w_user_target,
    w_item_target,
    P_target,
    Q_target,
    lambda_reg: float = 0.01,
    gamma: float = 0.1,
    learning_rate: float = 0.005,
    epochs: int = 20,
    common_users: Optional[Iterable[int]] = None,
    common_items: Optional[Iterable[int]] = None,
    do_bias_transfer: bool = True,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    common_users = set(common_users) if common_users is not None else set(range(R_target.shape[0]))
    common_items = set(common_items) if common_items is not None else set()
    triples = observed_triples(R_target)

    for _ in range(epochs):
        order = rng.permutation(len(triples))
        for j in order:
            u, i, r_ui = triples[j]
            u = int(u)
            i = int(i)
            pred = b_target + w_user_target[u] + w_item_target[i] + np.dot(P_target[u], Q_target[i])
            err = r_ui - pred
            w_user_target[u] += learning_rate * (err - lambda_reg * w_user_target[u])
            w_item_target[i] += learning_rate * (err - lambda_reg * w_item_target[i])
            pu = P_target[u].copy()
            P_target[u] += learning_rate * (err * Q_target[i] - lambda_reg * P_target[u])
            Q_target[i] += learning_rate * (err * pu - lambda_reg * Q_target[i])

            if u in common_users and u < P_source.shape[0]:
                P_target[u] -= learning_rate * gamma * (P_target[u] - P_source[u])
                if do_bias_transfer:
                    w_user_target[u] -= learning_rate * gamma * (w_user_target[u] - w_user_source[u])
            if i in common_items and i < Q_source.shape[0]:
                Q_target[i] -= learning_rate * gamma * (Q_target[i] - Q_source[i])
                if do_bias_transfer:
                    w_item_target[i] -= learning_rate * gamma * (w_item_target[i] - w_item_source[i])

    return b_target, w_user_target, w_item_target, P_target, Q_target


def run_mfb_once(
    R_source: np.ndarray,
    R_train: np.ndarray,
    R_test: np.ndarray,
    n_factors: int = 10,
    lambda_reg: float = 0.01,
    gamma: float = 0.1,
    learning_rate: float = 0.005,
    epochs: int = 20,
    seed: int = 42,
) -> Tuple[float, float]:
    """Train source MFB, transfer to target train ratings, evaluate on test."""
    R_source = _to_numpy(R_source)
    R_train = _to_numpy(R_train)
    R_test = _to_numpy(R_test)

    b_s, wu_s, wi_s, P_s, Q_s = train_mf_bias(
        R_source,
        n_factors=n_factors,
        lambda_reg=lambda_reg,
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
    )

    n_users, n_items_t = R_train.shape
    rng = np.random.default_rng(seed + 1)
    b_t = compute_global_mean(R_train) if np.any(R_train > 0) else b_s
    wu_t = np.zeros(n_users, dtype=np.float64)
    wi_t = np.zeros(n_items_t, dtype=np.float64)
    P_t = rng.normal(scale=0.1, size=(n_users, n_factors))
    Q_t = rng.normal(scale=0.1, size=(n_items_t, n_factors))

    # Shared-user indices (aligned rows); items are domain-specific → no item overlap.
    common_users = range(n_users)
    n_share = min(P_s.shape[0], n_users)
    P_t[:n_share] = P_s[:n_share]
    wu_t[:n_share] = wu_s[:n_share]

    transfer_training_bias(
        R_train,
        b_s,
        wu_s,
        wi_s,
        P_s,
        Q_s,
        b_t,
        wu_t,
        wi_t,
        P_t,
        Q_t,
        lambda_reg=lambda_reg,
        gamma=gamma,
        learning_rate=learning_rate,
        epochs=epochs,
        common_users=common_users,
        common_items=None,
        do_bias_transfer=True,
        seed=seed + 2,
    )
    pred = predict(b_t, wu_t, wi_t, P_t, Q_t)
    return evaluate(R_test, pred)
