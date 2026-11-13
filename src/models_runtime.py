"""Timed one-shot trainers for MF / MFB / NMF / SVD / DMF (revision runtime table)."""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .mfb import (
    compute_global_mean,
    evaluate,
    observed_triples,
    predict,
    run_mfb_once,
    train_mf_bias,
    transfer_training_bias,
)


def run_mf_once(R_source, R_train, R_test, n_factors=10, lambda_reg=0.01, gamma=0.1,
                learning_rate=0.005, epochs=20, seed=42) -> Tuple[float, float]:
    """MF without biases (same sparse SGD + user-factor transfer)."""
    rng = np.random.default_rng(seed)
    R_source = np.asarray(R_source)
    R_train = np.asarray(R_train)
    R_test = np.asarray(R_test)

    n_users, n_items_s = R_source.shape
    P_s = rng.normal(scale=0.1, size=(n_users, n_factors))
    Q_s = rng.normal(scale=0.1, size=(n_items_s, n_factors))
    triples_s = observed_triples(R_source)
    for _ in range(epochs):
        for j in rng.permutation(len(triples_s)):
            u, i, r = triples_s[j]
            u, i = int(u), int(i)
            err = r - np.dot(P_s[u], Q_s[i])
            pu = P_s[u].copy()
            P_s[u] += learning_rate * (err * Q_s[i] - lambda_reg * P_s[u])
            Q_s[i] += learning_rate * (err * pu - lambda_reg * Q_s[i])

    n_items_t = R_train.shape[1]
    P_t = rng.normal(scale=0.1, size=(n_users, n_factors))
    Q_t = rng.normal(scale=0.1, size=(n_items_t, n_factors))
    P_t[:] = P_s
    triples_t = observed_triples(R_train)
    for _ in range(epochs):
        for j in rng.permutation(len(triples_t)):
            u, i, r = triples_t[j]
            u, i = int(u), int(i)
            err = r - np.dot(P_t[u], Q_t[i])
            pu = P_t[u].copy()
            P_t[u] += learning_rate * (err * Q_t[i] - lambda_reg * P_t[u])
            Q_t[i] += learning_rate * (err * pu - lambda_reg * Q_t[i])
            P_t[u] -= learning_rate * gamma * (P_t[u] - P_s[u])
    pred = P_t @ Q_t.T
    return evaluate(R_test, pred)


def run_nmf_once(R_source, R_train, R_test, n_factors=10, alpha=0.01, gamma=0.1,
                 epochs=20, seed=42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    R_source = np.asarray(R_source, dtype=np.float64)
    R_train = np.asarray(R_train, dtype=np.float64)
    R_test = np.asarray(R_test, dtype=np.float64)
    n_users, n_items_s = R_source.shape
    W_s = rng.random((n_users, n_factors)) + 1e-4
    H_s = rng.random((n_factors, n_items_s)) + 1e-4
    for _ in range(epochs):
        num_H = W_s.T @ R_source
        den_H = (W_s.T @ W_s) @ H_s + alpha * H_s
        H_s *= num_H / np.maximum(den_H, 1e-10)
        num_W = R_source @ H_s.T
        den_W = W_s @ (H_s @ H_s.T) + alpha * W_s
        W_s *= num_W / np.maximum(den_W, 1e-10)

    n_items_t = R_train.shape[1]
    W_t = rng.random((n_users, n_factors)) + 1e-4
    H_t = rng.random((n_factors, n_items_t)) + 1e-4
    W_t[:] = W_s
    user_mask = np.ones(n_users)
    for _ in range(epochs):
        num_H = W_t.T @ R_train
        den_H = (W_t.T @ W_t) @ H_t + alpha * H_t
        H_t *= num_H / np.maximum(den_H, 1e-10)
        num_W = R_train @ H_t.T + gamma * (W_s * user_mask[:, None])
        den_W = W_t @ (H_t @ H_t.T) + alpha * W_t + gamma * (W_t * user_mask[:, None])
        W_t *= num_W / np.maximum(den_W, 1e-10)
    pred = W_t @ H_t
    return evaluate(R_test, pred)


def run_svd_once(R_source, R_train, R_test, n_factors=10, lambda_reg=0.01, gamma=0.1,
                 learning_rate=0.005, epochs=20, seed=42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    R_source = np.asarray(R_source)
    R_train = np.asarray(R_train)
    R_test = np.asarray(R_test)
    n_users = R_source.shape[0]
    # Truncated SVD on sparse source (warm-start), then target SGD fine-tune.
    S = csr_matrix(R_source)
    k = min(n_factors, min(S.shape) - 1)
    svd = TruncatedSVD(n_components=k, random_state=seed, n_iter=7)
    U = svd.fit_transform(S)  # (n_users, k)
    V = svd.components_.T  # (n_items_s, k)
    # Scale SVD factors into a stable SGD range (raw singular vectors can explode).
    scale = max(np.linalg.norm(U) / max(np.sqrt(U.size), 1.0), 1e-8)
    U = U / scale
    V = V / scale
    if k < n_factors:
        pad_u = np.zeros((n_users, n_factors - k))
        pad_v = np.zeros((V.shape[0], n_factors - k))
        U = np.hstack([U, pad_u])
        V = np.hstack([V, pad_v])

    n_items_t = R_train.shape[1]
    U_t = gamma * U
    V_t = rng.normal(scale=0.1, size=(n_items_t, n_factors))
    triples = observed_triples(R_train)
    for _ in range(epochs):
        for j in rng.permutation(len(triples)):
            u, i, r = triples[j]
            u, i = int(u), int(i)
            err = r - np.dot(U_t[u], V_t[i])
            uu = U_t[u].copy()
            U_t[u] += learning_rate * (err * V_t[i] - lambda_reg * U_t[u])
            V_t[i] += learning_rate * (err * uu - lambda_reg * V_t[i])
            np.clip(U_t[u], -10.0, 10.0, out=U_t[u])
            np.clip(V_t[i], -10.0, 10.0, out=V_t[i])
    pred = U_t @ V_t.T
    return evaluate(R_test, pred)


def run_dmf_once(R_source, R_train, R_test, n_factors=10, lambda_reg=0.01,
                 learning_rate=0.005, epochs=20, seed=42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    R_source = np.asarray(R_source)
    R_train = np.asarray(R_train)
    R_test = np.asarray(R_test)
    n_users, n_items_s = R_source.shape

    def train_dmf(R, n_items, user0=None, final_w0=None, final_b0=0.0):
        U = (user0.copy() if user0 is not None else 0.01 * rng.normal(size=(n_users, n_factors)))
        V = 0.01 * rng.normal(size=(n_items, n_factors))
        w = final_w0.copy() if final_w0 is not None else 0.01 * rng.normal(size=(n_factors,))
        b = float(final_b0)
        triples = observed_triples(R)
        for _ in range(epochs):
            for j in rng.permutation(len(triples)):
                u, i, r = triples[j]
                u, i = int(u), int(i)
                z = U[u] * V[i]
                pred = float(np.dot(z, w) + b)
                err = pred - r
                grad_w = err * z + lambda_reg * w
                grad_u = err * (w * V[i]) + lambda_reg * U[u]
                grad_v = err * (w * U[u]) + lambda_reg * V[i]
                w -= learning_rate * grad_w
                b -= learning_rate * err
                U[u] -= learning_rate * grad_u
                V[i] -= learning_rate * grad_v
        return U, V, w, b

    U_s, V_s, w_s, b_s = train_dmf(R_source, R_source.shape[1])
    U_t, V_t, w_t, b_t = train_dmf(R_train, R_train.shape[1], user0=U_s, final_w0=w_s, final_b0=b_s)
    # Predict test
    mask = R_test > 0
    us, is_ = np.where(mask)
    preds = np.array([float(np.dot(U_t[u] * V_t[i], w_t) + b_t) for u, i in zip(us, is_)])
    truths = R_test[mask]
    mae = float(np.mean(np.abs(preds - truths)))
    rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))
    return mae, rmse


def time_method(name, fn, *args, **kwargs) -> Dict:
    t0 = time.perf_counter()
    mae, rmse = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    epochs = int(kwargs.get("epochs", 20))
    return {
        "method": name,
        "seconds": elapsed,
        "sec_per_epoch_pair": elapsed / max(epochs, 1),  # source+target share budget label
        "mae": mae,
        "rmse": rmse,
    }
