"""NDCG@K / Recall@K on CX1 for mu, MFB, DMF (rating-ranking among test items)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.utils import shuffle

from .amazon_meta import ensure_amazon_meta
from .mfb import (
    compute_global_mean,
    observed_triples,
    predict,
    train_mf_bias,
    transfer_training_bias,
)
from .ranking import user_ranking_metrics
from .run_runtime import load_or_build_cx1, one_fold_split


def pred_mu(R_train: np.ndarray, shape) -> np.ndarray:
    mu = compute_global_mean(R_train)
    return np.full(shape, mu, dtype=np.float64)


def pred_mfb(R_source, R_train, n_factors=10, lambda_reg=0.01, gamma=0.1,
             learning_rate=0.005, epochs=20, seed=42) -> np.ndarray:
    b_s, wu_s, wi_s, P_s, Q_s = train_mf_bias(
        R_source, n_factors=n_factors, lambda_reg=lambda_reg,
        learning_rate=learning_rate, epochs=epochs, seed=seed,
    )
    n_users, n_items_t = R_train.shape
    rng = np.random.default_rng(seed + 1)
    b_t = compute_global_mean(R_train) if np.any(R_train > 0) else b_s
    wu_t = np.zeros(n_users)
    wi_t = np.zeros(n_items_t)
    P_t = rng.normal(scale=0.1, size=(n_users, n_factors))
    Q_t = rng.normal(scale=0.1, size=(n_items_t, n_factors))
    P_t[:] = P_s
    wu_t[:] = wu_s
    transfer_training_bias(
        R_train, b_s, wu_s, wi_s, P_s, Q_s, b_t, wu_t, wi_t, P_t, Q_t,
        lambda_reg=lambda_reg, gamma=gamma, learning_rate=learning_rate,
        epochs=epochs, common_users=range(n_users), common_items=None,
        do_bias_transfer=True, seed=seed + 2,
    )
    return predict(b_t, wu_t, wi_t, P_t, Q_t)


def pred_dmf(R_source, R_train, n_factors=10, lambda_reg=0.01,
             learning_rate=0.005, epochs=20, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_users = R_source.shape[0]

    def train(R, n_items, U0=None, w0=None, b0=0.0):
        U = U0.copy() if U0 is not None else 0.01 * rng.normal(size=(n_users, n_factors))
        V = 0.01 * rng.normal(size=(n_items, n_factors))
        w = w0.copy() if w0 is not None else 0.01 * rng.normal(size=(n_factors,))
        b = float(b0)
        triples = observed_triples(R)
        for _ in range(epochs):
            for j in rng.permutation(len(triples)):
                u, i, r = triples[j]
                u, i = int(u), int(i)
                z = U[u] * V[i]
                pred = float(np.dot(z, w) + b)
                err = pred - r
                w -= learning_rate * (err * z + lambda_reg * w)
                b -= learning_rate * err
                U[u] -= learning_rate * (err * (w * V[i]) + lambda_reg * U[u])
                V[i] -= learning_rate * (err * (w * U[u]) + lambda_reg * V[i])
        return U, V, w, b

    U_s, _, w_s, b_s = train(R_source, R_source.shape[1])
    U_t, V_t, w_t, b_t = train(R_train, R_train.shape[1], U0=U_s, w0=w_s, b0=b_s)
    # Full prediction matrix via broadcasting
    # pred[u,i] = sum_k U[u,k]*V[i,k]*w[k] + b
    return (U_t * w_t) @ V_t.T + b_t


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=None)
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-frac", type=float, default=0.2)
    parser.add_argument("--cache", default="cache_cx1.npz")
    parser.add_argument("--out", default="ndcg_cx1.json")
    args = parser.parse_args(argv)

    meta_path = ensure_amazon_meta(args.meta)
    R_s, R_t, meta = load_or_build_cx1(meta_path, args.threshold, Path(args.cache))
    print("CX1:", meta)
    R_train, R_test = one_fold_split(R_t, seed=args.seed, train_frac=args.train_frac)
    print(f"train={(R_train>0).sum()} test={(R_test>0).sum()}")

    jobs = [
        ("mu-baseline", lambda: pred_mu(R_train, R_t.shape)),
        ("MFB_CDRS", lambda: pred_mfb(R_s, R_train, epochs=args.epochs, seed=args.seed)),
        ("DMF_CDRS", lambda: pred_dmf(R_s, R_train, epochs=args.epochs, seed=args.seed)),
    ]

    rows = []
    for name, fn in jobs:
        print(f"\nFitting {name} ...")
        t0 = time.perf_counter()
        pred = fn()
        elapsed = time.perf_counter() - t0
        metrics = user_ranking_metrics(R_test, pred, ks=(5, 10))
        metrics["method"] = name
        metrics["seconds"] = elapsed
        rows.append(metrics)
        print(
            f"  {name}: NDCG@5={metrics['ndcg@5']:.4f} NDCG@10={metrics['ndcg@10']:.4f} "
            f"Recall@5={metrics['recall@5']:.4f} Recall@10={metrics['recall@10']:.4f} "
            f"users={int(metrics['n_users_eval'])} ({elapsed:.1f}s)"
        )

    Path(args.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")
    print("Wrote", args.out)

    print("=" * 72)
    print("COPY EVERYTHING BELOW THIS LINE AND PASTE INTO CURSOR CHAT")
    print("=" * 72)
    print("TASK: MTAP_Revision_ndcg_CX1")
    print(f"SEED={args.seed} EPOCHS={args.epochs} TR={args.train_frac}")
    print(f"PAIR=Book->Music users={meta.get('n_users')}")
    print()
    print("RANKING_TABLE")
    print("Method\tNDCG@5\tNDCG@10\tRecall@5\tRecall@10\tn_users")
    for r in rows:
        print(
            f"{r['method']}\t{r['ndcg@5']:.4f}\t{r['ndcg@10']:.4f}\t"
            f"{r['recall@5']:.4f}\t{r['recall@10']:.4f}\t{int(r['n_users_eval'])}"
        )
    print("=" * 72)
    print("END_OF_OUTPUT")
    print("=" * 72)


if __name__ == "__main__":
    main()
