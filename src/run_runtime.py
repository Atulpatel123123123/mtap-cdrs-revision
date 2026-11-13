"""Wall-clock runtime on CX1 (Book→Music), one fold, TR=20%, E=20."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.utils import shuffle

from .amazon_meta import PAPER_SPLITS, build_pair_matrices, ensure_amazon_meta, parse_amazon_meta
from .mfb import run_mfb_once
from .models_runtime import run_dmf_once, run_mf_once, run_nmf_once, run_svd_once, time_method


def load_or_build_cx1(meta_path, threshold, cache_path: Path):
    if cache_path.exists():
        print("Loading cache:", cache_path)
        z = np.load(cache_path, allow_pickle=True)
        return z["R_s"], z["R_t"], z["meta"].item()

    print("Using meta:", meta_path)
    print("Parsing amazon-meta...")
    t0 = time.time()
    df = parse_amazon_meta(meta_path)
    print(f"parsed rows={len(df)} in {time.time()-t0:.1f}s")
    df = df[df["Group"].isin(["Book", "Music", "DVD", "Video"])].copy()
    df = df.drop_duplicates(subset=["Customerid", "ASIN", "Group", "Rating"])
    _, source, target = PAPER_SPLITS[0]
    S, T, meta = build_pair_matrices(
        df, source, target, threshold, source_group=source, target_group=target
    )
    R_s, R_t = S.to_numpy(), T.to_numpy()
    meta["cx"] = "CX1"
    np.savez_compressed(cache_path, R_s=R_s, R_t=R_t, meta=meta)
    return R_s, R_t, meta


def one_fold_split(R_t, seed=42, train_frac=0.2, n_folds=5):
    pairs = [(u, i, R_t[u, i]) for u in range(R_t.shape[0]) for i in range(R_t.shape[1]) if R_t[u, i] > 0]
    pairs = shuffle(pairs, random_state=seed)
    fold_size = len(pairs) // n_folds
    te = pairs[:fold_size]
    tr = pairs[fold_size:]
    idx = shuffle(list(range(len(tr))), random_state=seed)
    tr = [tr[j] for j in idx[: max(1, int(len(tr) * train_frac))]]
    R_train = np.zeros_like(R_t)
    R_test = np.zeros_like(R_t)
    for u, i, r in tr:
        R_train[int(u), int(i)] = r
    for u, i, r in te:
        R_test[int(u), int(i)] = r
    return R_train, R_test


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=None)
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-frac", type=float, default=0.2)
    parser.add_argument("--cache", default="cache_cx1.npz")
    parser.add_argument("--out", default="runtime_cx1.json")
    args = parser.parse_args(argv)

    meta_path = ensure_amazon_meta(args.meta)
    R_s, R_t, meta = load_or_build_cx1(meta_path, args.threshold, Path(args.cache))
    print("CX1:", meta)
    R_train, R_test = one_fold_split(R_t, seed=args.seed, train_frac=args.train_frac)
    print(
        f"split ratings: train={(R_train>0).sum()} test={(R_test>0).sum()} "
        f"source={(R_s>0).sum()}"
    )

    common = dict(
        n_factors=10,
        epochs=args.epochs,
        seed=args.seed,
    )
    jobs = [
        ("MF_CDRS", run_mf_once, dict(lambda_reg=0.01, gamma=0.1, learning_rate=0.005, **common)),
        ("MFB_CDRS", run_mfb_once, dict(lambda_reg=0.01, gamma=0.1, learning_rate=0.005, **common)),
        ("NMF_CDRS", run_nmf_once, dict(alpha=0.01, gamma=0.1, **common)),
        ("SVD_CDRS", run_svd_once, dict(lambda_reg=0.01, gamma=0.1, learning_rate=0.005, **common)),
        ("DMF_CDRS", run_dmf_once, dict(lambda_reg=0.01, learning_rate=0.005, **common)),
    ]

    rows = []
    for name, fn, kw in jobs:
        print(f"\nTiming {name} ...")
        row = time_method(name, fn, R_s, R_train, R_test, **kw)
        rows.append(row)
        print(
            f"  {name}: {row['seconds']:.1f}s  "
            f"(~{row['sec_per_epoch_pair']:.2f}s / epoch-budget)  "
            f"MAE={row['mae']:.4f} RMSE={row['rmse']:.4f}"
        )

    Path(args.out).write_text(json.dumps({"meta": meta, "rows": rows, "epochs": args.epochs}, indent=2))
    print("Wrote", args.out)

    print("=" * 72)
    print("COPY EVERYTHING BELOW THIS LINE AND PASTE INTO CURSOR CHAT")
    print("=" * 72)
    print("TASK: MTAP_Revision_runtime_CX1")
    print(f"SEED={args.seed} EPOCHS={args.epochs} TR={args.train_frac}")
    print(f"PAIR=Book->Music users={meta.get('n_users')} items_s={meta.get('n_items_s')} items_t={meta.get('n_items_t')}")
    print()
    print("RUNTIME_TABLE")
    print("Method\tSeconds\tSecPerEpochBudget\tMAE\tRMSE")
    for r in rows:
        print(
            f"{r['method']}\t{r['seconds']:.1f}\t{r['sec_per_epoch_pair']:.2f}\t"
            f"{r['mae']:.4f}\t{r['rmse']:.4f}"
        )
    print("=" * 72)
    print("END_OF_OUTPUT")
    print("=" * 72)


if __name__ == "__main__":
    main()
