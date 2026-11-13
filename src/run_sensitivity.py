"""MFB hyperparameter sensitivity on CX1 (Book→Music): γ and d at TR=20%."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.utils import shuffle

from .amazon_meta import PAPER_SPLITS, build_pair_matrices, ensure_amazon_meta, parse_amazon_meta
from .mfb import run_mfb_once


def make_folds(R: np.ndarray, n_folds: int = 5, seed: int = 42):
    pairs = [(u, i, R[u, i]) for u in range(R.shape[0]) for i in range(R.shape[1]) if R[u, i] > 0]
    pairs = shuffle(pairs, random_state=seed)
    fold_size = len(pairs) // n_folds
    folds = []
    for k in range(n_folds):
        te_idx = set(range(k * fold_size, (k + 1) * fold_size if k < n_folds - 1 else len(pairs)))
        te = [pairs[j] for j in sorted(te_idx)]
        tr = [pairs[j] for j in range(len(pairs)) if j not in te_idx]
        folds.append((tr, te))
    return folds


def apply_triples(shape, triples):
    R = np.zeros(shape, dtype=np.float64)
    for u, i, r in triples:
        R[int(u), int(i)] = r
    return R


def subsample_train(tr, frac: float, seed: int):
    idx = shuffle(list(range(len(tr))), random_state=seed)
    cut = max(1, int(len(tr) * frac))
    return [tr[j] for j in idx[:cut]]


def summarize(vals):
    vals = np.asarray(vals, dtype=float)
    mean = float(vals.mean())
    ci = float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, ci


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=None)
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--folds", type=int, default=3, help="Folds for sensitivity (default 3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-frac", type=float, default=0.2)
    parser.add_argument("--cache", default="cache_cx1.npz")
    args = parser.parse_args(argv)

    cache_path = Path(args.cache)
    if cache_path.exists():
        print("Loading cached CX1 matrices:", cache_path)
        z = np.load(cache_path, allow_pickle=True)
        R_s = z["R_s"]
        R_t = z["R_t"]
        meta = {k: z["meta"].item()[k] for k in z["meta"].item()}
    else:
        meta_path = ensure_amazon_meta(args.meta)
        print("Using meta:", meta_path)
        print("Parsing amazon-meta (may take several minutes)...")
        t0 = time.time()
        df = parse_amazon_meta(meta_path)
        print(f"parsed rows={len(df)} in {time.time()-t0:.1f}s")
        df = df[df["Group"].isin(["Book", "Music", "DVD", "Video"])].copy()
        df = df.drop_duplicates(subset=["Customerid", "ASIN", "Group", "Rating"])

        cx, source, target = PAPER_SPLITS[0]  # CX1 Book→Music
        S, T, meta = build_pair_matrices(
            df, source, target, args.threshold, source_group=source, target_group=target
        )
        R_s = S.to_numpy()
        R_t = T.to_numpy()
        meta["cx"] = cx
        np.savez_compressed(cache_path, R_s=R_s, R_t=R_t, meta=meta)
        print("Cached:", cache_path)

    print("CX1 meta:", meta)
    folds = make_folds(R_t, n_folds=args.folds, seed=args.seed)

    gammas = [0.0, 0.01, 0.1, 1.0]
    dims = [5, 10, 20]
    base_d = 10
    base_g = 0.1

    gamma_rows = []
    print("\n==== gamma sweep (d=10, TR=20%) ====")
    for g in gammas:
        maes = []
        t0 = time.time()
        for k, (tr, te) in enumerate(folds):
            R_test = apply_triples(R_t.shape, te)
            R_train = apply_triples(R_t.shape, subsample_train(tr, args.train_frac, args.seed + k))
            mae, rmse = run_mfb_once(
                R_s,
                R_train,
                R_test,
                n_factors=base_d,
                gamma=g,
                epochs=args.epochs,
                seed=args.seed + 100 * k,
            )
            maes.append(mae)
            print(f"  gamma={g} fold={k+1} MAE={mae:.4f} RMSE={rmse:.4f}")
        m, ci = summarize(maes)
        gamma_rows.append((g, m, ci, maes, time.time() - t0))
        print(f"gamma={g}: MAE={m:.4f} ± {ci:.4f}  ({time.time()-t0:.1f}s)")

    dim_rows = []
    print("\n==== d sweep (gamma=0.1, TR=20%) ====")
    for d in dims:
        maes = []
        t0 = time.time()
        for k, (tr, te) in enumerate(folds):
            R_test = apply_triples(R_t.shape, te)
            R_train = apply_triples(R_t.shape, subsample_train(tr, args.train_frac, args.seed + k))
            mae, rmse = run_mfb_once(
                R_s,
                R_train,
                R_test,
                n_factors=d,
                gamma=base_g,
                epochs=args.epochs,
                seed=args.seed + 100 * k,
            )
            maes.append(mae)
            print(f"  d={d} fold={k+1} MAE={mae:.4f} RMSE={rmse:.4f}")
        m, ci = summarize(maes)
        dim_rows.append((d, m, ci, maes, time.time() - t0))
        print(f"d={d}: MAE={m:.4f} ± {ci:.4f}  ({time.time()-t0:.1f}s)")

    print("=" * 72)
    print("COPY EVERYTHING BELOW THIS LINE AND PASTE INTO CURSOR CHAT")
    print("=" * 72)
    print("TASK: MTAP_Revision_sensitivity_MFB_CX1")
    print(f"SEED={args.seed} FOLDS={args.folds} EPOCHS={args.epochs} TR={args.train_frac}")
    print(f"PAIR=Book->Music users={meta.get('n_users')} items_t={meta.get('n_items_t')}")
    print()
    print("GAMMA_SWEEP_d10")
    print("gamma\tMAE\tMAE_CI\tfolds")
    for g, m, ci, maes, _ in gamma_rows:
        print(f"{g}\t{m:.4f}\t{ci:.4f}\t" + ",".join(f"{x:.4f}" for x in maes))
    print()
    print("D_SWEEP_gamma0.1")
    print("d\tMAE\tMAE_CI\tfolds")
    for d, m, ci, maes, _ in dim_rows:
        print(f"{d}\t{m:.4f}\t{ci:.4f}\t" + ",".join(f"{x:.4f}" for x in maes))
    print("=" * 72)
    print("END_OF_OUTPUT")
    print("=" * 72)


if __name__ == "__main__":
    main()
