"""Run LightGCN_CDRS on all Amazon CX pairs + all MovieLens genre pairs.

Protocol matches paper: 5-fold, seed 42, TR 20/40/60/80%, d=10, E=20.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.utils import shuffle

from .amazon_meta import PAPER_SPLITS, build_pair_matrices, ensure_amazon_meta, parse_amazon_meta
from .gnn_cdrs import run_gnn_once
from .movielens import ML_PAIRS, build_genre_pair, load_ml1m


def cv_gnn(R_s, R_t, n_folds=5, seed=42, train_frac=0.2, epochs=20, **kw):
    pairs = [(u, i, R_t[u, i]) for u in range(R_t.shape[0]) for i in range(R_t.shape[1]) if R_t[u, i] > 0]
    pairs = shuffle(pairs, random_state=seed)
    fold_size = len(pairs) // n_folds
    maes, rmses = [], []
    for k in range(n_folds):
        te_idx = set(range(k * fold_size, (k + 1) * fold_size if k < n_folds - 1 else len(pairs)))
        te = [pairs[j] for j in sorted(te_idx)]
        tr = [pairs[j] for j in range(len(pairs)) if j not in te_idx]
        idx = shuffle(list(range(len(tr))), random_state=seed + k)
        tr = [tr[j] for j in idx[: max(1, int(len(tr) * train_frac))]]
        R_train = np.zeros_like(R_t)
        R_test = np.zeros_like(R_t)
        for u, i, r in tr:
            R_train[int(u), int(i)] = r
        for u, i, r in te:
            R_test[int(u), int(i)] = r
        t0 = time.time()
        mae, rmse = run_gnn_once(
            R_s, R_train, R_test, epochs=epochs, seed=seed + 10 * k, **kw
        )
        dt = time.time() - t0
        maes.append(mae)
        rmses.append(rmse)
        print(f"      fold {k+1}: MAE={mae:.4f} RMSE={rmse:.4f} ({dt:.1f}s)")
    m = float(np.mean(maes))
    r = float(np.mean(rmses))
    mci = float(1.96 * np.std(maes, ddof=1) / np.sqrt(len(maes))) if len(maes) > 1 else 0.0
    rci = float(1.96 * np.std(rmses, ddof=1) / np.sqrt(len(rmses))) if len(rmses) > 1 else 0.0
    return {
        "mae": m,
        "mae_ci": mci,
        "rmse": r,
        "rmse_ci": rci,
        "mae_folds": maes,
        "rmse_folds": rmses,
    }


def cache_dir(root: Path) -> Path:
    d = root / "cache_pairs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_or_build_amazon(root: Path, meta_path: str, rating_threshold: int = 30):
    """Parse once; cache each CX pair as .npz."""
    cdir = cache_dir(root)
    needed = []
    for cid, s, t in PAPER_SPLITS:
        p = cdir / f"{cid}_{s}_{t}.npz"
        if not p.exists():
            needed.append((cid, s, t))
    if needed:
        print("Parsing amazon-meta (slow, once)...", meta_path)
        df = parse_amazon_meta(meta_path)
        print("rows:", len(df), "groups:", sorted(df["Group"].dropna().unique().tolist())[:20])
        for cid, s, t in PAPER_SPLITS:
            p = cdir / f"{cid}_{s}_{t}.npz"
            if p.exists():
                continue
            print(f"  building {cid} {s}->{t} ...")
            # PAPER_SPLITS encodes fixed source/target (not density re-rank).
            S, T, meta = build_pair_matrices(
                df, s, t, rating_threshold=rating_threshold, source_group=s, target_group=t
            )
            meta["id"] = cid
            np.savez_compressed(
                p,
                R_s=S.to_numpy().astype(np.float32),
                R_t=T.to_numpy().astype(np.float32),
                meta=json.dumps(meta),
            )
            print("   ", meta)

    packs = []
    for cid, s, t in PAPER_SPLITS:
        p = cdir / f"{cid}_{s}_{t}.npz"
        z = np.load(p, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        packs.append((cid, f"{s}->{t}", z["R_s"].astype(np.float64), z["R_t"].astype(np.float64), meta))
    return packs


def load_or_build_movielens(root: Path, data_dir=None, min_ratings: int = 20):
    cdir = cache_dir(root)
    ratings, movies = None, None
    packs = []
    for cid, ga, gb in ML_PAIRS:
        p = cdir / f"{cid}_{ga}_{gb}.npz"
        if not p.exists():
            if ratings is None:
                print("Loading MovieLens-1M...")
                ratings, movies = load_ml1m(data_dir)
            print(f"  building {cid} {ga}/{gb} ...")
            S, T, meta = build_genre_pair(ratings, movies, ga, gb, min_ratings=min_ratings)
            meta["id"] = cid
            np.savez_compressed(
                p,
                R_s=S.to_numpy().astype(np.float32),
                R_t=T.to_numpy().astype(np.float32),
                meta=json.dumps(meta),
            )
            print("   ", meta)
        z = np.load(p, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        key = f"{meta['source']}->{meta['target']}"
        packs.append((cid, key, z["R_s"].astype(np.float64), z["R_t"].astype(np.float64), meta))
    return packs


def run_family(name, packs, fractions, folds, epochs, out_path: Path, gnn_kw):
    results = {
        "family": name,
        "folds": folds,
        "epochs": epochs,
        "fractions": list(fractions),
        "gnn": gnn_kw,
        "pairs": {},
    }
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
            print("Resuming from", out_path)
        except Exception:  # noqa: BLE001
            pass

    for cid, key, R_s, R_t, meta in packs:
        pack = results["pairs"].get(key, {"meta": meta, "methods": {"GNN": {}}})
        pack["meta"] = meta
        gnn = pack.setdefault("methods", {}).setdefault("GNN", {})
        print(f"\n==== {name} {cid} {key} users={R_s.shape[0]} items_t={R_t.shape[1]} ====")
        for frac in fractions:
            tr_key = str(int(frac * 100))
            if tr_key in gnn and "mae" in gnn[tr_key]:
                print(f"  TR={tr_key}% already done — skip")
                continue
            print(f"  GNN TR={tr_key}% ...")
            t0 = time.time()
            r = cv_gnn(
                R_s,
                R_t,
                n_folds=folds,
                seed=42,
                train_frac=frac,
                epochs=epochs,
                **gnn_kw,
            )
            r["seconds"] = time.time() - t0
            gnn[tr_key] = r
            pack["methods"]["GNN"] = gnn
            results["pairs"][key] = pack
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"    TR{tr_key} MAE={r['mae']:.4f}±{r['mae_ci']:.4f} RMSE={r['rmse']:.4f}±{r['rmse_ci']:.4f}")
        results["pairs"][key] = pack
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon-meta", default=None)
    parser.add_argument("--ml-data", default=None)
    parser.add_argument("--family", choices=["amazon", "movielens", "both"], default="both")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--factors", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gnn_kw = {
        "n_factors": args.factors,
        "n_layers": args.layers,
        "lambda_reg": 0.01,
        "gamma": 0.1,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
    }
    fractions = (0.2, 0.4, 0.6, 0.8)

    if args.family in ("amazon", "both"):
        preferred = args.amazon_meta or r"G:\My Drive\amazon-meta.txt"
        meta_path = ensure_amazon_meta(preferred)
        packs = load_or_build_amazon(root, meta_path)
        run_family(
            "Amazon",
            packs,
            fractions,
            args.folds,
            args.epochs,
            out_dir / "gnn_amazon_all_pairs.json",
            gnn_kw,
        )

    if args.family in ("movielens", "both"):
        packs = load_or_build_movielens(root, args.ml_data)
        run_family(
            "MovieLens-1M",
            packs,
            fractions,
            args.folds,
            args.epochs,
            out_dir / "gnn_movielens_all_pairs.json",
            gnn_kw,
        )

    print("DONE")


if __name__ == "__main__":
    main()
