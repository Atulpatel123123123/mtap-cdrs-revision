"""Second dataset family: MovieLens-1M genre pairs — μ / MF / MFB / DMF comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.utils import shuffle

from .mfb import run_mfb_once
from .models_runtime import run_dmf_once, run_mf_once
from .movielens import ML_PAIRS, build_genre_pair, load_ml1m
from .run_mu_baseline import cross_validate_mu_baseline
from .run_runtime import one_fold_split


def cv_method(fn, R_s, R_t, n_folds=3, seed=42, train_frac=0.2, epochs=20, **kw):
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
        mae, rmse = fn(R_s, R_train, R_test, epochs=epochs, seed=seed + 10 * k, **kw)
        maes.append(mae)
        rmses.append(rmse)
        print(f"    fold {k+1}: MAE={mae:.4f} RMSE={rmse:.4f}")
    m = float(np.mean(maes))
    r = float(np.mean(rmses))
    mci = float(1.96 * np.std(maes, ddof=1) / np.sqrt(len(maes))) if len(maes) > 1 else 0.0
    rci = float(1.96 * np.std(rmses, ddof=1) / np.sqrt(len(rmses))) if len(rmses) > 1 else 0.0
    return {"mae": m, "mae_ci": mci, "rmse": r, "rmse_ci": rci, "mae_folds": maes}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to ml-1m directory")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-ratings", type=int, default=20)
    parser.add_argument("--out", default="movielens_results.json")
    args = parser.parse_args(argv)

    print("Loading MovieLens-1M...")
    ratings, movies = load_ml1m(args.data)
    print("ratings:", len(ratings), "movies:", len(movies))

    results = {"seed": args.seed, "folds": args.folds, "epochs": args.epochs, "pairs": {}}
    for cid, ga, gb in ML_PAIRS:
        print(f"\n==== {cid} {ga}->{gb} (density-ordered) ====")
        S, T, meta = build_genre_pair(ratings, movies, ga, gb, min_ratings=args.min_ratings)
        meta["id"] = cid
        print(meta)
        R_s, R_t = S.to_numpy(), T.to_numpy()
        pair_key = f"{meta['source']}->{meta['target']}"
        pack = {"meta": meta, "methods": {}}

        print("  mu-baseline...")
        mu = cross_validate_mu_baseline(T, n_folds=args.folds, random_state=args.seed, fractions=(0.2, 0.8))
        pack["methods"]["mu"] = {
            "20": {"mae": mu[0.2]["summary"][0], "mae_ci": mu[0.2]["summary"][1],
                   "rmse": mu[0.2]["summary"][2], "rmse_ci": mu[0.2]["summary"][3]},
            "80": {"mae": mu[0.8]["summary"][0], "mae_ci": mu[0.8]["summary"][1],
                   "rmse": mu[0.8]["summary"][2], "rmse_ci": mu[0.8]["summary"][3]},
        }
        print(f"    TR20 MAE={pack['methods']['mu']['20']['mae']:.4f}")

        for name, fn, kw in [
            ("MF", run_mf_once, {"lambda_reg": 0.01, "gamma": 0.1, "learning_rate": 0.005, "n_factors": 10}),
            ("MFB", run_mfb_once, {"lambda_reg": 0.01, "gamma": 0.1, "learning_rate": 0.005, "n_factors": 10}),
            ("DMF", run_dmf_once, {"lambda_reg": 0.01, "learning_rate": 0.005, "n_factors": 10}),
        ]:
            print(f"  {name} TR=20%...")
            t0 = time.time()
            r20 = cv_method(fn, R_s, R_t, n_folds=args.folds, seed=args.seed, train_frac=0.2,
                            epochs=args.epochs, **kw)
            print(f"  {name} TR=80%...")
            r80 = cv_method(fn, R_s, R_t, n_folds=args.folds, seed=args.seed, train_frac=0.8,
                            epochs=args.epochs, **kw)
            pack["methods"][name] = {"20": r20, "80": r80, "seconds_20": time.time() - t0}
            print(f"    TR20 MAE={r20['mae']:.4f} ± {r20['mae_ci']:.4f}")

        results["pairs"][pair_key] = pack

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote", args.out)

    print("=" * 72)
    print("COPY EVERYTHING BELOW THIS LINE AND PASTE INTO CURSOR CHAT")
    print("=" * 72)
    print("TASK: MTAP_Revision_MovieLens_second_dataset")
    print(f"SEED={args.seed} FOLDS={args.folds} EPOCHS={args.epochs}")
    print()
    print("PAIR_META")
    for key, pack in results["pairs"].items():
        m = pack["meta"]
        print(f"{m['id']}\t{key}\tusers={m['n_users']}\titems_t={m['n_items_t']}\t"
              f"ratings_t={m['n_ratings_t']}\tdensity_t={m['density_t']:.6f}")
    print()
    print("RESULTS_TABLE")
    print("ID\tPair\tMethod\tTR%\tMAE\tMAE_CI\tRMSE\tRMSE_CI")
    for key, pack in results["pairs"].items():
        cid = pack["meta"]["id"]
        for method, blob in pack["methods"].items():
            for tr in ("20", "80"):
                r = blob[tr]
                print(f"{cid}\t{key}\t{method}\t{tr}\t{r['mae']:.4f}\t{r.get('mae_ci',0):.4f}\t"
                      f"{r['rmse']:.4f}\t{r.get('rmse_ci',0):.4f}")
    print("=" * 72)
    print("END_OF_OUTPUT")
    print("=" * 72)


if __name__ == "__main__":
    main()
