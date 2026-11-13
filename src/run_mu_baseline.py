"""Run global-mean (μ) baseline for all six CX pairs. Print paste-ready output."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.utils import shuffle

from .amazon_meta import (
    PAPER_MAP,
    PAPER_SPLITS,
    build_pair_matrices,
    ensure_amazon_meta,
    parse_amazon_meta,
)


def evaluate_mu_baseline(R_true, mu):
    mask = R_true > 0
    if not np.any(mask):
        return np.nan, np.nan
    err = R_true[mask] - mu
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def cross_validate_mu_baseline(
    target_rating_matrix,
    n_folds=5,
    random_state=42,
    fractions=(0.2, 0.4, 0.6, 0.8),
):
    R = (
        target_rating_matrix.to_numpy()
        if hasattr(target_rating_matrix, "to_numpy")
        else np.asarray(target_rating_matrix)
    )
    pairs = [(u, i, R[u, i]) for u in range(R.shape[0]) for i in range(R.shape[1]) if R[u, i] > 0]
    pairs = shuffle(pairs, random_state=random_state)
    fold_size = len(pairs) // n_folds

    frac_mae = {f: [] for f in fractions}
    frac_rmse = {f: [] for f in fractions}

    for k in range(n_folds):
        te_idx = set(range(k * fold_size, (k + 1) * fold_size if k < n_folds - 1 else len(pairs)))
        te = [pairs[j] for j in sorted(te_idx)]
        tr = [pairs[j] for j in range(len(pairs)) if j not in te_idx]

        R_test = np.zeros_like(R)
        for u, i, r in te:
            R_test[u, i] = r

        idx = shuffle(list(range(len(tr))), random_state=random_state)
        for frac in fractions:
            cut = max(1, int(len(tr) * frac))
            partial = [tr[j] for j in idx[:cut]]
            R_part = np.zeros_like(R)
            for u, i, r in partial:
                R_part[u, i] = r
            mu = R_part[R_part > 0].mean() if np.any(R_part > 0) else 0.0
            mae_p, rmse_p = evaluate_mu_baseline(R_test, mu)
            frac_mae[frac].append(mae_p)
            frac_rmse[frac].append(rmse_p)

    def summarize(maes, rmses):
        m = float(np.mean(maes))
        r = float(np.mean(rmses))
        mci = float(1.96 * np.std(maes, ddof=1) / np.sqrt(len(maes)))
        rci = float(1.96 * np.std(rmses, ddof=1) / np.sqrt(len(rmses)))
        return m, mci, r, rci

    out = {}
    for frac in fractions:
        out[frac] = {
            "mae_folds": frac_mae[frac],
            "rmse_folds": frac_rmse[frac],
            "summary": summarize(frac_mae[frac], frac_rmse[frac]),
        }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=None, help="Path to amazon-meta.txt")
    parser.add_argument("--threshold", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    meta_path = ensure_amazon_meta(args.meta)
    print("Using meta:", meta_path)
    print("Parsing amazon-meta (may take several minutes)...")
    df = parse_amazon_meta(meta_path)
    print("rows:", len(df))
    print(df["Group"].value_counts().head(10))
    print("rating min/max/mean:", df["Rating"].min(), df["Rating"].max(), df["Rating"].mean())

    required = ["Book", "Music", "DVD", "Video"]
    df = df[df["Group"].isin(required)].copy()
    df = df.drop_duplicates(subset=["Customerid", "ASIN", "Group", "Rating"])

    pair_data = {}
    for cx, source, target in PAPER_SPLITS:
        _S, T, meta = build_pair_matrices(
            df,
            source,
            target,
            args.threshold,
            source_group=source,
            target_group=target,
        )
        key = f"{meta['source']}->{meta['target']}"
        meta["cx"] = cx
        pair_data[key] = {"T": T, "meta": meta}
        print(cx, key, meta)

    all_results = {}
    for key, pack in pair_data.items():
        print("\n====", key, "====")
        res = cross_validate_mu_baseline(
            pack["T"], n_folds=args.folds, random_state=args.seed, fractions=(0.2, 0.4, 0.6, 0.8)
        )
        all_results[key] = res
        for frac, payload in res.items():
            m, mci, r, rci = payload["summary"]
            print(
                f"mu-baseline TR={int(frac*100)}%: MAE={m:.4f} ± {mci:.4f} | RMSE={r:.4f} ± {rci:.4f}"
            )

    print("=" * 72)
    print("COPY EVERYTHING BELOW THIS LINE AND PASTE INTO CURSOR CHAT")
    print("=" * 72)
    print("TASK: MTAP_Revision_mu_baseline")
    print(f"SEED={args.seed} FOLDS={args.folds} THRESHOLD={args.threshold}")
    print(f"PATH={meta_path}")
    print()
    print("PAIR_META")
    for key, pack in pair_data.items():
        m = pack["meta"]
        cx = PAPER_MAP.get(key, "NA")
        print(
            f"{cx}\t{key}\tusers={m['n_users']}\titems_t={m['n_items_t']}\t"
            f"ratings_t={m['n_ratings_t']}\tdensity_t={m['density_t']:.6f}\tmean_r={m['mean_rating_t']:.4f}"
        )
    print()
    print("MU_BASELINE_TABLE")
    print("CX\tPair\tTR%\tMAE\tMAE_CI\tRMSE\tRMSE_CI")
    for key, res in all_results.items():
        cx = PAPER_MAP.get(key, "NA")
        for frac, payload in res.items():
            m, mci, r, rci = payload["summary"]
            print(f"{cx}\t{key}\t{int(frac*100)}\t{m:.4f}\t{mci:.4f}\t{r:.4f}\t{rci:.4f}")
    print()
    print("MU_BASELINE_FOLDS_MAE_20PCT")
    for key, res in all_results.items():
        cx = PAPER_MAP.get(key, "NA")
        folds = ",".join(f"{x:.4f}" for x in res[0.2]["mae_folds"])
        print(f"{cx}\t{key}\t{folds}")
    print("=" * 72)
    print("END_OF_OUTPUT")
    print("=" * 72)


if __name__ == "__main__":
    main()
