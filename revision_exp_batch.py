"""Multi-CX significance + insert mu-baseline cells into notebooks."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent

PAIRS = {
    "CX1": {
        "MFB": [0.6789, 0.6880, 0.6799, 0.6877, 0.6826],
        "DMF": [0.7603, 0.7641, 0.7583, 0.7907, 0.7448],
    },
    "CX2": {
        "MFB": [0.6631, 0.6660, 0.6618, 0.6569, 0.6581],
        "DMF": [0.8072, 0.8245, 0.8244, 0.7964, 0.8121],
    },
    "CX3": {
        "MFB": [0.6617, 0.6614, 0.6586, 0.6633, 0.6604],
        "DMF": [0.7937, 0.8358, 0.7909, 0.8099, 0.7980],
    },
    "CX4": {
        "MFB": [0.6846, 0.6881, 0.6886, 0.6893, 0.6870],
        "DMF": [0.7741, 0.7693, 0.7690, 0.7661, 0.7536],
    },
    "CX5": {
        "MFB": [0.6815, 0.6781, 0.6942, 0.6847, 0.6958],
        "DMF": [0.7615, 0.7590, 0.7907, 0.7809, 0.7990],
    },
    "CX6": {
        "MFB": [0.7954, 0.7909, 0.7948, 0.7947, 0.7975],
        "DMF": [0.9138, 0.9137, 0.9145, 0.9122, 0.9145],
    },
}

MU_MD = (
    "# Revision extras: global-mean (μ) baseline\n"
    "Run after `target_rating_matrix` is built. Same seed 42 and fractions 20/40/60/80% as MF experiments."
)

MU_CODE = '''
import numpy as np
from sklearn.utils import shuffle

def evaluate_mu_baseline(R_true, mu):
    mask = R_true > 0
    if not np.any(mask):
        return np.nan, np.nan
    err = R_true[mask] - mu
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))

def cross_validate_mu_baseline(target_rating_matrix, n_folds=5, random_state=42,
                               fractions=(0.2, 0.4, 0.6, 0.8)):
    R = target_rating_matrix.to_numpy() if hasattr(target_rating_matrix, "to_numpy") else np.asarray(target_rating_matrix)
    pairs = [(u, i, R[u, i]) for u in range(R.shape[0]) for i in range(R.shape[1]) if R[u, i] > 0]
    pairs = shuffle(pairs, random_state=random_state)
    fold_size = len(pairs) // n_folds
    frac_mae = {f: [] for f in fractions}
    frac_rmse = {f: [] for f in fractions}
    full_mae, full_rmse = [], []

    for k in range(n_folds):
        te = pairs[k * fold_size : (k + 1) * fold_size] if k < n_folds - 1 else pairs[k * fold_size :]
        te_idx = set(range(k * fold_size, (k + 1) * fold_size if k < n_folds - 1 else len(pairs)))
        tr = [pairs[j] for j in range(len(pairs)) if j not in te_idx]
        R_train = np.zeros_like(R)
        R_test = np.zeros_like(R)
        for u, i, r in tr:
            R_train[u, i] = r
        for u, i, r in te:
            R_test[u, i] = r
        mu_full = R_train[R_train > 0].mean() if np.any(R_train > 0) else 0.0
        mae, rmse = evaluate_mu_baseline(R_test, mu_full)
        full_mae.append(mae)
        full_rmse.append(rmse)
        print(f"Fold {k+1}: FULL - MAE={mae:.4f}, RMSE={rmse:.4f} (mu={mu_full:.4f})")
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
            print(f"Fold {k+1}: {int(frac*100)}% - MAE={mae_p:.4f}, RMSE={rmse_p:.4f} (mu={mu:.4f})")

    def summarize(name, maes, rmses):
        m = np.mean(maes)
        r = np.mean(rmses)
        mci = 1.96 * np.std(maes, ddof=1) / np.sqrt(len(maes))
        rci = 1.96 * np.std(rmses, ddof=1) / np.sqrt(len(rmses))
        print(f"=== {name} ===")
        print(f"MAE : {m:.4f} ± {mci:.4f}")
        print(f"RMSE: {r:.4f} ± {rci:.4f}")
        return m, mci, r, rci

    summarize("mu-baseline FULL", full_mae, full_rmse)
    results = {}
    for frac in fractions:
        results[frac] = summarize(f"mu-baseline {int(frac*100)}%", frac_mae[frac], frac_rmse[frac])
    return results

mu_baseline_results = cross_validate_mu_baseline(target_rating_matrix)
print(mu_baseline_results)
'''.lstrip()


def print_stats():
    print("MFB vs DMF @20% target training")
    rows = []
    for cx, d in PAIRS.items():
        r = stats.ttest_rel(d["MFB"], d["DMF"])
        w = stats.wilcoxon(d["MFB"], d["DMF"])
        row = {
            "cx": cx,
            "mfb": float(np.mean(d["MFB"])),
            "dmf": float(np.mean(d["DMF"])),
            "t": float(r.statistic),
            "p": float(r.pvalue),
            "wilcox": float(w.pvalue),
        }
        rows.append(row)
        print(
            f"{cx}: MFB={row['mfb']:.4f} DMF={row['dmf']:.4f} "
            f"t={row['t']:.2f} p={row['p']:.2e} wilcox={row['wilcox']:.4f}"
        )
    return rows


def insert_mu_cells():
    skip = "Comprative_MF(B00k_Music).ipynb"
    for nbpath in sorted(BASE.glob("Comprative_MF*.ipynb")):
        if nbpath.name == skip:
            print("skip", nbpath.name)
            continue
        nb = json.loads(nbpath.read_text(encoding="utf-8"))
        insert_at = None
        for i, c in enumerate(nb["cells"]):
            src = "".join(c.get("source", []))
            if "target_rating_matrix =" in src:
                insert_at = i + 1
        if insert_at is None:
            print("no matrix in", nbpath.name)
            continue
        already = any(
            "mu-baseline" in "".join(c.get("source", [])).lower()
            or "global-mean" in "".join(c.get("source", [])).lower()
            for c in nb["cells"]
        )
        if already:
            print("already has mu cell", nbpath.name)
            continue
        md = {"cell_type": "markdown", "metadata": {}, "source": [MU_MD + "\n"]}
        code = {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [ln + "\n" for ln in MU_CODE.split("\n")],
        }
        if code["source"]:
            code["source"][-1] = code["source"][-1].rstrip("\n")
        nb["cells"][insert_at:insert_at] = [md, code]
        nbpath.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
        print("inserted into", nbpath.name, "at", insert_at)


if __name__ == "__main__":
    rows = print_stats()
    insert_mu_cells()
    out = BASE / "sig_mfb_vs_dmf_20pct.json"
    import json as _json

    out.write_text(_json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote", out)
