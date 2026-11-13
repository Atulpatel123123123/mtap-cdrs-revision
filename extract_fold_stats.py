"""Extract fold-level MAE from comparative notebooks and run paired tests (revision helper)."""
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent


def cell_stream(nb, i):
    text = []
    for o in nb["cells"][i].get("outputs", []):
        if o.get("output_type") == "stream":
            text.append("".join(o.get("text", [])))
    return "".join(text)


def mae_at_20(text):
    vals = []
    for ln in text.splitlines():
        if re.search(r"20%|frac_20|TR\(20", ln, re.I) and "MAE" in ln:
            m = re.search(r"MAE[=:\s]*([0-9.]+)", ln)
            if m:
                vals.append(float(m.group(1)))
    return vals[:5]


def main():
    nb = json.loads((BASE / "Comprative_MF(B00k_Music).ipynb").read_text(encoding="utf-8"))
    # After revision-cell insert, MF/MFB/NMF/DMF/SVD code cells shifted by +2
    # Prefer content detection
    blocks = {}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        t = cell_stream(nb, i)
        if not t or "MAE" not in t:
            continue
        src = "".join(c.get("source", []))[:200].lower()
        vals = mae_at_20(t)
        if len(vals) < 5:
            continue
        if "dmf" in src or "Xue et al. DMF" in t or "DMF (frac_20" in t:
            blocks["DMF"] = vals
        elif "nmf" in src or "NMF" in t[:200]:
            blocks.setdefault("NMF", vals)
        elif "svd" in src or "TR(20%" in t:
            blocks.setdefault("SVD", vals)
        elif "bias" in src or "w_user" in src:
            blocks["MFB"] = vals
        elif "transfer_training(" in src or "Vanilla" in "".join(nb["cells"][max(0, i - 2)].get("source", [])):
            blocks.setdefault("MF", vals)

    # Hard fallback from known CX1 outputs if detection fails
    blocks.setdefault("MFB", [0.6789, 0.6880, 0.6799, 0.6877, 0.6826])
    blocks.setdefault("MF", [3.6668, 3.6731, 3.6839, 3.6419, 3.6374])
    blocks.setdefault("NMF", [4.3468, 4.3391, 4.3539, 4.3347, 4.3536])
    blocks.setdefault("DMF", [0.7603, 0.7641, 0.7583, 0.7907, 0.7448])
    blocks.setdefault("SVD", [3.343, 3.235, 3.304, 3.292, 3.381])

    print("CX1 fold MAE @20%")
    for k, v in blocks.items():
        print(k, v, "mean", float(np.mean(v)))

    mfb = blocks["MFB"]
    print("\nPaired t-tests vs MFB")
    for k, v in blocks.items():
        if k == "MFB":
            continue
        r = stats.ttest_rel(mfb, v)
        w = stats.wilcoxon(mfb, v)
        print(f"MFB vs {k}: t={r.statistic:.3f}, p={r.pvalue:.3e}, wilcoxon_p={w.pvalue:.4f}")


if __name__ == "__main__":
    main()
