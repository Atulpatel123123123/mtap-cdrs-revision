# MTAP CDRS Revision Experiments

Code for MTAP major-revision extras (global-mean μ baseline, fold stats helpers).

## Colab (recommended)

1. Open [`colab_run.ipynb`](colab_run.ipynb) in Google Colab  
   or use:
   `https://colab.research.google.com/github/Atulpatel123123123/mtap-cdrs-revision/blob/main/colab_run.ipynb`
2. Run all cells.
3. Copy the printed block (`COPY EVERYTHING...` → `END_OF_OUTPUT`) and paste into Cursor.

When code is updated on GitHub, re-run the notebook (it `git pull`s latest). You only need the **output**.

## Local

```bash
pip install -r requirements.txt
python -m src.run_mu_baseline --meta /path/to/amazon-meta.txt
```

`amazon-meta.txt` is the SNAP Amazon co-purchasing meta file (Book/Music/DVD/Video). If missing, Colab runner downloads it from SNAP.
