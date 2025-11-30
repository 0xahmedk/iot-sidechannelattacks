# rf_small_repo

Minimal repository containing a small Random Forest training script that reads two CSVs
(attack vs benign) using only the first N rows to keep memory usage low.

Files:

- `train_random_forest_small.py` - main script
- `requirements.txt` - minimal dependencies

Quick start

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your CSVs next to the script or pass full paths. Example (from repo root):

```bash
python train_random_forest_small.py \
  --attack ../data/6h/data-channel-1.csv \
  --benign ../data/pass/data-channel-1.csv \
  --nrows 20000
```

4. For low-memory runs, reduce `--nrows` (e.g., 5000) and `--n_estimators` (e.g., 10).

Make it a git repo

```bash
cd rf_small_repo
git init
git add .
git commit -m "Initial import: small RF training script"
```

That's it — the folder is self-contained for training experiments on small subsets of CSVs.
