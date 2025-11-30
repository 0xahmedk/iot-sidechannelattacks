Note: this repo includes two small sample CSVs so you can run the example out-of-the-box:

- `attack_data.csv` (first ~20k rows copied from a workspace source)
- `benign_data.csv` (first ~20k rows copied from a workspace source)

Run the example using those files from the repo root:

```bash
python train_random_forest_small.py --attack attack_data.csv --benign benign_data.csv --nrows 20000
```
