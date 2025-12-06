#!/usr/bin/env python3
"""
Author: Ahmed Khan (25I-7633) MS AI FAST NUCES

Simple script to train and evaluate a Random Forest classifier using only a small subset
of rows from two CSV files (attack_data.csv and benign_data.csv). This keeps memory usage
low so it can run on modest machines.

- Reads the first `nrows` rows from each CSV (default 20000)
- Labels attack rows as 1 and benign rows as 0
- Drops rows with missing values
- Uses only numeric columns as features (safe choice for heterogeneous CSVs)
- Splits 70/30 (train/test)
- Trains RandomForestClassifier(n_estimators=50, n_jobs=-1)
- Prints accuracy, classification report, and a simple confusion matrix

Usage:
    python train_random_forest_small.py \
        --attack attack_data.csv --benign benign_data.csv --nrows 20000

Requirements:
    pandas, numpy, scikit-learn

"""

import argparse
import time
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_and_label(attack_path, benign_path, nrows):
    """Load attack and benign CSVs (first nrows each) and add a Label column.

    Returns a single concatenated DataFrame.
    """
    # Read only the first nrows from each file to keep memory low
    print(f"Reading up to {nrows} rows from: {attack_path} and {benign_path}")
    attack = pd.read_csv(attack_path, nrows=nrows)
    benign = pd.read_csv(benign_path, nrows=nrows)

    # Add Label column: 1 for attack, 0 for benign (file-level)
    attack = attack.copy()
    benign = benign.copy()
    attack['Label'] = 1
    benign['Label'] = 0

    # Concatenate and return
    df = pd.concat([attack, benign], ignore_index=True)

    # If the CSVs contain a per-row `Attacks` column, derive labels from it
    # (Label = 1 where Attacks > 0). This prevents accidental file-level
    # mislabeling when files contain mixed rows.
    if 'Attacks' in df.columns:
        # coerce to numeric safely and consider >0 as attack
        df['Label'] = (pd.to_numeric(df['Attacks'], errors='coerce') > 0).fillna(False).astype(int)
        print("Derived per-row labels from 'Attacks' column (Label = Attacks > 0).")

    # Coerce common numeric-ish columns to numeric types when possible so
    # downstream select_dtypes can find them even if some rows are strings.
    for c in ('Attacks', 'Power'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def engineered_preprocess(df, window_size=10):
    """
    Feature Engineering & Preprocessing (engineered version).
    Adds rolling window features (Mean, Std, Max, Min) to capture temporal patterns.
    Then drops NaNs and selects numeric features.
    """
    print(f"Generating rolling features with window size: {window_size} (engineered)...")

    # 1. Ensure data is sorted by Time (critical for rolling windows)
    if 'Time' in df.columns:
        # Coerce/parse Time to a single comparable dtype to avoid failures when
        # the column contains mixed types (ints, floats, strings).
        time_orig = df['Time']

        # Try numeric conversion first (e.g., epoch timestamps)
        time_num = pd.to_numeric(time_orig, errors='coerce')

        # Next, try to parse datetimes by attempting a list of common formats.
        # Specifying formats avoids pandas falling back to per-element parsing
        # and emitting the "Could not infer format" warning.
        common_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%Y-%m-%d',
            '%m/%d/%Y %H:%M:%S',
            '%d-%m-%Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%S',
        ]

        time_dt = pd.Series(pd.NaT, index=time_orig.index)
        for fmt in common_formats:
            parsed = pd.to_datetime(time_orig, format=fmt, errors='coerce')
            mask = parsed.notna() & time_dt.isna()
            if mask.any():
                time_dt.loc[mask] = parsed.loc[mask]

        # If numeric conversion produced the most non-null values, prefer numeric.
        if time_num.notna().sum() >= time_dt.notna().sum():
            df['Time'] = time_num
            n_invalid = time_num.isna().sum()
            if n_invalid:
                print(f"Note: {n_invalid} 'Time' values couldn't be converted to numeric and were set to NaN.")
        else:
            # If some values remain unparsed, we avoid calling pd.to_datetime without a
            # format (which triggers the warning). Instead, we keep the parsed values
            # and report how many failed — ask the user to add their specific format
            # if many entries failed to parse.
            df['Time'] = time_dt
            n_parsed = time_dt.notna().sum()
            n_total = len(time_dt)
            n_failed = n_total - n_parsed
            print(f"Parsed {n_parsed}/{n_total} 'Time' values using common formats; {n_failed} remain unparsed.")
            if n_failed:
                print("If you see many unparsed values, specify the exact Time format in the code or fix the CSV.")

        # Drop rows where Time could not be parsed/coerced
        before_time = len(df)
        df = df.dropna(subset=['Time'])
        after_time = len(df)
        if before_time != after_time:
            print(f"Dropped {before_time - after_time} rows with invalid 'Time' values.")

        # Now safe to sort by Time
        df = df.sort_values('Time')

    # 2. Create Temporal Features on the 'Power' column
    # This adds 'Memory' to your model: it sees the trend, not just the current value.
    if 'Power' in df.columns:
        # Ensure Power is numeric
        df['Power'] = pd.to_numeric(df['Power'], errors='coerce')

        # Calculate statistics over the last N rows
        df['Power_Mean'] = df['Power'].rolling(window=window_size).mean()
        df['Power_Std']  = df['Power'].rolling(window=window_size).std() # Captures volatility/jitter
        df['Power_Max']  = df['Power'].rolling(window=window_size).max() # Captures peaks
        df['Power_Min']  = df['Power'].rolling(window=window_size).min()
    else:
        print("WARNING: 'Power' column not found. Skipping rolling features.")

    # 3. Drop NaNs 
    # This will now remove the original NaNs AND the first 'window_size' rows 
    # which result in NaN during rolling calculations.
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"Dropped {before - after} rows (NaNs + rolling warmup). {after} rows remain.")

    if 'Label' not in df.columns:
        raise ValueError("DataFrame must contain a 'Label' column")

    # 4. Select Numeric Columns
    numeric = df.select_dtypes(include=[np.number]).copy()

    # Safety check for Label
    if 'Label' not in numeric.columns and 'Label' in df.columns:
        numeric['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)

    # Remove 'Attacks' to avoid leakage (Label is already derived from it)
    if 'Attacks' in numeric.columns:
        numeric = numeric.drop(columns=['Attacks'])

    # 5. Split Features (X) and Target (y)
    X = numeric.drop(columns=['Label'])
    y = numeric['Label'].astype(int)

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found. Check your CSV files.")

    print(f"Using {X.shape[0]} rows and {X.shape[1]} numeric features: {list(X.columns)}")
    return X, y


def simple_preprocess(df):
    """
    Simple preprocessing: drop NaNs and use only existing numeric columns (no temporal features).
    This is the baseline to compare against the engineered version.
    """
    print("Using simple features (no temporal/rolling engineering)...")

    df = df.copy()
    # Coerce commonly numeric-ish columns
    for c in ('Attacks', 'Power'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"Dropped {before - after} rows due to NaNs. {after} rows remain.")

    if 'Label' not in df.columns:
        raise ValueError("DataFrame must contain a 'Label' column")

    numeric = df.select_dtypes(include=[np.number]).copy()
    if 'Label' not in numeric.columns and 'Label' in df.columns:
        numeric['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)

    if 'Attacks' in numeric.columns:
        numeric = numeric.drop(columns=['Attacks'])

    X = numeric.drop(columns=['Label'])
    y = numeric['Label'].astype(int)

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found. Check your CSV files.")

    print(f"Using {X.shape[0]} rows and {X.shape[1]} numeric features: {list(X.columns)}")
    return X, y
def train_and_evaluate(X, y, n_estimators=50):
    """Split data, train RandomForest, and print evaluation metrics."""
    # Use stratified split if both classes are present
    unique_labels = y.unique()
    stratify = y if len(unique_labels) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify
    )

    print(f"Training on {len(X_train)} rows, evaluating on {len(X_test)} rows.")

    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)

    start = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"Training completed in {elapsed:.2f} seconds.")

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix with explicit labels [0, 1] so it prints consistent rows/cols
    labels = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("Confusion Matrix (rows=true class, cols=predicted):")
    # Print a simple text-based matrix
    header = "\t" + "\t".join(str(l) for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = "\t".join(str(x) for x in cm[i])
        print(f"{row_label}\t{row}")
    # Return a small summary for programmatic comparison
    return {
        'accuracy': acc,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X.shape[1],
    }

def main():
    parser = argparse.ArgumentParser(
        description="Train a small Random Forest on a subset of two CSVs (attack vs benign)."
    )
    parser.add_argument('--attack', type=str, default='attack_data.csv', help='Path to attack CSV')
    parser.add_argument('--benign', type=str, default='benign_data.csv', help='Path to benign CSV')
    parser.add_argument('--nrows', type=int, default=20000, help='Number of rows to read from each CSV')
    parser.add_argument('--n_estimators', type=int, default=50, help='Number of trees for RandomForest')
    parser.add_argument('--feature-mode', type=str, default='engineered',
                        choices=['simple', 'engineered', 'both'],
                        help="Which feature set to use: simple (baseline), engineered (with rolling features), or both")

    args = parser.parse_args()

    # 1. Load Data
    try:
        df = load_and_label(args.attack, args.benign, args.nrows)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("One of the CSV files is empty or invalid.")
        sys.exit(1)

    # 2. Preprocess according to requested feature mode
    mode = args.feature_mode
    results = {}

    def _run_mode(which):
        if which == 'simple':
            return simple_preprocess(df)
        return engineered_preprocess(df)

    try:
        if mode in ('simple', 'engineered'):
            X, y = _run_mode(mode)
            # Save per-mode processed file (helpful when debugging)
            out = f"processed_data_with_features_{mode}.csv"
            debug_df = X.copy()
            debug_df['Label'] = y
            debug_df.to_csv(out, index=False)
            print(f"[DEBUG] Saved processed data to '{out}'")

            metrics = train_and_evaluate(X, y, n_estimators=args.n_estimators)
            results[mode] = metrics
        else:  # both
            for which in ('simple', 'engineered'):
                X, y = _run_mode(which)
                out = f"processed_data_with_features_{which}.csv"
                debug_df = X.copy()
                debug_df['Label'] = y
                debug_df.to_csv(out, index=False)
                print(f"[DEBUG] Saved processed data to '{out}'")

                metrics = train_and_evaluate(X, y, n_estimators=args.n_estimators)
                results[which] = metrics

    except ValueError as e:
        print(f"Preprocessing error: {e}")
        sys.exit(1)

    # 3. If both modes were run, print a concise comparison
    if args.feature_mode == 'both':
        print('\nComparison of feature modes:')
        for m, stats in results.items():
            print(f"- {m}: acc={stats['accuracy']:.4f}, train={stats['n_train']}, test={stats['n_test']}, features={stats['n_features']}")

if __name__ == '__main__':
    main()
