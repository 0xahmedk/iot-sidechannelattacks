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

    # Add Label column: 1 for attack, 0 for benign
    attack = attack.copy()
    benign = benign.copy()
    attack['Label'] = 1
    benign['Label'] = 0

    # Concatenate and return
    df = pd.concat([attack, benign], ignore_index=True)
    return df


def preprocess(df):
    """Drop NaNs and select numeric features only (safe for heterogeneous CSVs).

    Returns X (features dataframe) and y (labels Series).
    """
    # Drop rows with any missing values to avoid surprises during training
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"Dropped {before - after} rows containing NaNs. {after} rows remain.")

    if 'Label' not in df.columns:
        raise ValueError("DataFrame must contain a 'Label' column")

    # Keep only numeric columns (this automatically excludes strings like timestamps)
    numeric = df.select_dtypes(include=[np.number])

    if 'Label' not in numeric.columns:
        # If Label was not numeric for some reason, coerce it
        numeric['Label'] = df['Label'].astype(int)

    # Separate features and label
    X = numeric.drop(columns=['Label'])
    y = numeric['Label'].astype(int)

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found. Check your CSV files.")

    print(f"Using {X.shape[0]} rows and {X.shape[1]} numeric features for training.")
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


def main():
    parser = argparse.ArgumentParser(
        description="Train a small Random Forest on a subset of two CSVs (attack vs benign)."
    )
    parser.add_argument('--attack', type=str, default='attack_data.csv', help='Path to attack CSV')
    parser.add_argument('--benign', type=str, default='benign_data.csv', help='Path to benign CSV')
    parser.add_argument('--nrows', type=int, default=20000, help='Number of rows to read from each CSV')
    parser.add_argument('--n_estimators', type=int, default=50, help='Number of trees for RandomForest')

    args = parser.parse_args()

    try:
        df = load_and_label(args.attack, args.benign, args.nrows)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("One of the CSV files is empty or invalid.")
        sys.exit(1)

    try:
        X, y = preprocess(df)
    except ValueError as e:
        print(f"Preprocessing error: {e}")
        sys.exit(1)

    train_and_evaluate(X, y, n_estimators=args.n_estimators)


if __name__ == '__main__':
    main()
