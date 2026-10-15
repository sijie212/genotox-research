#!/usr/bin/env python3
"""Train a binary Ames mutagenicity classifier on the AmesFormer clean open Ames set.

Pipeline:
  1. Load the dataset (download from the public AmesFormer mirror if missing).
  2. Parse SMILES with RDKit; drop unparseable rows.
  3. Featurize with Morgan fingerprints (radius 2, 2048 bits).
  4. Stratified 80/10/10 train/validation/test split (seed 42).
  5. Train RandomForest and HistGradientBoosting on the train split; pick the
     better one by validation ROC-AUC; refit the winner on train+validation.
  6. Evaluate on the held-out test set and save the model + metrics.

Usage:
    python train.py [--data data_combined.csv] [--out model.joblib]

Outputs:
    model.joblib   dict with keys: model, model_type, metrics, fingerprint,
                   n_train, dataset, label
    metrics.json   test-set metrics (also printed to stdout)
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

DATA_URL = ("https://raw.githubusercontent.com/luke-a-thompson/AmesFormer/"
            "main/data_cleaning/Combined_2s_as_0s_publication.csv")
RANDOM_STATE = 42
FP_RADIUS = 2
FP_NBITS = 2048


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Downloading dataset from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    df = pd.read_csv(path)
    required = {"smiles", "ames"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    df = df[["smiles", "ames"]].dropna()
    df = df[df["smiles"].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    df["ames"] = df["ames"].astype(int)
    if set(df["ames"].unique()) - {0, 1}:
        raise ValueError(f"Labels must be binary 0/1, got {sorted(df['ames'].unique())}")
    return df


def smiles_to_fp(smiles: str, radius: int = FP_RADIUS, n_bits: int = FP_NBITS):
    """Morgan fingerprint as a numpy array of 0/1 floats, or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")


def featurize(df: pd.DataFrame):
    fps, labels, kept_smiles = [], [], []
    n_bad = 0
    for smi, y in zip(df["smiles"], df["ames"]):
        fp = smiles_to_fp(str(smi))
        if fp is None:
            n_bad += 1
            continue
        fps.append(fp)
        labels.append(int(y))
        kept_smiles.append(smi)
    X = np.asarray(fps, dtype=np.float32)
    y = np.asarray(labels, dtype=int)
    return X, y, kept_smiles, n_bad


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),  # [[TN, FP], [FN, TP]]
    }


def main():
    ap = argparse.ArgumentParser(description="Train Ames mutagenicity classifier.")
    ap.add_argument("--data", default="data_combined.csv", help="Path to Ames CSV (downloaded if missing).")
    ap.add_argument("--out", default="model.joblib", help="Where to save the trained model.")
    args = ap.parse_args()

    df = load_data(args.data)
    print(f"Dataset: {len(df)} rows, {int((df['ames'] == 1).sum())} mutagenic (1) / "
          f"{int((df['ames'] == 0).sum())} non-mutagenic (0)")

    X, y, _, n_bad = featurize(df)
    print(f"Fingerprints: {X.shape[0]} usable ({n_bad} SMILES failed to parse and were dropped)")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_tmp)
    print(f"Split (stratified, seed {RANDOM_STATE}): "
          f"train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=250, n_jobs=-1, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, random_state=RANDOM_STATE),
    }
    val_scores = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_scores[name] = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        print(f"Validation ROC-AUC: {name} = {val_scores[name]:.4f}")

    winner = max(val_scores, key=val_scores.get)
    print(f"Selected model: {winner} (best validation ROC-AUC)")

    # Refit the winner on train + validation, then evaluate once on held-out test.
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    final = candidates[winner]
    final.fit(X_trainval, y_trainval)
    metrics = evaluate(final, X_test, y_test)
    print("Test metrics:", json.dumps(metrics, indent=2))

    payload = {
        "model": final,
        "model_type": winner,
        "metrics": metrics,
        "fingerprint": {"type": "Morgan", "radius": FP_RADIUS, "n_bits": FP_NBITS},
        "label": "1 = mutagenic (Ames positive), 0 = non-mutagenic (Ames negative)",
        "n_train": int(len(y_trainval)),
        "dataset": "AmesFormer Combined_2s_as_0s_publication.csv "
                   "(Hansen + ISSSTY + EURL sources, deduped)",
    }
    joblib.dump(payload, args.out)
    print(f"Saved model to {args.out}")

    with open("metrics.json", "w") as f:
        json.dump({"model_type": winner, "test": metrics}, f, indent=2)
    print("Saved metrics to metrics.json")


if __name__ == "__main__":
    main()
