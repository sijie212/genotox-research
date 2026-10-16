#!/usr/bin/env python3
"""Few-shot evaluation + final model for MNvit on frozen ChemBERTa embeddings.

Protocol (fixed, seed 42):
  * Data: data/mnvit_baderna.csv (Baderna et al. 2020, 379 compounds).
  * Split: stratified 70/15/15 train/val/test (seed 42). The test set is
    fixed and never used for training or model selection.
  * Embeddings: frozen ChemBERTa (see embed.py), mean-pooled, 384-dim.
  * Few-shot: for each training-subset size in {16, 32, 64, 128}, draw
    5 stratified subsets from the train split (seeds 0-4), train a
    classifier on each, and report mean +/- std of test metrics.
    Two classifier families are compared: logistic regression
    (class_weight='balanced') and kNN (k=5, cosine distance).
  * Final model: logistic regression trained on train+val (85% of the
    data), evaluated once on the fixed test set. Saved to
    model_ivmn.joblib (small: coefficients only; embeddings stay on HF).

Outputs: metrics.json, data/splits.csv, model_ivmn.joblib,
data/embeddings.npz (cache, regenerable via embed.py).
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from embed import DEFAULT_MODEL, embed_smiles, load_embedding_model

SEED = 42
SUBSET_SIZES = [16, 32, 64, 128]
N_REPEATS = 5
DATA_CSV = "data/mnvit_baderna.csv"
EMB_CACHE = "data/embeddings.npz"
SPLITS_CSV = "data/splits.csv"
MODEL_OUT = "model_ivmn.joblib"
METRICS_OUT = "metrics.json"


def valid_smiles_mask(smiles_list):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    return [Chem.MolFromSmiles(s) is not None for s in smiles_list]


def make_classifier(kind):
    if kind == "logreg":
        return LogisticRegression(C=1.0, class_weight="balanced",
                                  max_iter=2000, random_state=SEED)
    if kind == "knn":
        return KNeighborsClassifier(n_neighbors=5, metric="cosine")
    raise ValueError(f"unknown classifier: {kind}")


def metrics_for(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main():
    df = pd.read_csv(DATA_CSV)
    keep = valid_smiles_mask(df["smiles"].astype(str).tolist())
    dropped = int(np.sum(~np.array(keep)))
    if dropped:
        print(f"dropped {dropped} unparseable SMILES")
    df = df[keep].reset_index(drop=True)
    smiles = df["smiles"].astype(str).tolist()
    y = df["label"].astype(int).to_numpy()

    # Fixed stratified 70/15/15 split (seed 42).
    idx = np.arange(len(df))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.15, stratify=y, random_state=SEED)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.15 / 0.85,
        stratify=y[idx_trainval], random_state=SEED)
    split_of = {}
    for i in idx_train:
        split_of[i] = "train"
    for i in idx_val:
        split_of[i] = "val"
    for i in idx_test:
        split_of[i] = "test"
    df["split"] = [split_of[i] for i in idx]
    df.to_csv(SPLITS_CSV, index=False)
    print(f"split: train={len(idx_train)} val={len(idx_val)} "
          f"test={len(idx_test)}")

    # Embeddings (cached; regenerable via embed.py).
    if os.path.exists(EMB_CACHE):
        cache = np.load(EMB_CACHE, allow_pickle=True)
        if list(cache["smiles"]) == smiles:
            emb = cache["embeddings"]
            print(f"loaded cached embeddings {EMB_CACHE}: {emb.shape}")
        else:
            emb = None
    else:
        emb = None
    if emb is None:
        tokenizer, model = load_embedding_model(DEFAULT_MODEL)
        emb = embed_smiles(smiles, tokenizer, model)
        np.savez_compressed(EMB_CACHE, embeddings=emb,
                            smiles=np.array(smiles))
        print(f"computed embeddings: {emb.shape}")

    X_train, y_train = emb[idx_train], y[idx_train]
    X_val, y_val = emb[idx_val], y[idx_val]
    X_test, y_test = emb[idx_test], y[idx_test]

    # Few-shot evaluation: stratified subsets of the train split.
    fewshot = {}
    for n in SUBSET_SIZES:
        for kind in ("logreg", "knn"):
            runs = []
            for rep in range(N_REPEATS):
                sub_idx, _ = train_test_split(
                    np.arange(len(idx_train)), train_size=n,
                    stratify=y_train, random_state=rep)
                clf = make_classifier(kind)
                clf.fit(X_train[sub_idx], y_train[sub_idx])
                proba = clf.predict_proba(X_test)[:, 1]
                runs.append(metrics_for(y_test, clf.predict(X_test), proba))
            agg = {}
            for key in ("accuracy", "roc_auc", "precision", "recall", "f1"):
                vals = [r[key] for r in runs]
                agg[key] = {"mean": float(np.mean(vals)),
                            "std": float(np.std(vals))}
            fewshot[f"{kind}@{n}"] = agg
            print(f"few-shot {kind} n={n}: "
                  f"AUC {agg['roc_auc']['mean']:.3f}+/-{agg['roc_auc']['std']:.3f} "
                  f"acc {agg['accuracy']['mean']:.3f} "
                  f"F1 {agg['f1']['mean']:.3f}")

    # Final models on train+val, evaluated once on the fixed test set.
    X_full = np.concatenate([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    final = {}
    for kind in ("logreg", "knn"):
        clf = make_classifier(kind)
        clf.fit(X_full, y_full)
        proba = clf.predict_proba(X_test)[:, 1]
        final[kind] = metrics_for(y_test, clf.predict(X_test), proba)
        final[kind]["n"] = int(len(y_test))
        print(f"final {kind} on test: {final[kind]}")

    # Save the logistic-regression final model (the default).
    clf_final = make_classifier("logreg")
    clf_final.fit(X_full, y_full)
    joblib.dump({
        "model": clf_final,
        "embedding_model": DEFAULT_MODEL,
        "hidden_size": int(emb.shape[1]),
        "label_meaning": {0: "MNvit negative", 1: "MNvit positive"},
    }, MODEL_OUT)
    print(f"saved {MODEL_OUT} "
          f"({os.path.getsize(MODEL_OUT)} bytes)")

    metrics = {
        "dataset": {
            "source": "Baderna et al. 2020, J. Hazard. Mater. 385:121638, "
                      "SI mmc2 (see data/prepare_baderna.py)",
            "n_compounds": int(len(df)),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "dropped_unparseable_smiles": dropped,
        },
        "embedding_model": DEFAULT_MODEL,
        "embedding_dim": int(emb.shape[1]),
        "split": {"seed": SEED, "train": int(len(idx_train)),
                  "val": int(len(idx_val)), "test": int(len(idx_test))},
        "few_shot": fewshot,
        "final_test": final,
        "final_model": "logreg trained on train+val",
    }
    with open(METRICS_OUT, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"wrote {METRICS_OUT}")


if __name__ == "__main__":
    main()
