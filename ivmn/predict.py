#!/usr/bin/env python3
"""Predict in vitro micronucleus (MNvit) outcome for SMILES strings.

Loads the saved logistic-regression classifier (model_ivmn.joblib) and the
frozen ChemBERTa embedding model from Hugging Face (downloaded on first
run, cached locally afterwards). Outputs the predicted class
(0 = MNvit negative, 1 = MNvit positive) plus the positive-class
probability.

Usage:
    python predict.py "CCO" "Nc1ccc2ccccc2c1"
    python predict.py --csv input.csv --smiles-col smiles --out predictions.csv
"""
import argparse
import os
import sys

import joblib
import pandas as pd

from embed import embed_smiles, load_embedding_model

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model_ivmn.joblib")


def main():
    ap = argparse.ArgumentParser(description="Predict MNvit outcome")
    ap.add_argument("smiles", nargs="*", help="SMILES strings to predict")
    ap.add_argument("--csv", help="Input CSV with a SMILES column")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--out", help="Output CSV path (required with --csv)")
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        inputs = df[args.smiles_col].astype(str).tolist()
    elif args.smiles:
        inputs = list(args.smiles)
        df = None
    else:
        ap.print_help()
        sys.exit(1)

    bundle = joblib.load(args.model)
    clf = bundle["model"]
    tokenizer, model = load_embedding_model(bundle["embedding_model"])

    # RDKit validation: ChemBERTa tokenizes anything, so screen garbage.
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    valid = [Chem.MolFromSmiles(s) is not None for s in inputs]

    preds, probas = [], []
    ok_idx = [i for i, v in enumerate(valid) if v]
    if ok_idx:
        emb = embed_smiles([inputs[i] for i in ok_idx], tokenizer, model)
        p = clf.predict(emb)
        pr = clf.predict_proba(emb)[:, 1]
        for j, i in enumerate(ok_idx):
            preds.append((i, int(p[j]), float(pr[j])))
    by_idx = {i: (c, pr) for i, c, pr in preds}

    out_rows = []
    for i, s in enumerate(inputs):
        if i in by_idx:
            cls, proba = by_idx[i]
            out_rows.append({
                "smiles": s,
                "prediction": cls,
                "probability_positive": round(proba, 4),
                "label": "MNvit positive" if cls == 1 else "MNvit negative",
                "error": "",
            })
        else:
            out_rows.append({
                "smiles": s,
                "prediction": "",
                "probability_positive": "",
                "label": "",
                "error": "invalid SMILES",
            })
    out_df = pd.DataFrame(out_rows)
    if args.csv:
        if not args.out:
            ap.error("--out is required with --csv")
        out_df.to_csv(args.out, index=False)
        print(f"wrote {args.out} ({len(out_df)} rows)")
    else:
        print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
