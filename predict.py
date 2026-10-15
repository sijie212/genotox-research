#!/usr/bin/env python3
"""Predict Ames mutagenicity for SMILES using the trained model in model.joblib.

Usage:
    # Single / multiple SMILES on the command line:
    python predict.py "CCO" "c1ccc(cc1)[N+](=O)[O-]" --model model.joblib

    # Batch from a CSV (requires a column of SMILES):
    python predict.py --csv input.csv --smiles-col smiles --out predictions.csv

Output columns (stdout table or CSV): smiles, prediction (0/1),
probability_mutagenic, label.
"""
from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def smiles_to_fp(smiles: str, radius: int, n_bits: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")


def predict(model, fp_params, smiles_list):
    radius = fp_params.get("radius", 2)
    n_bits = fp_params.get("n_bits", 2048)
    rows = []
    for smi in smiles_list:
        fp = smiles_to_fp(str(smi), radius, n_bits)
        if fp is None:
            rows.append({"smiles": smi, "prediction": None,
                         "probability_mutagenic": None,
                         "label": "INVALID SMILES"})
            continue
        proba = float(model.predict_proba(fp.reshape(1, -1))[0, 1])
        pred = int(proba >= 0.5)
        rows.append({"smiles": smi, "prediction": pred,
                     "probability_mutagenic": round(proba, 4),
                     "label": "mutagenic" if pred == 1 else "non-mutagenic"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Predict Ames mutagenicity from SMILES.")
    ap.add_argument("smiles", nargs="*", help="One or more SMILES strings.")
    ap.add_argument("--model", default="model.joblib", help="Path to trained model file.")
    ap.add_argument("--csv", help="Input CSV containing a SMILES column.")
    ap.add_argument("--smiles-col", default="smiles", help="SMILES column name in --csv.")
    ap.add_argument("--out", help="Write predictions to this CSV (else print table).")
    args = ap.parse_args()

    payload = joblib.load(args.model)
    model = payload["model"]
    fp_params = payload.get("fingerprint", {})

    if args.csv:
        df = pd.read_csv(args.csv)
        if args.smiles_col not in df.columns:
            sys.exit(f"Column '{args.smiles_col}' not found in {args.csv} "
                     f"(columns: {list(df.columns)})")
        smiles_list = df[args.smiles_col].astype(str).tolist()
    elif args.smiles:
        smiles_list = args.smiles
    else:
        sys.exit("Provide SMILES on the command line or use --csv.")

    results = predict(model, fp_params, smiles_list)
    if args.out:
        results.to_csv(args.out, index=False)
        print(f"Wrote {len(results)} predictions to {args.out}")
    else:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
