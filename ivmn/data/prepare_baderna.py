#!/usr/bin/env python3
"""Prepare the labeled MNvit dataset from Baderna et al. 2020, SI Table (mmc2).

Source: Baderna D. et al., "New in silico models to predict in vitro
micronucleus induction as marker of genotoxicity", J. Hazard. Mater. 385
(2020) 121638, doi:10.1016/j.jhazmat.2019.121638 — Supplementary Table mmc2
(DATASET sheet): 380 compounds with NAME, CAS, (VEGA) SMILES, MN activity
(1 = positive, 0 = negative) and the authors' TRaining/TeSt split flag.

Why this source: the Toxics 2023 (MDPI) supplementary Tables S1/S2 list
compound names/SMILES/CAS but NO outcome labels ("Toxicity data is available
from the corresponding author upon reasonable request"), so that SI cannot
supply a labeled training set. Baderna mmc2 is the closest peer-reviewed,
publicly downloadable, labeled MNvit set (its sources: OECD eChemPortal,
ECVAM retrospective validation of the in vitro micronucleus test, SCCS/SCCP
opinions, and literature — see mmc1).

This script downloads mmc2 from the publisher CDN, converts it to CSV,
drops the single duplicate SMILES (same label on both rows), and writes
data/mnvit_baderna.csv with columns: smiles, label, name, cas,
author_split (TR/TS, preserved for provenance; the training protocol in
train_fewshot.py uses its own stratified split).
"""
import csv
import os
import urllib.request

MMC2_URL = ("https://ars.els-cdn.com/content/image/1-s2.0-S0304389419315924"
            "-mmc2.xlsx")
RAW = "data/baderna-mmc2.xlsx"
OUT = "data/mnvit_baderna.csv"


def main():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(RAW):
        print(f"downloading {MMC2_URL}")
        req = urllib.request.Request(MMC2_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(RAW, "wb") as fh:
            fh.write(resp.read())
    import openpyxl
    wb = openpyxl.load_workbook(RAW, read_only=True, data_only=True)
    rows = list(wb["DATASET"].iter_rows(values_only=True))[1:]

    seen = {}
    n_dup = 0
    for _id, name, cas, smiles, activity, split in rows:
        smiles = (smiles or "").strip()
        if not smiles or activity is None:
            continue
        if smiles in seen:
            n_dup += 1
            if seen[smiles]["label"] != int(activity):
                print(f"WARN conflicting duplicate SMILES dropped: {smiles}")
            continue
        seen[smiles] = {
            "smiles": smiles,
            "label": int(activity),
            "name": name or "",
            "cas": cas or "",
            "author_split": split or "",
        }
    out_rows = sorted(seen.values(), key=lambda r: r["smiles"])
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["smiles", "label", "name", "cas",
                                           "author_split"])
        w.writeheader()
        w.writerows(out_rows)
    n_pos = sum(r["label"] for r in out_rows)
    print(f"wrote {OUT}: {len(out_rows)} compounds "
          f"({n_pos} positive / {len(out_rows)-n_pos} negative), "
          f"{n_dup} duplicate rows dropped")


if __name__ == "__main__":
    main()
