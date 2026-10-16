#!/usr/bin/env python3
"""Harvest a labeled in vitro micronucleus (MNvit) dataset from ChEMBL.

ATTEMPTED AND ABANDONED (kept for provenance): this script runs and finds
in vitro MN assays, but yields ZERO labeled records, because ChEMBL
toxicity (T-type) MN assay records carry no outcome in any structured
field (activity_comment is null; the call lives only in the source
publication). Do not re-run expecting labels. The labeled dataset used
by this project comes from data/prepare_baderna.py instead.

(Original rationale, kept for context: the Toxics 2023 SI lists
names/SMILES/CAS but NO outcome labels, so that SI cannot supply a
labeled training set.)

  * assays whose description contains "micronucleus", filtered to in vitro
    MN-induction assays (must mention "in vitro"; must NOT mention
    "antimutagenic" or "in vivo");
  * activities with activity_comment Active/Inactive only;
  * canonical SMILES from the activity record;
  * dedupe by canonical SMILES; compounds with conflicting calls across
    assays are dropped as ambiguous (documented in the log).

Output: data/chembl_mnvit_labeled.csv with columns
  smiles, label (1 = Active/positive, 0 = Inactive/negative),
  n_assays, n_active_calls, n_inactive_calls
"""
import csv
import json
import sys
import urllib.parse
import urllib.request
from collections import defaultdict

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
OUT = "data/chembl_mnvit_labeled.csv"


def get_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_assays():
    url = (f"{CHEMBL}/assay.json?description__icontains=micronucleus"
           f"&limit=1000")
    data = get_json(url)
    return data.get("assays", [])


def is_invitro_mn_induction(desc):
    d = (desc or "").lower()
    if "micronucleus" not in d and "micronuclei" not in d:
        return False
    if "antimutagenic" in d or "anti-mutagenic" in d:
        return False
    if "in vivo" in d:
        return False
    if "in vitro" in d:
        return True
    # Some MN induction assays omit "in vitro" but name a cell system.
    cell_markers = ("cho cells", "v79 cells", "tk6", "lymphocytes",
                    "chok1", "cells assessed", "cells by micronucleus")
    return any(m in d for m in cell_markers)


def fetch_activities(assay_id):
    url = (f"{CHEMBL}/activity.json?assay_chembl_id={assay_id}"
           f"&limit=1000")
    data = get_json(url)
    return data.get("activities", [])


def main():
    assays = fetch_assays()
    print(f"assays mentioning micronucleus: {len(assays)}", file=sys.stderr)
    kept = [a for a in assays
            if is_invitro_mn_induction(a.get("description"))]
    print(f"kept in vitro MN-induction assays: {len(kept)}", file=sys.stderr)
    for a in kept:
        print("  KEEP", a.get("assay_chembl_id"),
              (a.get("description") or "")[:100], file=sys.stderr)

    calls = defaultdict(lambda: {"active": 0, "inactive": 0, "assays": set()})
    n_act_records = 0
    for a in kept:
        aid = a.get("assay_chembl_id")
        try:
            acts = fetch_activities(aid)
        except Exception as exc:  # noqa: BLE001 - log and continue
            print(f"  WARN {aid}: {exc}", file=sys.stderr)
            continue
        for act in acts:
            comment = (act.get("activity_comment") or "").strip()
            if comment not in ("Active", "Inactive"):
                continue
            smiles = (act.get("canonical_smiles") or "").strip()
            if not smiles:
                continue
            # Skip records flagged as suspect.
            dvc = act.get("data_validity_comment") or ""
            if "transcription error" in dvc.lower():
                continue
            n_act_records += 1
            slot = calls[smiles]
            if comment == "Active":
                slot["active"] += 1
            else:
                slot["inactive"] += 1
            slot["assays"].add(aid)
    print(f"labeled activity records: {n_act_records}", file=sys.stderr)

    rows = []
    n_conflict = 0
    for smiles, slot in calls.items():
        if slot["active"] > 0 and slot["inactive"] > 0:
            n_conflict += 1
            continue  # ambiguous: conflicting calls across assays
        label = 1 if slot["active"] > 0 else 0
        rows.append({
            "smiles": smiles,
            "label": label,
            "n_assays": len(slot["assays"]),
            "n_active_calls": slot["active"],
            "n_inactive_calls": slot["inactive"],
        })
    print(f"unique compounds after dedupe: {len(rows)}; "
          f"dropped for conflicting calls: {n_conflict}", file=sys.stderr)
    n_pos = sum(r["label"] for r in rows)
    print(f"positives: {n_pos} ({100*n_pos/max(len(rows),1):.1f}%), "
          f"negatives: {len(rows)-n_pos}", file=sys.stderr)

    import os
    os.makedirs("data", exist_ok=True)
    rows.sort(key=lambda r: (r["smiles"],))
    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["smiles", "label", "n_assays",
                            "n_active_calls", "n_inactive_calls"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
