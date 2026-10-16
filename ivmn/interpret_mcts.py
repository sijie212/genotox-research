#!/usr/bin/env python3
"""MCTS rationale extraction for the IVMN model.

For a given input molecule (SMILES), searches for a MINIMAL substructure
(rationale) that preserves the IVMN model's prediction -- i.e. the
fragment that "drives" the prediction. This is the standard
deletion-based MCTS rationale approach.

How it works
------------
* State: a SMILES string, starting from the full input molecule.
* Action: delete one heavy atom (with its bonds) and keep the largest
  connected fragment (RDKit). Actions yielding an empty/invalid
  fragment are skipped.
* Reward of a fragment: the model's predicted probability of the
  target class, minus a small size penalty so the search prefers
  smaller fragments that still keep the prediction.
* Search: Monte Carlo Tree Search with UCB1 selection. Nodes with at
  most --max-atoms heavy atoms are terminal (not expanded further).
* The reported rationale is the smallest evaluated fragment that
  preserves the predicted class (probability on the target-class side
  of 0.5). It is verified to be a valid SMILES and a substructure of
  the input molecule before it is reported.

The embedding + model-loading code is reused from the existing
modules (embed.py, model_ivmn.joblib) -- nothing about the trained
model is reimplemented here.

Usage:
    python interpret_mcts.py "Nc1ccc2ccccc2c1" [--simulations N] \
        [--max-atoms K] [--target {0,1}] [--out out.json]
"""
import argparse
import json
import math
import os
import sys
import time

import joblib
import numpy as np
from rdkit import Chem
from rdkit import RDLogger

from embed import embed_smiles, load_embedding_model

RDLogger.DisableLog("rdApp.*")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model_ivmn.joblib")


# --------------------------------------------------------------------------
# Model scoring (reuses embed.py + the saved classifier)
# --------------------------------------------------------------------------
class Scorer:
    """Caches model probabilities per fragment SMILES."""

    def __init__(self, model_path=MODEL_PATH):
        bundle = joblib.load(model_path)
        self.clf = bundle["model"]
        self.tokenizer, self.model = load_embedding_model(
            bundle["embedding_model"])
        self._cache = {}

    def prob_positive(self, smiles):
        """P(MNvit positive) for one fragment SMILES (cached)."""
        if smiles not in self._cache:
            emb = embed_smiles([smiles], self.tokenizer, self.model)
            self._cache[smiles] = float(
                self.clf.predict_proba(emb)[0, 1])
        return self._cache[smiles]

    @property
    def n_evaluations(self):
        return len(self._cache)


# --------------------------------------------------------------------------
# Fragmentation
# --------------------------------------------------------------------------
def heavy_atom_count(mol):
    return mol.GetNumHeavyAtoms()


def _sanitize_fragment(frag):
    """Sanitize one fragment; repair broken aromaticity via Kekulization.

    Deleting an atom out of an aromatic ring leaves a fragment whose
    aromatic flags no longer describe a ring, and plain SanitizeMol
    raises AtomKekulizeException. Kekulize(clearAromaticFlags=True)
    rewrites those bonds in Kekule form, which substructure matching
    still recognizes as equivalent.
    """
    try:
        Chem.SanitizeMol(frag)
        return frag
    except Exception:
        pass
    try:
        Chem.Kekulize(frag, clearAromaticFlags=True)
        Chem.SanitizeMol(frag)
        return frag
    except Exception:
        return None


def delete_one_atom_children(smiles):
    """All distinct fragments from deleting one heavy atom.

    For each heavy atom: remove it, keep the largest connected
    fragment, canonicalize. Returns a sorted list of canonical SMILES
    (deterministic order). Skips empty / invalid / unchanged results.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    children = set()
    for idx in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 1:
            continue  # heavy atoms only
        rw = Chem.RWMol(mol)
        rw.RemoveAtom(idx)
        try:
            frags = Chem.GetMolFrags(rw, asMols=True, sanitizeFrags=False)
        except Exception:
            continue
        best = None
        for frag in frags:
            frag = _sanitize_fragment(frag)
            if frag is None or frag.GetNumHeavyAtoms() == 0:
                continue
            if (best is None
                    or frag.GetNumHeavyAtoms() > best.GetNumHeavyAtoms()):
                best = frag
        if best is None:
            continue
        child = Chem.MolToSmiles(best)
        if child and child != smiles:
            children.add(child)
    return sorted(children)


# --------------------------------------------------------------------------
# MCTS
# --------------------------------------------------------------------------
class Node:
    __slots__ = ("smiles", "parent", "children", "untried", "visits",
                 "total_reward", "n_heavy")

    def __init__(self, smiles, parent=None, n_heavy=None):
        self.smiles = smiles
        self.parent = parent
        self.children = []
        self.untried = None  # lazily filled with delete_one_atom_children()
        self.visits = 0
        self.total_reward = 0.0
        self.n_heavy = n_heavy

    @property
    def mean_reward(self):
        return self.total_reward / self.visits if self.visits else 0.0


def ucb1(node, parent_visits, c):
    if node.visits == 0:
        return float("inf")
    return (node.mean_reward
            + c * math.sqrt(math.log(parent_visits) / node.visits))


def run_mcts(scorer, root_smiles, target, simulations=200, max_atoms=12,
             size_penalty=0.05, ucb_c=math.sqrt(2), time_limit=120.0,
             seed=42):
    """Run MCTS from root_smiles. Returns (root, evaluated, n_simulations).

    evaluated: dict smiles -> {"p_positive": float, "n_heavy": int,
    "reward": float}. Reward = P(target class) - size_penalty *
    (n_heavy / n_heavy_full).
    """
    del seed  # selection/expansion are deterministic (sorted order)
    root_mol = Chem.MolFromSmiles(root_smiles)
    n_full = root_mol.GetNumHeavyAtoms()
    evaluated = {}

    def reward_of(smiles):
        if smiles not in evaluated:
            p_pos = scorer.prob_positive(smiles)
            mol = Chem.MolFromSmiles(smiles)
            n_heavy = mol.GetNumHeavyAtoms()
            p_target = p_pos if target == 1 else 1.0 - p_pos
            evaluated[smiles] = {
                "p_positive": p_pos,
                "n_heavy": n_heavy,
                "reward": p_target - size_penalty * (n_heavy / n_full),
            }
        return evaluated[smiles]["reward"]

    root = Node(root_smiles, n_heavy=n_full)
    reward_of(root_smiles)
    root.visits = 1
    root.total_reward = evaluated[root_smiles]["reward"]

    t_start = time.time()
    n_sim = 0
    for _ in range(simulations):
        if time.time() - t_start > time_limit:
            break
        # ---- Selection / expansion ----
        node = root
        path = [node]
        while True:
            if node.n_heavy <= max_atoms:
                break  # terminal: small enough, do not expand further
            if node.untried is None:
                node.untried = delete_one_atom_children(node.smiles)
            if node.untried:
                child_smiles = node.untried.pop(0)  # sorted order
                child = Node(child_smiles, parent=node,
                             n_heavy=Chem.MolFromSmiles(
                                 child_smiles).GetNumHeavyAtoms())
                node.children.append(child)
                node = child
                path.append(node)
                break
            if not node.children:
                break  # terminal: no valid deletions
            node = max(node.children,
                       key=lambda ch: (ucb1(ch, node.visits, ucb_c),
                                       ch.smiles))
            path.append(node)
        # ---- Evaluation ----
        r = reward_of(node.smiles)
        # ---- Backpropagation ----
        for anc in path:
            anc.visits += 1
            anc.total_reward += r
        n_sim += 1
    return root, evaluated, n_sim


def preserves_class(p_positive, target):
    """True if the predicted class for the fragment equals target."""
    return (p_positive > 0.5) if target == 1 else (p_positive < 0.5)


def pick_rationale(evaluated, input_mol, target):
    """Smallest class-preserving fragment (tie-break: highest P(target)).

    Verifies validity + substructure match before accepting. Returns
    the rationale dict or None if no fragment preserves the prediction.
    """
    def sort_key(item):
        smiles, info = item
        p_target = info["p_positive"] if target == 1 \
            else 1.0 - info["p_positive"]
        return (info["n_heavy"], -p_target, smiles)

    for smiles, info in sorted(evaluated.items(), key=sort_key):
        if not preserves_class(info["p_positive"], target):
            continue
        frag = Chem.MolFromSmiles(smiles)
        if frag is None:
            continue
        if not input_mol.HasSubstructMatch(frag):
            continue
        return {
            "smiles": smiles,
            "n_heavy_atoms": info["n_heavy"],
            "probability_positive": round(info["p_positive"], 4),
            "reward": round(info["reward"], 4),
        }
    return None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="MCTS rationale extraction for the IVMN model")
    ap.add_argument("smiles", help="Input molecule SMILES")
    ap.add_argument("--simulations", type=int, default=200,
                    help="MCTS simulations (default 200)")
    ap.add_argument("--max-atoms", type=int, default=12,
                    help="Stop expanding fragments with at most this many "
                         "heavy atoms (default 12)")
    ap.add_argument("--target", type=int, choices=[0, 1], default=None,
                    help="Target class to explain (default: model's "
                         "predicted class for the full molecule)")
    ap.add_argument("--size-penalty", type=float, default=0.05,
                    help="Size penalty weight in the reward (default 0.05)")
    ap.add_argument("--ucb-c", type=float, default=math.sqrt(2),
                    help="UCB exploration constant (default sqrt(2))")
    ap.add_argument("--time-limit", type=float, default=120.0,
                    help="Wall-clock cap in seconds (default 120)")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Top candidate fragments to report (default 5)")
    ap.add_argument("--out", help="Write machine-readable JSON here")
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    input_mol = Chem.MolFromSmiles(args.smiles)
    if input_mol is None:
        print(f"error: invalid SMILES: {args.smiles}", file=sys.stderr)
        sys.exit(1)
    if input_mol.GetNumHeavyAtoms() == 0:
        print("error: molecule has no heavy atoms", file=sys.stderr)
        sys.exit(1)

    scorer = Scorer(args.model)
    p_full = scorer.prob_positive(args.smiles)
    pred_class = 1 if p_full > 0.5 else 0
    target = args.target if args.target is not None else pred_class

    _, evaluated, n_sim = run_mcts(
        scorer, args.smiles, target,
        simulations=args.simulations, max_atoms=args.max_atoms,
        size_penalty=args.size_penalty, ucb_c=args.ucb_c,
        time_limit=args.time_limit)

    rationale = pick_rationale(evaluated, input_mol, target)

    # Top candidates by reward (class-preserving first, then rest).
    def cand_key(item):
        smiles, info = item
        return (-info["reward"], info["n_heavy"], smiles)

    top = []
    for smiles, info in sorted(evaluated.items(), key=cand_key)[:args.top_k]:
        top.append({
            "smiles": smiles,
            "n_heavy_atoms": info["n_heavy"],
            "probability_positive": round(info["p_positive"], 4),
            "reward": round(info["reward"], 4),
            "preserves_class": preserves_class(info["p_positive"], target),
        })

    result = {
        "input_smiles": args.smiles,
        "input_n_heavy_atoms": input_mol.GetNumHeavyAtoms(),
        "predicted_class": pred_class,
        "predicted_probability_positive": round(p_full, 4),
        "target_class": target,
        "rationale": rationale,
        "top_candidates": top,
        "n_simulations": n_sim,
        "n_evaluated_fragments": len(evaluated),
        "failure_reason": (None if rationale else
                           "no evaluated fragment preserved the "
                           "predicted class"),
    }

    # ---- Human-readable summary ----
    print(f"Input SMILES: {args.smiles}")
    print(f"Model prediction: class {pred_class} "
          f"({'MNvit positive' if pred_class == 1 else 'MNvit negative'}), "
          f"P(positive) = {p_full:.4f}")
    print(f"Target class explained: {target}")
    print(f"MCTS: {n_sim} simulations, "
          f"{len(evaluated)} unique fragments evaluated")
    if rationale is None:
        print("\nNo rationale found: no evaluated fragment preserved the "
              "predicted class.")
    else:
        print(f"\nRationale (minimal class-preserving substructure):")
        print(f"  SMILES: {rationale['smiles']}")
        print(f"  heavy atoms: {rationale['n_heavy_atoms']} "
              f"(input: {input_mol.GetNumHeavyAtoms()})")
        print(f"  P(positive) = {rationale['probability_positive']:.4f} "
              f"-> class "
              f"{1 if rationale['probability_positive'] > 0.5 else 0}")
        print(f"  reward: {rationale['reward']:.4f}")
    print(f"\nTop {len(top)} candidate fragments (by reward):")
    for i, c in enumerate(top, 1):
        keep = "keeps prediction" if c["preserves_class"] \
            else "does NOT keep prediction"
        print(f"  {i}. {c['smiles']}  n_heavy={c['n_heavy_atoms']}  "
              f"P(pos)={c['probability_positive']:.4f}  "
              f"reward={c['reward']:.4f}  [{keep}]")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
