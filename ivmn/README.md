# In Vitro Micronucleus (MNvit) Prediction — HF Embeddings + Few-Shot Learning

A binary classifier that predicts the in vitro micronucleus assay outcome
(`1` = MNvit positive, `0` = MNvit negative) from a chemical structure
(SMILES). Frozen Hugging Face molecular embeddings (ChemBERTa) +
a lightweight classifier — no transformer fine-tuning.

## What it is

- **Endpoint:** in vitro mammalian cell micronucleus test (OECD 487),
  binary call.
- **Embeddings:** `DeepChem/ChemBERTa-77M-MLM` (RoBERTa, 384-dim),
  mean-pooled last hidden state, frozen. The embedding model is NOT in
  this repo: it downloads from Hugging Face on first run and is cached
  locally (`~/.cache/huggingface`) for offline reuse afterwards.
- **Classifier:** logistic regression (`C=1.0`, `class_weight='balanced'`)
  on the frozen embeddings. kNN (k=5, cosine) is evaluated alongside for
  comparison. The saved classifier is 2.5 KB (coefficients only).

## Dataset

**Baderna et al. 2020**, "New in silico models to predict in vitro
micronucleus induction as marker of genotoxicity", *J. Hazard. Mater.*
385:121638 (doi:10.1016/j.jhazmat.2019.121638), Supplementary Table mmc2.

- **379 compounds** after dedupe (1 duplicate SMILES dropped):
  **227 positive (59.9%) / 152 negative (40.1%)**.
- Sources behind that set: OECD eChemPortal, ECVAM retrospective
  validation of the in vitro micronucleus test, SCCS/SCCP opinions, and
  literature (see the SI's Section S1).
- The authors' own train/test flag is preserved in the `author_split`
  column, but the protocol below uses its own stratified split so the
  few-shot evaluation is clean.

Why not the Toxics 2023 set (Khondkaryan et al., *Toxics* 2023, 11:785)?
That paper's SI Tables S1/S2 list names/SMILES/CAS but **no outcome
labels** ("Toxicity data is available from the corresponding author upon
reasonable request"), so it cannot supply a labeled training set. A
ChEMBL harvest was also attempted (`data/fetch_chembl_mnvit.py`) and
abandoned: ChEMBL MN assay records carry no outcome in any structured
field. Baderna mmc2 is the closest peer-reviewed, publicly downloadable,
labeled MNvit set.

## Install and run (fresh clone)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. Build the labeled dataset (downloads Baderna SI mmc2 from the publisher)
.venv/bin/python data/prepare_baderna.py

# 2. Train + few-shot evaluation (embeds via ChemBERTa, ~2 min on CPU
#    after the one-time HF model download)
.venv/bin/python train_fewshot.py

# 3. Predict
.venv/bin/python predict.py "CCO" "Nc1ccc2ccccc2c1"
.venv/bin/python predict.py --csv input.csv --smiles-col smiles --out predictions.csv
```

First run needs internet (Hugging Face model download, ~300 MB).
Behind a corporate proxy, `embed.py` strips bracketed IPv6 entries from
`NO_PROXY`/`no_proxy` before importing `transformers`, because
`huggingface_hub`'s HTTP client crashes on them (`Invalid port: ':1]'`).

## Results

Fixed stratified split (seed 42): train 265 / val 57 / test 57.
Few-shot: 5 stratified subsets per size, drawn from the train split,
evaluated on the fixed test set. Mean ± std over the 5 repeats.

### Few-shot test metrics (test n = 57)

| Train size | Classifier | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 16 | logreg | 0.54 ± 0.07 | 0.56 ± 0.08 | 0.61 ± 0.06 | 0.68 ± 0.08 | 0.64 ± 0.06 |
| 16 | kNN | 0.58 ± 0.04 | 0.53 ± 0.07 | 0.61 ± 0.03 | 0.82 ± 0.08 | 0.70 ± 0.04 |
| 32 | logreg | 0.60 ± 0.03 | 0.62 ± 0.04 | 0.65 ± 0.03 | 0.73 ± 0.07 | 0.68 ± 0.03 |
| 32 | kNN | 0.56 ± 0.02 | 0.57 ± 0.03 | 0.62 ± 0.03 | 0.68 ± 0.06 | 0.65 ± 0.02 |
| 64 | logreg | 0.57 ± 0.05 | 0.61 ± 0.05 | 0.62 ± 0.04 | 0.71 ± 0.05 | 0.66 ± 0.04 |
| 64 | kNN | 0.57 ± 0.04 | 0.59 ± 0.05 | 0.62 ± 0.03 | 0.72 ± 0.04 | 0.67 ± 0.03 |
| 128 | logreg | 0.58 ± 0.03 | 0.62 ± 0.04 | 0.64 ± 0.03 | 0.68 ± 0.05 | 0.66 ± 0.03 |
| 128 | kNN | 0.60 ± 0.06 | 0.64 ± 0.05 | 0.64 ± 0.04 | 0.76 ± 0.05 | 0.70 ± 0.04 |

### Final model (logreg trained on train+val, 322 compounds; test n = 57)

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **logreg (shipped)** | **0.63** | **0.68** | **0.69** | **0.71** | **0.70** |
| kNN (comparison) | 0.61 | 0.62 | 0.64 | 0.79 | 0.71 |

Confusion matrix (logreg): [[TN=12, FP=11], [FN=10, TP=24]].

## Interpretability (MCTS)

`interpret_mcts.py` finds a **rationale**: a minimal substructure of
an input molecule that preserves the model's prediction — the
fragment the prediction "rests on". It is a deletion-based Monte
Carlo Tree Search:

- **State:** a SMILES string, starting from the full input molecule.
- **Action:** delete one heavy atom and keep the largest connected
  fragment (RDKit). Empty/invalid fragments are skipped; fragments
  whose aromaticity breaks on deletion are repaired via Kekulization.
- **Reward:** the model's predicted probability of the target class
  for the fragment, minus a small size penalty (`--size-penalty`,
  default 0.05), so smaller fragments that keep the prediction win.
- **Search:** MCTS with UCB1 selection (`--simulations`, default
  200). Nodes with at most `--max-atoms` heavy atoms (default 12)
  are terminal. Each new fragment costs one ChemBERTa forward pass,
  so a molecule finishes in ~1–2 min on CPU.
- **Reported rationale:** the smallest evaluated fragment that
  preserves the predicted class, verified to be a valid SMILES and a
  substructure of the input. If no fragment preserves the prediction,
  the script says so instead of returning a bogus rationale.

```bash
.venv/bin/python interpret_mcts.py "Nc1ccc2ccccc2c1"
.venv/bin/python interpret_mcts.py "CCO" --target 0 --simulations 100 \
    --out rationale.json
```

Output: input prediction + probability, the rationale SMILES with
its heavy-atom count and probability, and the top candidate
fragments by reward (with whether each keeps the prediction).
`--out` writes the same result as JSON. `--target {0,1}` overrides
which class to explain (default: the model's predicted class).

Verified examples (200 simulations each):

| Molecule | Model call | Rationale | Rationale P(pos) |
|---|---|---|---|
| Toxaphene (label 1) | positive, 0.64 | `ClCC(Cl)C(Cl)C(Cl)Cl` (9 atoms) | 0.75 |
| Phenobarbital (label 0) | negative, 0.39 | `O=CNCNC=O` (7 atoms) | 0.29 |
| Ethanol | positive, 0.52 | `CCO` (whole molecule, 3 atoms) | 0.52 |

**Limitation — read this before citing a rationale.** A rationale is
a *model-explanation* device: it shows which part of the structure
the model relies on to keep its prediction. It is **not** proof of a
biological mechanism — it does not show that the fragment causes
micronuclei, and a fragment that preserves a prediction can still be
an artifact of a weak model (this one: test AUC 0.68). Use rationales
to sanity-check what the model learned, not as mechanistic evidence.

## Caveats — read before using

- **Modest performance.** Frozen ChemBERTa embeddings + a linear head
  underperform the source paper's descriptor/fingerprint ensemble on
  this dataset (their ensemble reached higher AUC on their own test
  set). Treat this model as a **screening baseline / few-shot scaffold**,
  not a replacement for experimental MNvit or for a tuned production
  QSAR. The value here is the runnable HF-embedding + few-shot pipeline;
  with more labeled data, expect better results.
- **Small data, wide uncertainty.** Test n = 57, so every metric carries
  a wide confidence interval (±~0.07 AUC or more). One borderline
  compound flipping changes accuracy by ~2 points.
- **Labels mix conditions.** The Baderna labels pool different cell lines
  (CHO, V79, HPBL, TK6, …) and ±S9 metabolic activation into one binary
  call, and different sources may disagree on edge cases. Do not use
  this model to overrule experimental data.
- **Probabilities near 0.5 are low-confidence.** E.g. ethanol (CCO)
  scores ~0.52 — effectively a coin flip, not a real positive call.
- **Applicability domain.** 379 mostly industrial chemicals; pharma-like
  or novel chemotypes may be out of domain. Check similarity to the
  training set before trusting a prediction.
- **Not a regulatory replacement.** For ICH S2 / OECD 487 decisions,
  follow the established battery (Ames + MNvit/CA + in vivo follow-up).

## Files

| File | What it is |
|---|---|
| `embed.py` | HF embedding extraction (ChemBERTa, mean-pooled) |
| `train_fewshot.py` | Few-shot evaluation + final model training; writes `metrics.json` |
| `predict.py` | CLI / CSV prediction with the saved classifier |
| `interpret_mcts.py` | MCTS rationale extraction (minimal class-preserving substructure) |
| `model_ivmn.joblib` | Saved logreg classifier (2.5 KB) |
| `metrics.json` | Few-shot + final test metrics (machine-readable) |
| `requirements.txt` | Pinned deps (torch CPU, transformers, scikit-learn, rdkit, …) |
| `data/prepare_baderna.py` | Downloads Baderna SI mmc2 → `data/mnvit_baderna.csv` |
| `data/mnvit_baderna.csv` | Labeled dataset (379 compounds) |
| `data/splits.csv` | Train/val/test assignment (seed 42) |
| `data/embeddings.npz` | Cached ChemBERTa embeddings (379 × 384, regenerable via `embed.py`) |
| `data/fetch_chembl_mnvit.py` | Abandoned ChEMBL harvest (documents why it yields no labels) |
