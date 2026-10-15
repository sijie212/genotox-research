# Ames Mutagenicity Prediction Model

A trained binary classifier that predicts bacterial reverse mutation (Ames)
mutagenicity from a chemical structure (SMILES). Fingerprint baseline — no deep
learning. Public data only.

## What it is

- **Endpoint:** Ames mutagenicity — `1` = mutagenic, `0` = non-mutagenic.
- **Model:** scikit-learn `RandomForestClassifier` (250 trees), selected over
  `HistGradientBoostingClassifier` by validation ROC-AUC.
- **Features:** RDKit Morgan fingerprints, radius 2, 2048 bits.
- **Training time:** ~2 minutes on CPU.

## Dataset

Source: **AmesFormer clean open Ames set**
(`Combined_2s_as_0s_publication.csv`), downloaded from the public mirror at
<https://github.com/luke-a-thompson/AmesFormer/blob/main/data_cleaning/Combined_2s_as_0s_publication.csv>.
It aggregates Hansen 2009, ISSSTY, and EURL Ames data, deduplicated by the
AmesFormer authors (proprietary Honma data excluded).

- **Rows:** 10,377 total; 10,375 usable (2 SMILES failed RDKit parsing and
  were dropped).
- **Class balance:** 4,691 mutagenic (45.2%) / 5,684 non-mutagenic (54.8%).
- **Split:** stratified 80/10/10 train/validation/test, seed 42
  (train 8,300 / val 1,037 / test 1,038). The winner was chosen on validation,
  refit on train+validation, and evaluated once on the held-out test set.
  (The source file's own `split` column was not used.)

## Test-set performance (held-out, n = 1,038)

| Metric | Value |
|---|---|
| Accuracy | 0.8545 |
| ROC-AUC | 0.9136 |
| Precision | 0.8664 |
| Recall | 0.8017 |
| F1 | 0.8328 |

Confusion matrix `[[TN, FP], [FN, TP]]` = `[[511, 58], [93, 376]]`.

## Usage

### Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Retrain

```bash
python train.py                # downloads data if data_combined.csv is missing
python train.py --data data_combined.csv --out model.joblib
```

This re-runs the full pipeline (fingerprinting → split → model selection →
refit → evaluation) and overwrites `model.joblib` and `metrics.json`.

### Predict

```bash
# One or more SMILES on the command line
python predict.py "CCO" "Nc1ccc2ccccc2c1"

# Batch from a CSV (must contain a SMILES column, default name: smiles)
python predict.py --csv input.csv --smiles-col smiles --out predictions.csv
```

Output columns: `smiles`, `prediction` (0/1), `probability_mutagenic`, `label`.
Invalid SMILES are reported as `INVALID SMILES` rather than crashing.

## Files

| File | Description |
|---|---|
| `train.py` | Reproducible training script (download/load → fingerprints → split → model selection → evaluate → save). |
| `predict.py` | Loads `model.joblib`; predicts for CLI SMILES or a CSV. |
| `model.joblib` | Trained model + metadata (model type, fingerprint params, test metrics). ~98 MB. |
| `data_combined.csv` | The training dataset as downloaded (AmesFormer mirror). |
| `metrics.json` | Test-set metrics from the last training run. |
| `requirements.txt` | Pinned Python dependencies. |

## Caveats

- **Label noise:** inter-laboratory Ames reproducibility error is ~15% (Kazius
  2005), so accuracy near 85% is effectively at the experimental ceiling.
  Do not expect a much better model from these labels alone.
- **Applicability domain:** the training data are dominated by industrial
  chemicals; pharma-like / novel chemotypes may be outside the model's domain.
  Treat predictions on such compounds as screening-level, not definitive.
- **Not a regulatory replacement:** this is a research/screening baseline, not
  a substitute for experimental Ames testing or a validated QSAR for
  regulatory submission.
- **S9 / strain resolution:** labels are overall Ames calls; the model does
  not distinguish ±S9 metabolic activation or individual tester strains.
- **Model file size:** `model.joblib` is ~98 MB (just under GitHub's 100 MB
  single-file limit). Use Git LFS or a release asset if you need to version
  it more comfortably.
