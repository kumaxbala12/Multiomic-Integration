# Multi-Omics Integration Pipeline (Transcriptomics + Methylation)

Integrate two omics layers (e.g., RNA expression + DNA methylation) to find joint signals and build a classifier.

## Expected input files (put in `data/raw/`)
- `expression.csv` — rows=samples; columns: `sample_id`, gene_1, gene_2, ...
- `methylation.csv` — rows=samples; columns: `sample_id`, cpg_1, cpg_2, ...
- `labels.csv` — columns: `sample_id`, `label` (case/control or disease subtype)

> You can rename columns in the notebook or pass arguments to scripts.

## What’s included
- **Notebook**: `notebooks/01_integration_and_baseline.ipynb`
  - EDA for each omics
  - Preprocessing (log1p, scaling)
  - CCA-based integration to obtain joint components
  - Baseline classifier (LogReg / RF) using integrated features
  - Plots + saved artifacts
- **Scripts**:
  - `scripts/preprocess_twoomics.py` — align samples, transform, scale; save processed matrices
  - `scripts/integrate_cca.py` — run CCA to get joint components; save components + loadings
  - `scripts/train_classifier.py` — train classifier on components; save model + metrics
- **Results**: figures, models, and processed data

## Quickstart
```bash
pip install -r requirements.txt

# 1) Preprocess
python scripts/preprocess_twoomics.py   --expr data/raw/expression.csv   --meth data/raw/methylation.csv   --labels data/raw/labels.csv   --out data/processed

# 2) Integrate (CCA with k components)
python scripts/integrate_cca.py   --X1 data/processed/X_expr.csv   --X2 data/processed/X_meth.csv   --k 20   --out data/processed

# 3) Train classifier on integrated components
python scripts/train_classifier.py   --components data/processed/CCA_components.csv   --labels data/processed/labels_aligned.csv   --out results/models
```

## Notes
- CCA is from scikit-learn; good for linear shared structure. Swap for PLS if you prefer.
- For large omics, consider preliminary feature filtering (variance, MAD).
- You can try downstream enrichment by ranking genes from CCA loadings (not implemented here).
