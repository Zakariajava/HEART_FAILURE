===================================================================
# Heart Disease Prediction - Tabular MLP Baseline
===================================================================

Overview
--------
This repo contains an end-to-end baseline for predicting heart disease from tabular clinical data (`heart.csv`).  
The goal is to build a transparent, reproducible pipeline before trying more complex models.

What I did
----------
- Performed a quick data audit (shapes, dtypes, missing values, distributions).
- Encoded categorical features with `OneHotEncoder` and mapped Sex to a binary feature.
- Applied custom preprocessing with `ColumnTransformer`:
  - Winsorization & value clipping for plausible ranges.
  - Zero-as-median imputation for invalid cholesterol values.
  - Robust scaling and `log1p` where appropriate.
- Converted the processed features to PyTorch tensors and built a `TabularDataset` + `DataLoader`.
- Implemented a Multi-Layer Perceptron with BatchNorm, GELU, Dropout and `BCEWithLogitsLoss`.
- Trained with AdamW, mixed precision (when available), validation monitoring and early stopping on AUC.
- Evaluated on a held-out test set with ROC-AUC, PR-AUC, accuracy, F1 and confusion matrix.

Results (Test)
--------------
- ROC-AUC ≈ 0.91
- AP ≈ 0.91
- Accuracy ≈ 0.89
- F1-score ≈ 0.90