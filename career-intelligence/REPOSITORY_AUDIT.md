# Repository and modelling audit — 2026-11-15

## Scope reviewed

Main Python scripts: `svr_test_rfe.py`, `feature_Extraction_pca.py`, `SVR linear test.py`, `LSTM basketball data.py`; archived `output_tests.txt`; `README.md`; dataset inventory and representative player tables; original rookies/careers training sources. The repository contains about 13,960 tracked tree entries, primarily archived game CSVs; this audit does **not** claim each archival file or PDF was manually read.

## Scientific issues

1. `feature_Extraction_pca.py` contains a scaler fit on the full dataset before splitting, a PCA fit on unscaled features followed by transformations of scaled features, and attempted classification on encoded continuous regression outcomes.
2. `SVR linear test.py` scales before splitting, converts target values using `LabelEncoder`, calculates `sqrt(-MAE)` while labelling it MAE, and uses deprecated `sklearn.cross_validation` and `time.clock()`.
3. `svr_test_rfe.py` applies feature ranking on all available observations before cross-validating augmented subsets. This leaks selection information into folds and runs `eval()` on model strings.
4. Archived `output_tests.txt` shows near-perfect in-period PER regression (e.g., decision tree R² near 1.0), but these are not forward-time career predictions; PER is derived from same-season stats and may be reconstructed mechanically.
5. LSTM examples appear to adapt generic time-series code; temporal grouping, player boundaries, normalization and evaluation must be redesigned before using it to claim multi-season performance forecasting.
6. Archived rookie/career tables only extend to 2016/2018, and late-cohort career lengths are right-censored. Naively labeling 2016 rookies as short careers would introduce severe bias.
7. Identifiers are inconsistent across datasets (NBA IDs, names and sports database player IDs); name-only joins are ambiguous. This prototype excludes duplicate matches but production requires canonical IDs.

## What changed

- Preserved original thesis experiments.
- Created an independent, reproducible `career-intelligence` application and `ml` research module.
- Fitted temporal L2 logistic baseline on 13 rookie features; 988 training, 120 validation, 172 test.
- Validated against prevalence-only baseline using ROC-AUC, Brier and log loss; no future cohorts were used for labels.
- Exposed model inputs, version and limitations; risk-banded empirical career spans are explicitly not forecasts of individual yearly statistics.
- Implemented an independent NBA live scoreboard adapter, with honest fallback/error states.

## Next scientific priorities

Replace stale archive and name joins; build panel dataset; split calendar years and players; evaluate survival censored outcomes; add next-season production regression and uncertainty; monitor drift and subgroup calibration. Do not claim improvement on current 2026 players until verified with recent data.
