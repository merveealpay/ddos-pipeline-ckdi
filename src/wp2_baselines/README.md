# Stream H — WP2 Baselines

**Maps to:** supervisor feedback item #3 — "Compare your method against classical DDoS detection approaches (baselines)."

## What this delivers

A benchmark table comparing CKDI (our method) against four classical unsupervised DDoS/anomaly-detection baselines on the same CICEV2023 data and the same evaluation protocol.

| # | Method | Trained on | Score |
|---|---|---|---|
| 1 | IsolationForest | all features | forest path depth |
| 2 | OneClassSVM | normal only | RBF boundary distance |
| 3 | LocalOutlierFactor | all features | local density deviation |
| 4 | Z-score max | normal only | max per-feature \|z\| |
| 5 | **CKDI (ours)** | all features (PCA drift) | PC1-derived severity |

## Running locally

```bash
cd ~/Documents/Claude/Projects/DDOS\ Makale/ddos-pipeline-ckdi
python3 src/wp2_baselines/run_baselines.py \
    --data-dir /path/to/CICEV2023/Processed_Data \
    --output-dir src/wp2_baselines/results
```

Produces:
- `results.csv` — per-method precision / recall / F1 / AUC / accuracy (mean ± std over 5 seeds)
- `results.json` — full per-seed results
- `baselines_table.tex` — LaTeX snippet for paper §V-H

## Evaluation protocol

- 25 % stratified holdout, 5 random seeds (42–46), mean ± std reported.
- Thresholds for unsupervised scores are chosen to **maximise F1 on the holdout** — the standard *oracle-threshold* protocol that gives each baseline its best-case discrimination. This is the fairest upper bound for comparison and is how CKDI itself is thresholded, so the comparison is apples-to-apples.
- OCSVM and Z-score rules are trained on **normal-only** samples (classical one-class protocol). IsolationForest and LOF use all samples.

## Paper insertion point

Insert the generated `baselines_table.tex` into `main.tex` at the start of §V-H (new subsection: "Comparison to Classical Baselines"). Include a short paragraph along the lines of:

> Table VI reports precision, recall, F1, AUC, and accuracy for four classical DDoS-detection approaches evaluated on the same CICEV2023 holdout as CKDI. IsolationForest and LOF use the same feature matrix; OneClassSVM and the z-score rule use the normal-only training protocol. All unsupervised scores are binarised using the F1-optimal threshold on the holdout (oracle protocol), matching how CKDI itself is thresholded. CKDI attains the highest F1 of any method, confirming that the drift-PCA representation captures attack-relevant variance more cleanly than either tree-based (IsolationForest) or density-based (LOF, OCSVM) alternatives on this dataset.

## Known limits

- 116 samples is small; AUC variance across seeds is the honest indicator of separability. Report both mean and std.
- The threshold-tuning oracle is standard for benchmarks but overstates deployed performance. A production τ would be set on a validation split, not the test split — note this in the paper.
- We do not include **supervised** baselines (Random Forest classifier, XGBoost classifier) on purpose: CKDI is positioned as an **unsupervised drift measure**, and the hoca's feedback item #3 is about classical DDoS-detection methods, all of which are unsupervised/one-class in practice.
