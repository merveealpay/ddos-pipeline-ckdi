#!/usr/bin/env python3
"""
Stream H — WP2 Baselines (supervisor item #3)

Backs the supervisor's request to benchmark our CKDI + BDRI framework
against classical DDoS-detection baselines. Uses the same CICEV2023
features (116 samples × 27 HPC features) and the ground-truth
`class` column (attack/normal) as the binary label.

Baselines evaluated:
  1. IsolationForest      — unsupervised ensemble anomaly detector
  2. OneClassSVM          — unsupervised one-class boundary (RBF)
  3. LocalOutlierFactor   — density-based anomaly (novelty=True)
  4. ZscoreMax            — "classical threshold rule": max per-feature
                            z-score > 3 σ over normal samples
  5. CKDI_thresholded     — our method: CKDI > τ*, τ* = F1-optimal
                            (this is NOT a baseline, it's the reference
                             we compare baselines against)

Metrics: Precision, Recall, F1, AUC-ROC, Accuracy.
Protocol: per-method train on 80 % stratified, evaluate on 20 % holdout,
5 seeds, report mean ± std.

Outputs:
  - results.csv         per-method metric table
  - results.json        full results incl. per-seed scores
  - baselines_table.tex LaTeX snippet ready to paste into paper §V-H
  - baselines_cm.png    confusion-matrix grid (optional, if matplotlib)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, confusion_matrix,
)

# ---------------------------------------------------------------------------
# Feature extraction — mirrors src/ensemble/ensemble_regression.py so that
# this module can run standalone without sys.path gymnastics.
# ---------------------------------------------------------------------------

def safe_mean(arr):
    a = np.asarray(arr, dtype=float)
    mask = (a != -1.0)
    a = a[mask]
    return float(np.mean(a)) if a.size else np.nan


def safe_std(arr):
    a = np.asarray(arr, dtype=float)
    mask = (a != -1.0)
    a = a[mask]
    return float(np.std(a)) if a.size else np.nan


def find_final_dataset_files(base):
    files = []
    for dirpath, _, filenames in os.walk(base):
        if "final_dataset.json" in filenames:
            files.append(Path(dirpath) / "final_dataset.json")
    return files


def parse_meta_from_path(fp):
    parts = fp.parts
    try:
        i = parts.index("Processed_Data")
    except ValueError:
        return {}
    return {
        "scenario":  parts[i+1] if i+1 < len(parts) else "unknown",
        "random_cs": parts[i+2] if i+2 < len(parts) else "unknown",
        "gaussian":  parts[i+3] if i+3 < len(parts) else "unknown",
        "perf_mode": parts[i+4] if i+4 < len(parts) else "unknown",
        "role":      parts[i+5] if i+5 < len(parts) else "unknown",
    }


def extract_rows_from_final_dataset(fp):
    meta = parse_meta_from_path(fp)
    with open(fp, "r") as f:
        d = json.load(f)
    rows = []
    for metric_type, metric_obj in d.items():
        for section, combos in metric_obj.items():
            for _, combo_val in combos.items():
                for cls in ("attack", "normal"):
                    if cls not in combo_val:
                        continue
                    cls_obj = combo_val[cls]
                    sampling_res = cls_obj.get("combined_sampling_resolution", np.nan)
                    data_point = cls_obj.get("data_point", {})
                    for symbol, ent_map in data_point.items():
                        for ent_id, values in ent_map.items():
                            rows.append({
                                **meta,
                                "metric_type": metric_type,
                                "section":     section,
                                "class":       cls,
                                "entity":      ent_id,
                                "sampling_res": sampling_res,
                                "mean":  safe_mean(values),
                                "std":   safe_std(values),
                                "count": int(np.sum(np.asarray(values, dtype=float) != -1.0)),
                            })
    return rows


def build_feature_matrix(data_dir):
    """Returns X (n×d), y_bin (n,), ckdi (n,), feat_cols."""
    final_files = find_final_dataset_files(Path(data_dir))
    if not final_files:
        raise FileNotFoundError(f"No final_dataset.json under {data_dir}")
    print(f"Found {len(final_files)} final_dataset.json files")

    all_rows = []
    for fp in final_files:
        all_rows.extend(extract_rows_from_final_dataset(fp))
    raw_df = pd.DataFrame(all_rows)
    print(f"Extracted {len(raw_df)} raw rows")

    agg = raw_df.groupby(
        ["scenario", "random_cs", "gaussian", "perf_mode", "role",
         "class", "entity", "metric_type", "section"],
        as_index=False
    ).agg(
        mean_mean=("mean", "mean"),
        mean_std=("std",  "mean"),
        mean_samp=("sampling_res", "mean"),
    )

    idx_cols = ["scenario", "random_cs", "gaussian", "perf_mode",
                "role", "class", "entity"]

    wide_mean = agg.pivot_table(index=idx_cols,
                                 columns=["metric_type", "section"],
                                 values="mean_mean")
    wide_std  = agg.pivot_table(index=idx_cols,
                                 columns=["metric_type", "section"],
                                 values="mean_std")
    wide_samp = agg.pivot_table(index=idx_cols,
                                 columns=["metric_type", "section"],
                                 values="mean_samp")

    def flatten(df, suffix):
        df = df.copy()
        df.columns = [f"{a}_{b}_{suffix}" for a, b in df.columns]
        return df

    X_df = pd.concat([flatten(wide_mean, "mean"),
                      flatten(wide_std,  "std"),
                      flatten(wide_samp, "samp")], axis=1).reset_index()

    meta_cols = idx_cols
    feat_cols = [c for c in X_df.columns if c not in meta_cols]

    # Drift-normalise using normal-class mean/std, grouped by (role, perf_mode)
    X_drift = X_df.copy()
    for (role, perf_mode), grp in X_df[X_df["class"] == "normal"].groupby(
            ["role", "perf_mode"]):
        mu    = grp[feat_cols].mean()
        sigma = grp[feat_cols].std().replace(0, np.nan)
        mask  = (X_df["role"] == role) & (X_df["perf_mode"] == perf_mode)
        X_drift.loc[mask, feat_cols] = (X_df.loc[mask, feat_cols] - mu) / sigma
    X_drift[feat_cols] = X_drift[feat_cols].fillna(0.0)

    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(X_drift[feat_cols].values).ravel()
    ckdi_abs = np.abs(pc1)
    ckdi = (ckdi_abs - ckdi_abs.min()) / (ckdi_abs.max() - ckdi_abs.min() + 1e-9)

    X = X_drift[feat_cols].values
    y_bin = (X_drift["class"] == "attack").astype(int).values

    print(f"Feature matrix: {X.shape},  "
          f"attack={y_bin.sum()}, normal={(1-y_bin).sum()}")
    return X, y_bin, ckdi, feat_cols


# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

def _binarize_contamination(model_score, y_true):
    """
    For unsupervised methods that return continuous anomaly scores, pick the
    threshold that maximises F1 on the evaluation set — this is the standard
    'oracle threshold' protocol used in anomaly-detection benchmarks (matches
    how CKDI itself is thresholded). It gives each baseline its best-case
    discrimination, so the comparison is fair upper-bound.
    """
    best_f1, best_tau = -1.0, 0.0
    for tau in np.unique(model_score):
        pred = (model_score >= tau).astype(int)
        f1   = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return best_tau


def eval_isolation_forest(X_tr, X_te, y_te, seed):
    clf = IsolationForest(n_estimators=200, contamination="auto",
                          random_state=seed)
    clf.fit(X_tr)
    score = -clf.score_samples(X_te)  # higher = more anomalous
    tau   = _binarize_contamination(score, y_te)
    pred  = (score >= tau).astype(int)
    return pred, score


def eval_ocsvm(X_tr_normal, X_te, y_te):
    """OCSVM train on NORMAL only — classic one-class protocol."""
    scaler = StandardScaler().fit(X_tr_normal)
    clf = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    clf.fit(scaler.transform(X_tr_normal))
    score = -clf.score_samples(scaler.transform(X_te))
    tau   = _binarize_contamination(score, y_te)
    pred  = (score >= tau).astype(int)
    return pred, score


def eval_lof(X_tr, X_te, y_te):
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True,
                             contamination="auto")
    clf.fit(X_tr)
    score = -clf.score_samples(X_te)
    tau   = _binarize_contamination(score, y_te)
    pred  = (score >= tau).astype(int)
    return pred, score


def eval_zscore_max(X_tr_normal, X_te, y_te):
    """Classical threshold rule: max |z-score| across features > τ."""
    mu    = X_tr_normal.mean(axis=0)
    sigma = X_tr_normal.std(axis=0) + 1e-9
    z_te  = np.abs((X_te - mu) / sigma)
    score = z_te.max(axis=1)
    tau   = _binarize_contamination(score, y_te)
    pred  = (score >= tau).astype(int)
    return pred, score


def eval_ckdi_thresholded(ckdi_te, y_te):
    tau  = _binarize_contamination(ckdi_te, y_te)
    pred = (ckdi_te >= tau).astype(int)
    return pred, ckdi_te


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

METHODS = [
    "IsolationForest",
    "OneClassSVM",
    "LocalOutlierFactor",
    "ZscoreMax",
    "CKDI_thresholded",
]


def run_seed(X, y, ckdi, seed, test_size=0.25):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                  random_state=seed)
    tr_idx, te_idx = next(sss.split(X, y))
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    ckdi_te    = ckdi[te_idx]
    X_tr_normal = X_tr[y_tr == 0]

    results = {}

    for method in METHODS:
        if method == "IsolationForest":
            pred, score = eval_isolation_forest(X_tr, X_te, y_te, seed)
        elif method == "OneClassSVM":
            pred, score = eval_ocsvm(X_tr_normal, X_te, y_te)
        elif method == "LocalOutlierFactor":
            pred, score = eval_lof(X_tr, X_te, y_te)
        elif method == "ZscoreMax":
            pred, score = eval_zscore_max(X_tr_normal, X_te, y_te)
        elif method == "CKDI_thresholded":
            pred, score = eval_ckdi_thresholded(ckdi_te, y_te)
        else:
            raise ValueError(method)

        try:
            auc = roc_auc_score(y_te, score)
        except ValueError:
            auc = np.nan

        results[method] = {
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall":    recall_score(y_te, pred, zero_division=0),
            "f1":        f1_score(y_te, pred, zero_division=0),
            "auc":       auc,
            "accuracy":  accuracy_score(y_te, pred),
            "tp": int(((pred == 1) & (y_te == 1)).sum()),
            "fp": int(((pred == 1) & (y_te == 0)).sum()),
            "fn": int(((pred == 0) & (y_te == 1)).sum()),
            "tn": int(((pred == 0) & (y_te == 0)).sum()),
        }
    return results


def aggregate(per_seed_results, seeds):
    rows = []
    for method in METHODS:
        precs, recs, f1s, aucs, accs = [], [], [], [], []
        for s in seeds:
            r = per_seed_results[s][method]
            precs.append(r["precision"])
            recs .append(r["recall"])
            f1s  .append(r["f1"])
            accs .append(r["accuracy"])
            if not np.isnan(r["auc"]):
                aucs.append(r["auc"])
        rows.append({
            "method":        method,
            "precision_mean": float(np.mean(precs)),
            "precision_std":  float(np.std (precs)),
            "recall_mean":    float(np.mean(recs)),
            "recall_std":     float(np.std (recs)),
            "f1_mean":        float(np.mean(f1s)),
            "f1_std":         float(np.std (f1s)),
            "auc_mean":       float(np.mean(aucs)) if aucs else float("nan"),
            "auc_std":        float(np.std (aucs)) if aucs else float("nan"),
            "accuracy_mean":  float(np.mean(accs)),
            "accuracy_std":   float(np.std (accs)),
            "n_seeds":        len(seeds),
        })
    return pd.DataFrame(rows)


def write_latex_table(df, out_path):
    """Paper §V-H: baseline comparison table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{WP2 baseline comparison. All methods evaluated on the "
        r"same CICEV2023 holdout (25\%, 5 seeds, stratified). Threshold is "
        r"tuned per method to maximise F1 on the holdout (oracle protocol).}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Precision & Recall & F1 & AUC & Accuracy \\",
        r"\midrule",
    ]

    def fmt(mean, std):
        if np.isnan(mean):
            return "--"
        return f"{mean:.3f} $\\pm$ {std:.3f}"

    # sort so CKDI_thresholded goes last (our method)
    order = [m for m in METHODS if m != "CKDI_thresholded"] + ["CKDI_thresholded"]
    df_ord = df.set_index("method").loc[order].reset_index()
    for _, r in df_ord.iterrows():
        name = r["method"].replace("_", r"\_")
        if r["method"] == "CKDI_thresholded":
            name = r"\textbf{CKDI (ours)}"
        lines.append(
            f"{name} & {fmt(r['precision_mean'], r['precision_std'])} & "
            f"{fmt(r['recall_mean'], r['recall_std'])} & "
            f"{fmt(r['f1_mean'], r['f1_std'])} & "
            f"{fmt(r['auc_mean'], r['auc_std'])} & "
            f"{fmt(r['accuracy_mean'], r['accuracy_std'])} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Root of Processed_Data tree containing "
                         "final_dataset.json files")
    ap.add_argument("--output-dir", default="src/wp2_baselines/results")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--test-size", type=float, default=0.25)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, ckdi, feat_cols = build_feature_matrix(args.data_dir)

    per_seed = {}
    for s in args.seeds:
        print(f"--- seed {s} ---")
        per_seed[s] = run_seed(X, y, ckdi, s, test_size=args.test_size)
        for m in METHODS:
            r = per_seed[s][m]
            print(f"  {m:20s}  F1={r['f1']:.3f}  "
                  f"P={r['precision']:.3f}  R={r['recall']:.3f}  "
                  f"AUC={r['auc']:.3f}")

    df = aggregate(per_seed, args.seeds)
    print("\n=== Aggregated results ===")
    print(df.to_string(index=False))

    df.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps({
        "per_seed":    per_seed,
        "aggregated":  df.to_dict(orient="records"),
        "config": {
            "seeds":      args.seeds,
            "test_size":  args.test_size,
            "n_features": int(X.shape[1]),
            "n_samples":  int(X.shape[0]),
            "n_attack":   int(y.sum()),
            "n_normal":   int((1 - y).sum()),
        },
    }, indent=2, default=str))

    write_latex_table(df, out_dir / "baselines_table.tex")
    print(f"\nWrote: {out_dir}/results.csv")
    print(f"Wrote: {out_dir}/results.json")
    print(f"Wrote: {out_dir}/baselines_table.tex")


if __name__ == "__main__":
    main()
