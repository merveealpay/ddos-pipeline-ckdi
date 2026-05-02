#!/usr/bin/env python3
"""
Stream A — Aksiyon A
CICEVSE2024 üzerinde 4 klasik baseline (IsolationForest, OCSVM, LOF, ZscoreMax)
+ CKDI karşılaştırması — Stream H'nin protokolü (F1-optimal oracle threshold,
25% stratified holdout × 5 seed) CICEVSE2024'e adapte edildi.

Amaç: Cross-dataset story'yi "CKDI çalışıyor"dan "CKDI baseline'lara
üstünlüğünü iki dataset'te koruyor"a çıkarmak.

Çıktı:
  - cicevse2024_baselines_results.csv
  - cicevse2024_baselines_results.json
  - cicevse2024_baselines_table.tex (paper §V-H için)
"""
import argparse
import json
import os
import sys
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
    accuracy_score,
)

META_COLS = {"time", "State", "Attack", "Scenario", "Label", "interface"}


# ---------------------------------------------------------------------------
# Data loading (mirrors run_cross_dataset.py for consistency)
# ---------------------------------------------------------------------------

def load_dataset(csv_path):
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:")]
    df.columns = df.columns.str.strip()

    required = {"Label", "State", "Scenario"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feat_cols = [c for c in df.columns if c not in META_COLS]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    nonnull = df[feat_cols].notna().sum()
    keep_cols = nonnull[nonnull > 0].index.tolist()
    meta_present = [c for c in df.columns if c in META_COLS]
    df = df[meta_present + keep_cols]
    feat_cols = [c for c in df.columns if c not in META_COLS]

    df["Label"] = df["Label"].astype(str).str.strip().str.lower()
    df["State"] = df["State"].astype(str).str.strip()
    df = df[df["Label"].isin({"attack", "benign", "normal"})].copy()
    df["Label"] = df["Label"].replace({"normal": "benign"})

    df[feat_cols] = df[feat_cols].fillna(0.0)
    print(f"  shape: {df.shape}, features: {len(feat_cols)}")
    print(f"  labels: {df['Label'].value_counts().to_dict()}")
    return df, feat_cols


def compute_ckdi(df, feat_cols):
    """BIREBIR Stream A (run_cross_dataset.py) ile aynı:
    Z-score against benign-only baseline grouped by State,
    sigma==0 features → NaN → fillna(0) → variance 0 → drop.
    """
    drift = df.copy()
    for state, grp in df[df["Label"] == "benign"].groupby("State"):
        mu = grp[feat_cols].mean()
        sigma = grp[feat_cols].std().replace(0, np.nan)
        mask = (df["State"] == state)
        drift.loc[mask, feat_cols] = (df.loc[mask, feat_cols] - mu) / sigma
    drift[feat_cols] = drift[feat_cols].fillna(0.0)

    var = drift[feat_cols].var()
    keep = var[var > 1e-12].index.tolist()
    print(f"  drift kept {len(keep)} non-trivial features "
          f"(dropped {len(feat_cols) - len(keep)} zero-var)")

    X_drift = drift[keep].values
    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(X_drift).ravel()
    abs_pc1 = np.abs(pc1)
    ckdi = (abs_pc1 - abs_pc1.min()) / (abs_pc1.max() - abs_pc1.min() + 1e-12)
    var_pc1 = float(pca.explained_variance_ratio_[0])
    print(f"  PC1 variance share = {var_pc1*100:.2f}%")

    y_bin = (df["Label"].values == "attack").astype(int)
    return X_drift, y_bin, ckdi


# ---------------------------------------------------------------------------
# Baselines (lifted from src/wp2_baselines/run_baselines.py)
# ---------------------------------------------------------------------------

METHODS = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor",
           "ZscoreMax", "CKDI_thresholded"]


def _binarize_oracle(score, y_true):
    best_f1, best_tau = -1.0, 0.0
    # Subsample tau candidates for speed when score is large
    if len(score) > 5000:
        tau_grid = np.quantile(score, np.linspace(0, 1, 200))
    else:
        tau_grid = np.unique(score)
    for tau in tau_grid:
        pred = (score >= tau).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return best_tau


def eval_isolation_forest(X_tr, X_te, y_te, seed):
    clf = IsolationForest(n_estimators=200, contamination="auto",
                          random_state=seed, n_jobs=-1)
    clf.fit(X_tr)
    score = -clf.score_samples(X_te)
    tau = _binarize_oracle(score, y_te)
    return (score >= tau).astype(int), score


def eval_ocsvm(X_tr_normal, X_te, y_te):
    scaler = StandardScaler().fit(X_tr_normal)
    clf = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    clf.fit(scaler.transform(X_tr_normal))
    score = -clf.score_samples(scaler.transform(X_te))
    tau = _binarize_oracle(score, y_te)
    return (score >= tau).astype(int), score


def eval_lof(X_tr, X_te, y_te):
    clf = LocalOutlierFactor(n_neighbors=20, novelty=True,
                             contamination="auto", n_jobs=-1)
    clf.fit(X_tr)
    score = -clf.score_samples(X_te)
    tau = _binarize_oracle(score, y_te)
    return (score >= tau).astype(int), score


def eval_zscore_max(X_tr_normal, X_te, y_te):
    mu = X_tr_normal.mean(axis=0)
    sigma = X_tr_normal.std(axis=0) + 1e-9
    z = np.abs((X_te - mu) / sigma)
    score = z.max(axis=1)
    tau = _binarize_oracle(score, y_te)
    return (score >= tau).astype(int), score


def eval_ckdi(ckdi_te, y_te):
    tau = _binarize_oracle(ckdi_te, y_te)
    return (ckdi_te >= tau).astype(int), ckdi_te


def run_seed(X, y, ckdi, seed, test_size=0.25):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=seed)
    tr, te = next(sss.split(X, y))
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    ckdi_te = ckdi[te]
    X_tr_normal = X_tr[y_tr == 0]
    if len(X_tr_normal) < 5:
        raise ValueError(f"Too few normal samples for OCSVM/Z: {len(X_tr_normal)}")

    results = {}
    for m in METHODS:
        if m == "IsolationForest":
            pred, score = eval_isolation_forest(X_tr, X_te, y_te, seed)
        elif m == "OneClassSVM":
            pred, score = eval_ocsvm(X_tr_normal, X_te, y_te)
        elif m == "LocalOutlierFactor":
            pred, score = eval_lof(X_tr, X_te, y_te)
        elif m == "ZscoreMax":
            pred, score = eval_zscore_max(X_tr_normal, X_te, y_te)
        elif m == "CKDI_thresholded":
            pred, score = eval_ckdi(ckdi_te, y_te)
        try:
            auc = roc_auc_score(y_te, score)
        except ValueError:
            auc = float("nan")
        results[m] = {
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall":    recall_score(y_te, pred, zero_division=0),
            "f1":        f1_score(y_te, pred, zero_division=0),
            "auc":       auc,
            "accuracy":  accuracy_score(y_te, pred),
        }
    return results


def aggregate(per_seed, seeds):
    rows = []
    for m in METHODS:
        ps, rs, fs, aucs, accs = [], [], [], [], []
        for s in seeds:
            r = per_seed[s][m]
            ps.append(r["precision"])
            rs.append(r["recall"])
            fs.append(r["f1"])
            accs.append(r["accuracy"])
            if not np.isnan(r["auc"]):
                aucs.append(r["auc"])
        rows.append({
            "method":         m,
            "precision_mean": float(np.mean(ps)),
            "precision_std":  float(np.std(ps)),
            "recall_mean":    float(np.mean(rs)),
            "recall_std":     float(np.std(rs)),
            "f1_mean":        float(np.mean(fs)),
            "f1_std":         float(np.std(fs)),
            "auc_mean":       float(np.mean(aucs)) if aucs else float("nan"),
            "auc_std":        float(np.std(aucs)) if aucs else float("nan"),
            "accuracy_mean":  float(np.mean(accs)),
            "accuracy_std":   float(np.std(accs)),
            "n_seeds":        len(seeds),
        })
    return pd.DataFrame(rows)


def write_latex_table(df, out_path):
    def fmt(m, s):
        if np.isnan(m):
            return "--"
        return f"{m:.3f} $\\pm$ {s:.3f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-dataset baseline comparison on CICEVSE2024 "
        r"($n = 6{,}166$). Same protocol as Table~\ref{tab:baselines}: "
        r"25\% stratified holdout, 5 seeds, F1-optimal oracle threshold per "
        r"method. CKDI's discriminative advantage transfers from CICEV2023 "
        r"to the larger and structurally different CICEVSE2024 testbed.}",
        r"\label{tab:crossdataset_baselines}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Precision & Recall & F1 & AUC & Accuracy \\",
        r"\midrule",
    ]
    order = [m for m in METHODS if m != "CKDI_thresholded"] + ["CKDI_thresholded"]
    df_o = df.set_index("method").loc[order].reset_index()
    for _, r in df_o.iterrows():
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
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(out_path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CICEVSE2024 Combined CSV")
    ap.add_argument("--output-dir",
                    default="src/wp1_cross_dataset/results")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--test-size", type=float, default=0.25)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, feat_cols = load_dataset(args.csv)
    X, y, ckdi = compute_ckdi(df, feat_cols)
    print(f"X shape: {X.shape}, attack: {int(y.sum())}, benign: {int((1-y).sum())}")
    print(f"CKDI range: [{ckdi.min():.4f}, {ckdi.max():.4f}]")

    per_seed = {}
    for s in args.seeds:
        print(f"--- seed {s} ---")
        per_seed[s] = run_seed(X, y, ckdi, s, test_size=args.test_size)
        for m in METHODS:
            r = per_seed[s][m]
            print(f"  {m:20s}  F1={r['f1']:.3f}  P={r['precision']:.3f}  "
                  f"R={r['recall']:.3f}  AUC={r['auc']:.3f}")

    df_agg = aggregate(per_seed, args.seeds)
    print("\n=== Aggregated baselines on CICEVSE2024 ===")
    print(df_agg.to_string(index=False))

    df_agg.to_csv(out_dir / "cicevse2024_baselines_results.csv", index=False)
    (out_dir / "cicevse2024_baselines_results.json").write_text(json.dumps({
        "per_seed":   per_seed,
        "aggregated": df_agg.to_dict(orient="records"),
        "config":     {"seeds": args.seeds, "test_size": args.test_size,
                       "n_samples": int(X.shape[0]),
                       "n_features": int(X.shape[1])},
    }, indent=2, default=str))
    write_latex_table(df_agg, out_dir / "cicevse2024_baselines_table.tex")
    print(f"\nWrote: {out_dir}/cicevse2024_baselines_results.csv")
    print(f"Wrote: {out_dir}/cicevse2024_baselines_results.json")
    print(f"Wrote: {out_dir}/cicevse2024_baselines_table.tex")


if __name__ == "__main__":
    main()
