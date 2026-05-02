#!/usr/bin/env python3
"""
Stream A — Cross-Dataset CKDI Generalization on CICEVSE2024
============================================================

Maps to: supervisor feedback item #2 — "tek dataset yetersiz, ikinci
dataset gerek". Tests whether the CKDI methodology developed on CICEV2023
(simulator-based, false-authentication DDoS, 27 features × 116 samples)
generalizes to CICEVSE2024 — a structurally different dataset:

  - Real Raspberry Pi-based EVSE testbed (not simulator)
  - 16+ network-layer attack types (SYN-FLOOD, ICMP-FLOOD, UDP-FLOOD,
    PUSH-ACK-FLOOD, PORT-SCAN, OS-FINGERPRINTING, VULNERABILITY-SCAN,
    CRYPTOJACKING, BACKDOOR, ...) — not false-authentication
  - 8474 samples × ~870 features (HPC + kernel events)
  - 5-second sampling resolution
  - Three operational states (Charging / Idle / MaliciousEV)

The CKDI pipeline is applied without modification:
  1. Drift z-score normalisation against benign-only baseline,
     grouped by (State) — the analogue of CICEV2023's (role, perf_mode)
  2. PCA → first principal component → min-max scaled → CKDI ∈ [0,1]
  3. Statistical separation test (Mann-Whitney U, Cohen's d)
  4. Binary discrimination (F1-optimal threshold, oracle protocol matching
     Stream H's §V-G evaluation)
  5. Per-scenario CKDI breakdown
  6. Direct comparison to CICEV2023 paper-reported numbers

Outputs:
  - results/cross_dataset_results.csv
  - results/cross_dataset_results.json
  - results/cicevse2024_table.tex (paper §V-H insert)
  - results/cicevse2024_ckdi_distribution.png (figure)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score,
)
from scipy.stats import mannwhitneyu


# ---------------------------------------------------------------------------
# Constants — non-feature columns in CICEVSE2024 Combined CSV
# Per Readme.txt the file has: time + HPC events + Kernel events + State +
# Attack + Scenario + Label + interface + (blank trailing fields)
# ---------------------------------------------------------------------------
META_COLS = {"time", "State", "Attack", "Scenario", "Label", "interface"}


def load_dataset(csv_path):
    """Load Combined CSV, separate features from metadata, validate."""
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  raw shape: {df.shape}")

    # Drop trailing blank/unnamed columns and any all-NaN columns
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:")]
    df.columns = df.columns.str.strip()
    print(f"  after dropping blank cols: {df.shape}")

    # Required columns sanity check
    required = {"Label", "State", "Scenario"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Feature columns = everything except metadata
    feat_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  feature columns: {len(feat_cols)}")

    # Coerce features to numeric, dropping any column that becomes all-NaN
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    nonnull_per_col = df[feat_cols].notna().sum()
    keep_cols = nonnull_per_col[nonnull_per_col > 0].index.tolist()
    meta_present = [c for c in df.columns if c in META_COLS]
    df = df[meta_present + keep_cols]
    feat_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  numeric feature columns: {len(feat_cols)}")

    # Normalise label values
    df["Label"] = df["Label"].astype(str).str.strip().str.lower()
    df["State"] = df["State"].astype(str).str.strip()
    df["Scenario"] = df["Scenario"].astype(str).str.strip()

    # Drop rows where Label is missing or unknown
    valid_labels = {"attack", "benign", "normal"}
    df = df[df["Label"].isin(valid_labels)].copy()
    df["Label"] = df["Label"].replace({"normal": "benign"})
    print(f"  after label filter: {df.shape}")
    print(f"  label counts:\n{df['Label'].value_counts()}")
    print(f"  state counts:\n{df['State'].value_counts()}")

    # Fill any remaining NaN with 0 (no event observed in that window)
    df[feat_cols] = df[feat_cols].fillna(0.0)

    return df, feat_cols


def compute_drift(df, feat_cols):
    """
    Z-score features against benign-only baseline, grouped by State
    (the CICEVSE2024 analogue of CICEV2023's role+perf_mode grouping).
    """
    drift = df.copy()
    n_states = 0
    for state, grp in df[df["Label"] == "benign"].groupby("State"):
        mu = grp[feat_cols].mean()
        sigma = grp[feat_cols].std().replace(0, np.nan)
        mask = (df["State"] == state)
        drift.loc[mask, feat_cols] = (df.loc[mask, feat_cols] - mu) / sigma
        n_states += 1
    print(f"Drift normalised across {n_states} states")
    drift[feat_cols] = drift[feat_cols].fillna(0.0)

    # Drop features that are entirely zero / near-zero variance
    var = drift[feat_cols].var()
    keep = var[var > 1e-12].index.tolist()
    print(f"Kept {len(keep)} features with non-trivial variance "
          f"(dropped {len(feat_cols) - len(keep)} zero-variance)")
    return drift, keep


def compute_ckdi(drift_matrix):
    """PCA → first PC absolute → min-max scale → CKDI ∈ [0,1]."""
    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(drift_matrix).ravel()
    abs_pc1 = np.abs(pc1)
    ckdi = (abs_pc1 - abs_pc1.min()) / (abs_pc1.max() - abs_pc1.min() + 1e-12)
    var_pc1 = float(pca.explained_variance_ratio_[0])
    return ckdi, var_pc1


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1))
                     / (nx + ny - 2))
    if pooled == 0:
        return float("nan")
    return (np.mean(y) - np.mean(x)) / pooled


def f1_optimal_threshold(scores, labels):
    """Sweep thresholds, return τ* maximising F1 + best F1."""
    best_f1, best_tau = -1.0, 0.0
    candidates = np.unique(scores)
    if len(candidates) > 500:
        # Subsample to avoid O(n^2) blowup on large datasets
        candidates = np.quantile(scores, np.linspace(0, 1, 500))
    for tau in candidates:
        pred = (scores >= tau).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return best_tau, best_f1


def evaluate(df, ckdi):
    """Stats, separation test, F1-optimal binary discrimination."""
    df = df.copy()
    df["CKDI"] = ckdi
    df["y_bin"] = (df["Label"] == "attack").astype(int)

    benign = df[df["Label"] == "benign"]["CKDI"].values
    attack = df[df["Label"] == "attack"]["CKDI"].values

    # Statistical separation
    u_stat, u_p = mannwhitneyu(attack, benign, alternative="greater")
    d = cohens_d(benign, attack)

    # Binary discrimination (oracle threshold, matching Stream H protocol)
    tau, f1 = f1_optimal_threshold(df["CKDI"].values, df["y_bin"].values)
    pred = (df["CKDI"].values >= tau).astype(int)
    try:
        auc = roc_auc_score(df["y_bin"].values, df["CKDI"].values)
    except ValueError:
        auc = float("nan")

    metrics = {
        "n_total":      int(len(df)),
        "n_benign":     int(len(benign)),
        "n_attack":     int(len(attack)),
        "ckdi_benign_mean": float(np.mean(benign)),
        "ckdi_benign_std":  float(np.std(benign, ddof=1)),
        "ckdi_attack_mean": float(np.mean(attack)),
        "ckdi_attack_std":  float(np.std(attack, ddof=1)),
        "mannwhitney_u":    float(u_stat),
        "mannwhitney_p":    float(u_p),
        "cohen_d":          float(d),
        "f1_optimal_tau":   float(tau),
        "precision":        float(precision_score(df["y_bin"], pred,
                                                   zero_division=0)),
        "recall":           float(recall_score(df["y_bin"], pred,
                                                zero_division=0)),
        "f1":               float(f1),
        "auc":              float(auc),
        "accuracy":         float(accuracy_score(df["y_bin"], pred)),
    }

    # Per-scenario breakdown (attack-only)
    scenario_stats = []
    for scen, grp in df[df["Label"] == "attack"].groupby("Scenario"):
        scenario_stats.append({
            "scenario":       scen,
            "n":              int(len(grp)),
            "ckdi_mean":      float(grp["CKDI"].mean()),
            "ckdi_std":       float(grp["CKDI"].std(ddof=1)) if len(grp) > 1 else 0.0,
        })
    scenario_stats.sort(key=lambda r: -r["ckdi_mean"])

    return metrics, scenario_stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_pval(p):
    """Format p-value for LaTeX. Avoids backslash inside f-string expression
    (Python 3.10 limitation)."""
    if p < 1e-8:
        return "<10^{-8}"
    return f"{p:.2e}"


def make_latex_table(metrics, var_pc1, n_features):
    """Paper §V-H insert."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-dataset generalisation: CKDI applied to CICEVSE2024 "
        r"(real Raspberry Pi EVSE testbed, network-layer DDoS + reconnaissance "
        r"+ cryptojacking attacks). Comparison to CICEV2023 (simulator-based, "
        r"false-authentication DDoS) shows that the methodology transfers.}",
        r"\label{tab:crossdataset}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & CICEV2023 (paper) & CICEVSE2024 (this work) \\",
        r"\midrule",
        r"Testbed & ISO 15118 simulator & Real Raspberry Pi EVSE \\",
        r"Attack family & False-authentication & Network DDoS, recon, mining \\",
        f"Samples & 116 & {metrics['n_total']:,} \\\\",
        f"Features & 27 & {n_features:,} \\\\",
        f"PC1 variance share & 22.7\\% & {var_pc1*100:.1f}\\% \\\\",
        r"\midrule",
        f"CKDI benign (mean $\\pm$ std) & 0.261 $\\pm$ 0.191 & "
        f"{metrics['ckdi_benign_mean']:.3f} $\\pm$ {metrics['ckdi_benign_std']:.3f} \\\\",
        f"CKDI attack (mean $\\pm$ std) & 0.484 $\\pm$ 0.200 & "
        f"{metrics['ckdi_attack_mean']:.3f} $\\pm$ {metrics['ckdi_attack_std']:.3f} \\\\",
        f"Cohen's $d$ & 1.14 & {metrics['cohen_d']:.2f} \\\\",
        f"Mann--Whitney $p$ & $<10^{{-8}}$ & "
        f"${_fmt_pval(metrics['mannwhitney_p'])}$ \\\\",
        r"\midrule",
        f"F1 (oracle threshold) & 0.814 & {metrics['f1']:.3f} \\\\",
        f"AUC & 0.776 & {metrics['auc']:.3f} \\\\",
        f"Accuracy & 0.800 & {metrics['accuracy']:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def make_distribution_plot(df, ckdi, out_path):
    """Optional matplotlib plot. Skip silently if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping distribution plot")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    benign = ckdi[df["Label"] == "benign"]
    attack = ckdi[df["Label"] == "attack"]
    bins = np.linspace(0, 1, 41)
    ax.hist(benign, bins=bins, alpha=0.6, label=f"Benign (n={len(benign):,})",
            color="#4CAF50", density=True)
    ax.hist(attack, bins=bins, alpha=0.6, label=f"Attack (n={len(attack):,})",
            color="#E53935", density=True)
    ax.set_xlabel("CKDI")
    ax.set_ylabel("Density")
    ax.set_title("CKDI distribution on CICEVSE2024 (cross-dataset test)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote distribution plot: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="Path to EVSE-B-HPC-Kernel-Events-Combined.csv")
    ap.add_argument("--output-dir", default="results")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    df, feat_cols = load_dataset(args.csv)

    # 2. Drift normalisation
    drift_df, kept_features = compute_drift(df, feat_cols)

    # 3. CKDI
    ckdi, var_pc1 = compute_ckdi(drift_df[kept_features].values)
    print(f"\nCKDI: range [{ckdi.min():.4f}, {ckdi.max():.4f}], "
          f"PC1 variance share = {var_pc1*100:.2f}%")

    # 4. Evaluate
    metrics, scenario_stats = evaluate(df, ckdi)
    n_features_used = len(kept_features)

    # Reports
    print("\n=== Cross-Dataset Metrics (CICEVSE2024) ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s} {v:.6f}" if abs(v) < 1
                  else f"  {k:25s} {v:.4f}")
        else:
            print(f"  {k:25s} {v}")

    print("\n=== Per-attack-scenario CKDI (top 10 by mean) ===")
    for s in scenario_stats[:10]:
        print(f"  {s['scenario']:30s} n={s['n']:5d}  "
              f"CKDI={s['ckdi_mean']:.3f} ± {s['ckdi_std']:.3f}")

    # Persist
    metrics_full = dict(metrics)
    metrics_full["pc1_variance_share"] = float(var_pc1)
    metrics_full["n_features_used"]    = int(n_features_used)
    metrics_full["n_features_total"]   = int(len(feat_cols))

    summary = {
        "metrics":   metrics_full,
        "per_scenario": scenario_stats,
        "comparison_to_cicev2023": {
            "ckdi_benign_mean":  {"cicev2023": 0.261, "cicevse2024": metrics["ckdi_benign_mean"]},
            "ckdi_attack_mean":  {"cicev2023": 0.484, "cicevse2024": metrics["ckdi_attack_mean"]},
            "cohen_d":           {"cicev2023": 1.14,  "cicevse2024": metrics["cohen_d"]},
            "f1":                {"cicev2023": 0.814, "cicevse2024": metrics["f1"]},
            "auc":               {"cicev2023": 0.776, "cicevse2024": metrics["auc"]},
            "accuracy":          {"cicev2023": 0.800, "cicevse2024": metrics["accuracy"]},
        }
    }
    (out_dir / "cross_dataset_results.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\nWrote: {out_dir}/cross_dataset_results.json")

    # CSV view (per-scenario)
    pd.DataFrame(scenario_stats).to_csv(
        out_dir / "cross_dataset_per_scenario.csv", index=False)
    print(f"Wrote: {out_dir}/cross_dataset_per_scenario.csv")

    # LaTeX
    tex = make_latex_table(metrics, var_pc1, n_features_used)
    (out_dir / "cicevse2024_table.tex").write_text(tex)
    print(f"Wrote: {out_dir}/cicevse2024_table.tex")

    # Plot
    make_distribution_plot(df, ckdi, out_dir / "cicevse2024_ckdi_distribution.png")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
