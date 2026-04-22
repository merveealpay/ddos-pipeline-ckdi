import os
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from scipy.optimize import minimize

import matplotlib.pyplot as plt


# ----------------------------
# CONFIG
# ----------------------------
PROCESSED_DATA_DIR = Path("Processed_Data")  # zip'i açtıktan sonra bu klasör burada olmalı
OUT_FIG_DIR = Path("figures")
OUT_RES_DIR = Path("results_outputs")

# Service mapping parameters (paper’daki analitik eşleme)
L_BASE = 1.0
MTTR_BASE = 1.0
DELTA = 1.0   # latency slope
ALPHA = 2.0   # availability decay
BETA  = 3.0   # RIF exponential-saturation rate (paper eq. 502)

LAMBDA_OPT = 1.0  # optimization objective correlation weight


# ----------------------------
# HELPERS
# ----------------------------
def safe_mean(arr):
    a = np.asarray(arr, dtype=float)
    a = a[a != -1.0]
    return float(np.mean(a)) if a.size else np.nan

def safe_std(arr):
    a = np.asarray(arr, dtype=float)
    a = a[a != -1.0]
    return float(np.std(a)) if a.size else np.nan

def find_final_dataset_files(base: Path):
    files = []
    for dirpath, _, filenames in os.walk(base):
        if "final_dataset.json" in filenames:
            files.append(Path(dirpath) / "final_dataset.json")
    return files

def parse_meta_from_path(fp: Path):
    # .../Processed_Data/<Scenario>/<Random_CS_...>/<Gaussian_...>/<TOP|STAT|RECORD>/<GS|CS>/final_dataset.json
    parts = fp.parts
    i = parts.index("Processed_Data")
    return {
        "scenario": parts[i+1],
        "random_cs": parts[i+2],
        "gaussian": parts[i+3],
        "perf_mode": parts[i+4],
        "role": parts[i+5],
    }

def extract_rows_from_final_dataset(fp: Path):
    meta = parse_meta_from_path(fp)
    with open(fp, "r") as f:
        d = json.load(f)

    rows = []
    for metric_type, metric_obj in d.items():               # branch / cycles / instructions
        for section, combos in metric_obj.items():          # common / exclusive / all
            for _, combo_val in combos.items():
                for cls in ("attack", "normal"):
                    if cls not in combo_val:
                        continue
                    cls_obj = combo_val[cls]
                    sampling_res = cls_obj.get("combined_sampling_resolution", np.nan)

                    data_point = cls_obj.get("data_point", {})  # symbol -> entity -> values[]
                    for symbol, ent_map in data_point.items():
                        for ent_id, values in ent_map.items():
                            rows.append({
                                **meta,
                                "metric_type": metric_type,
                                "section": section,
                                "class": cls,
                                "entity": ent_id,
                                "sampling_res": sampling_res,
                                "mean": safe_mean(values),
                                "std": safe_std(values),
                                "count": int(np.sum(np.asarray(values, dtype=float) != -1.0)),
                            })
    return rows


# ----------------------------
# PIPELINE
# ----------------------------
def main():
    if not PROCESSED_DATA_DIR.exists():
        raise FileNotFoundError(
            f"{PROCESSED_DATA_DIR} bulunamadı. Zip’i açıp içinde Processed_Data klasörü olduğundan emin ol."
        )

    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RES_DIR.mkdir(parents=True, exist_ok=True)

    final_files = find_final_dataset_files(PROCESSED_DATA_DIR)
    if not final_files:
        raise RuntimeError("final_dataset.json dosyası bulunamadı. Klasör yapısını kontrol et.")

    # 1) Parse all final_dataset.json into long table
    all_rows = []
    for fp in final_files:
        all_rows.extend(extract_rows_from_final_dataset(fp))
    raw_df = pd.DataFrame(all_rows)

    # 2) Aggregate to sample-level features: for each (scenario, random_cs, gaussian, perf_mode, role, class, entity)
    agg = raw_df.groupby(
        ["scenario", "random_cs", "gaussian", "perf_mode", "role", "class", "entity", "metric_type", "section"],
        as_index=False
    ).agg(
        mean_mean=("mean", "mean"),
        mean_std=("std", "mean"),
        mean_sampling=("sampling_res", "mean"),
    )

    wide_mean = agg.pivot_table(
        index=["scenario", "random_cs", "gaussian", "perf_mode", "role", "class", "entity"],
        columns=["metric_type", "section"],
        values="mean_mean",
    )
    wide_std = agg.pivot_table(
        index=["scenario", "random_cs", "gaussian", "perf_mode", "role", "class", "entity"],
        columns=["metric_type", "section"],
        values="mean_std",
    )
    wide_samp = agg.pivot_table(
        index=["scenario", "random_cs", "gaussian", "perf_mode", "role", "class", "entity"],
        columns=["metric_type", "section"],
        values="mean_sampling",
    )

    def flatten(df, suffix):
        df = df.copy()
        df.columns = [f"{a}_{b}_{suffix}" for a, b in df.columns]
        return df

    X = pd.concat([flatten(wide_mean, "mean"), flatten(wide_std, "std"), flatten(wide_samp, "samp")], axis=1).reset_index()

    meta_cols = ["scenario", "random_cs", "gaussian", "perf_mode", "role", "class", "entity"]
    feat_cols = [c for c in X.columns if c not in meta_cols]

    # 3) Baseline-referenced drift (normal baseline, per role+perf_mode)
    X_drift = X.copy()
    for (role, perf_mode), grp in X[X["class"] == "normal"].groupby(["role", "perf_mode"]):
        mu = grp[feat_cols].mean()
        sigma = grp[feat_cols].std().replace(0, np.nan)
        mask = (X["role"] == role) & (X["perf_mode"] == perf_mode)
        X_drift.loc[mask, feat_cols] = (X.loc[mask, feat_cols] - mu) / sigma

    X_drift[feat_cols] = X_drift[feat_cols].fillna(0.0)

    # 4) CKDI via PCA: use |PC1| then scale [0,1] (attack severity magnitude)
    pca = PCA(n_components=1, random_state=42)
    pc1 = pca.fit_transform(X_drift[feat_cols].values).ravel()
    ckdi_abs = np.abs(pc1)
    CKDI = (ckdi_abs - ckdi_abs.min()) / (ckdi_abs.max() - ckdi_abs.min() + 1e-9)
    X_drift["CKDI"] = CKDI

    # 5) Analytical service mapping (paper §IV, eqs. 478, 479, 502)
    Latency = L_BASE * (1 + DELTA * CKDI)              # paper eq. 478
    Availability = np.exp(-ALPHA * CKDI)               # paper eq. 479
    RIF = 1.0 - np.exp(-BETA * CKDI**2)                # paper eq. 502 (saturating)
    MTTR = MTTR_BASE * (1.0 + RIF)                     # derived for diagnostic plots only

    SDI = (Latency - L_BASE) / L_BASE
    ALR = 1 - Availability

    # 6) Optimize BDRI weights with SLSQP
    def bdri_from_w(w):
        return 1 - (w[0] * SDI + w[1] * ALR + w[2] * RIF)

    normal_mask = (X_drift["class"].values == "normal")

    def objective(w):
        b = bdri_from_w(w)
        var_norm = float(np.var(b[normal_mask]))
        corr = float(np.corrcoef(CKDI, b)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        return var_norm - LAMBDA_OPT * corr

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0, 1), (0, 1), (0, 1)]
    w0 = np.array([1/3, 1/3, 1/3], dtype=float)

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 300})
    w_opt = res.x

    BDRI_eq = bdri_from_w(w0)
    BDRI_opt = bdri_from_w(w_opt)

    # 7) Save summary CSVs for LaTeX tables
    bdri_compare = pd.DataFrame({
        "Method": ["Equal-weight", "Optimized"],
        "Var_BDRI_normal": [float(np.var(BDRI_eq[normal_mask])), float(np.var(BDRI_opt[normal_mask]))],
        "Corr_CKDI_BDRI": [
            float(np.corrcoef(CKDI, BDRI_eq)[0, 1]),
            float(np.corrcoef(CKDI, BDRI_opt)[0, 1]),
        ],
    })

    weights_df = pd.DataFrame({"w1_SDI": [w_opt[0]], "w2_ALR": [w_opt[1]], "w3_RIF": [w_opt[2]]})

    X_out = X_drift.copy()
    X_out["Latency"] = Latency
    X_out["Availability"] = Availability
    X_out["MTTR"] = MTTR
    X_out["BDRI_equal"] = BDRI_eq
    X_out["BDRI_opt"] = BDRI_opt

    X_out.to_csv(OUT_RES_DIR / "sample_level_ckdi_bdri.csv", index=False)
    bdri_compare.to_csv(OUT_RES_DIR / "bdri_comparison.csv", index=False)
    weights_df.to_csv(OUT_RES_DIR / "optimized_weights.csv", index=False)

    # 8) Figures (PDF) for Overleaf
    # Fig1: CKDI distribution (Normal vs Attack)
    plt.figure()
    plt.hist(X_out.loc[X_out["class"] == "normal", "CKDI"], bins=20, alpha=0.7, label="Normal")
    plt.hist(X_out.loc[X_out["class"] == "attack", "CKDI"], bins=20, alpha=0.7, label="Attack")
    plt.xlabel("CKDI"); plt.ylabel("Count"); plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "ckdi_distribution.pdf")
    plt.close()

    # Fig2: CKDI vs Availability
    plt.figure()
    plt.scatter(X_out["CKDI"], X_out["Availability"], s=18)
    plt.xlabel("CKDI"); plt.ylabel("Availability (modeled)")
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "ckdi_vs_availability.pdf")
    plt.close()

    # Fig3: CKDI vs Latency
    plt.figure()
    plt.scatter(X_out["CKDI"], X_out["Latency"], s=18)
    plt.xlabel("CKDI"); plt.ylabel("Latency (modeled)")
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "ckdi_vs_latency.pdf")
    plt.close()

    # Fig4: BDRI escalation (sorted CKDI)
    order = np.argsort(X_out["CKDI"].values)
    plt.figure()
    plt.plot(X_out["CKDI"].values[order], X_out["BDRI_opt"].values[order])
    plt.xlabel("CKDI (sorted)"); plt.ylabel("BDRI (optimized)")
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "bdri_escalation.pdf")
    plt.close()

    # Fig5: BDRI equal vs optimized (optional, strong for paper)
    plt.figure()
    plt.plot(X_out["CKDI"].values[order], X_out["BDRI_equal"].values[order], label="Equal-weight")
    plt.plot(X_out["CKDI"].values[order], X_out["BDRI_opt"].values[order], label="Optimized")
    plt.xlabel("CKDI (sorted)"); plt.ylabel("BDRI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIR / "bdri_equal_vs_opt.pdf")
    plt.close()

    print("DONE ✅")
    print(f"Figures: {OUT_FIG_DIR.resolve()}")
    print(f"Tables/CSVs: {OUT_RES_DIR.resolve()}")
    print("Optimized weights:", w_opt)


if __name__ == "__main__":
    main()